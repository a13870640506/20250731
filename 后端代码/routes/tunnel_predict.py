"""
隧道位移预测相关路由
对应前端TunnelPredict.vue
"""
from flask import request, jsonify
import os
import json
import traceback
import numpy as np
import torch
import joblib
from datetime import datetime

from routes import tunnel_predict_bp
from my_model import TimeSeriesTransformer

# 常量定义
MODEL_DIR = './models'
SCALER_DIR = './scalers'


def _build_sequence(poisson_ratio, friction_angle, cohesion, dilation_angle,
                    elastic_modulus, input_scaler, device):
    """根据单次输入构造伪时序数据并完成标准化"""
    # 构造4个时间步的参数矩阵
    params = np.array([
        poisson_ratio,
        friction_angle,
        cohesion,
        dilation_angle,
        elastic_modulus
    ], dtype=np.float32)
    values = np.tile(params, (4, 1))

    # 伪造时间间隔，首个时间步为0，其余为1
    deltas = np.array([0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

    # 合并后进行标准化
    features = np.hstack([values, deltas])
    norm = input_scaler.transform(features)
    values_norm = norm[:, :values.shape[1]]
    deltas_norm = norm[:, values.shape[1]:]

    # 构造注意力掩码并转为张量
    mask = np.ones(4, dtype=bool)
    values_tensor = torch.from_numpy(values_norm).unsqueeze(0).to(device)
    deltas_tensor = torch.from_numpy(deltas_norm).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    return values_tensor, deltas_tensor, mask_tensor


@tunnel_predict_bp.route('/models', methods=['GET'])
def get_available_models():
    """获取可用的模型列表"""
    try:
        # 查找所有可用模型
        model_list = []
        
        # 检查主模型目录
        if os.path.exists(os.path.join(MODEL_DIR, 'best_model.pth')):
            model_list.append({
                'name': '主模型',
                'path': MODEL_DIR,
                'date': datetime.fromtimestamp(os.path.getctime(os.path.join(MODEL_DIR, 'best_model.pth'))).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # 检查子模型目录
        for root, dirs, _ in os.walk(MODEL_DIR):
            for dir_name in dirs:
                if dir_name.startswith('model_'):
                    model_path = os.path.join(root, dir_name)
                    model_file = os.path.join(model_path, 'best_model.pth')
                    
                    if os.path.exists(model_file):
                        # 获取相对路径
                        rel_path = os.path.relpath(model_path, MODEL_DIR)
                        model_list.append({
                            'name': dir_name,
                            'path': os.path.join(MODEL_DIR, rel_path),
                            'date': datetime.fromtimestamp(os.path.getctime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
                        })
        
        # 按日期排序，最新的在前
        model_list.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': model_list
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'获取模型列表失败: {str(e)}'
        })

@tunnel_predict_bp.route('/predict', methods=['POST'])
def predict_displacement():
    """根据围岩参数预测隧道位移"""
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请提供预测参数'
            })
        
        model_path = data.get('model_path')
        input_params = data.get('input_params')
        
        if not model_path:
            return jsonify({
                'success': False,
                'message': '请提供模型路径'
            })
        
        if not input_params:
            return jsonify({
                'success': False,
                'message': '请提供围岩参数'
            })
        
        # 检查必要的输入参数
        required_params = ['poisson_ratio', 'friction_angle', 'cohesion', 'dilation_angle', 'elastic_modulus']
        for param in required_params:
            if param not in input_params:
                return jsonify({
                    'success': False,
                    'message': f'缺少必要的输入参数: {param}'
                })
        
        # 加载标准化器
        try:
            # 首先尝试从SCALER_DIR加载
            input_scaler_path = os.path.join(SCALER_DIR, 'input_scaler.pkl')
            output_scaler_path = os.path.join(SCALER_DIR, 'output_scaler.pkl')
            
            # 如果标准化器不存在，尝试从模型目录加载
            if not os.path.exists(input_scaler_path) or not os.path.exists(output_scaler_path):
                print(f"在{SCALER_DIR}中未找到标准化器，尝试从模型目录加载")
                input_scaler_path = os.path.join(model_path, 'input_scaler.pkl')
                output_scaler_path = os.path.join(model_path, 'output_scaler.pkl')
                
                # 如果模型目录也没有，尝试在models_正演目录查找
                if not os.path.exists(input_scaler_path) or not os.path.exists(output_scaler_path):
                    print("在模型目录中未找到标准化器，尝试从models_正演目录加载")
                    input_scaler_path = os.path.join('./models_正演', 'input_scaler.pkl')
                    output_scaler_path = os.path.join('./models_正演', 'output_scaler.pkl')
            
            print(f"加载输入标准化器: {input_scaler_path}")
            print(f"加载输出标准化器: {output_scaler_path}")
            
            # 加载标准化器
            input_scaler = joblib.load(input_scaler_path)
            output_scaler = joblib.load(output_scaler_path)
            
            print("标准化器加载成功")
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'加载标准化器失败: {str(e)}'
            })
        
        # 加载模型
        try:
            # 确定模型文件路径
            model_file = os.path.join(model_path, 'best_model.pth')
            print(f"尝试加载模型: {model_file}")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_file):
                return jsonify({
                    'success': False,
                    'message': f'模型文件不存在: {model_file}'
                })
            
            # 加载模型参数
            model_params = {
                'input_dim': 7,  # 默认值
                'd_model': 256,
                'nhead': 4,
                'num_layers': 2
            }
            
            # 尝试从模型目录加载模型参数
            params_file = os.path.join(model_path, 'model_params.json')
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    model_params.update(json.load(f))
                    print(f"从{params_file}加载模型参数: {model_params}")
            else:
                print(f"使用默认模型参数: {model_params}")
            
            # 创建模型实例
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {device}")
            
            model = TimeSeriesTransformer(
                input_dim=model_params['input_dim'],
                d_model=model_params['d_model'],
                nhead=model_params['nhead'],
                num_layers=model_params['num_layers']
            ).to(device)
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            print("模型加载成功并设置为评估模式")
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'加载模型失败: {str(e)}'
            })
        
        # 准备输入数据
        try:
            # 从输入参数中提取值并构造伪时序序列
            poisson_ratio = float(input_params['poisson_ratio'])
            friction_angle = float(input_params['friction_angle'])
            cohesion = float(input_params['cohesion'])
            dilation_angle = float(input_params['dilation_angle'])
            elastic_modulus = float(input_params['elastic_modulus'])

            values_tensor, deltas_tensor, mask_tensor = _build_sequence(
                poisson_ratio,
                friction_angle,
                cohesion,
                dilation_angle,
                elastic_modulus,
                input_scaler,
                device
            )
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'准备输入数据失败: {str(e)}'
            })
        
        # 执行预测
        try:
            with torch.no_grad():
                output = model(values_tensor, deltas_tensor, mask_tensor)
                
            # 转换为numpy数组
            output_np = output.cpu().numpy()
            
            # 反标准化输出
            prediction = output_scaler.inverse_transform(output_np)[0]  # 取第一个样本
            
            # 提取预测结果
            crown_settlement1 = float(prediction[0])  # 拱顶下沉1
            crown_settlement2 = float(prediction[1])  # 拱顶下沉2
            convergence1 = float(prediction[2])       # 周边收敛1
            convergence2 = float(prediction[3])       # 周边收敛2
            foot_settlement = float(prediction[4])    # 拱脚下沉
            
            # 构建响应
            result = {
                'crown_settlement1': crown_settlement1,
                'crown_settlement2': crown_settlement2,
                'convergence1': convergence1,
                'convergence2': convergence2,
                'foot_settlement': foot_settlement,
                'metrics': {
                    'r2': 0.95,  # 模型的整体R2值（示例）
                    'mse': 0.01,  # 模型的整体MSE值（示例）
                    'mape': 5.2   # 模型的整体MAPE值（示例）
                }
            }
            
            return jsonify({
                'success': True,
                'message': '预测成功',
                'data': result
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'预测失败: {str(e)}'
            })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'预测请求处理失败: {str(e)}'
        })
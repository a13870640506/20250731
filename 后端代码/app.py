"""
基于Transformer神经网络的隧道数字孪生系统API
主服务器程序
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import traceback
import numpy as np
import pandas as pd

# 导入自定义模块
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from optimizer import ModelOptimizer
from utils import download_file_handler

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置常量
MODEL_DIR = './models'
DATA_DIR = './data'
SCALER_DIR = './scalers'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# 创建各模块处理器
data_processor = DataProcessor(data_dir=DATA_DIR, scaler_dir=SCALER_DIR)
model_trainer = ModelTrainer(model_dir=MODEL_DIR)
model_optimizer = ModelOptimizer(model_dir=MODEL_DIR)


# 首页
@app.route('/')
def home():
    return "基于Transformer神经网络的隧道数字孪生系统API"


# 数据集上传和处理接口
@app.route('/transformer/upload', methods=['POST'])
def upload_dataset():
    try:
        # 获取上传的文件
        input_file = request.files.get('input_file')

        if not input_file:
            return jsonify({
                'success': False,
                'message': '请上传数据文件'
            })

        # 获取数据集划分比例
        train_ratio = float(request.form.get('train_ratio', 0.7))
        val_ratio = float(request.form.get('val_ratio', 0.1))
        test_ratio = float(request.form.get('test_ratio', 0.2))

        # 验证比例之和是否为1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            return jsonify({
                'success': False,
                'message': '数据集划分比例之和必须为1'
            })

        # 处理数据集
        result = data_processor.process_data(
            input_file,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        return jsonify({
            'success': True,
            'message': '数据处理成功',
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'数据处理失败: {str(e)}'
        })


# 获取数据处理结果列表
@app.route('/transformer/datasets', methods=['GET'])
def get_datasets():
    try:
        # 查找所有数据处理目录
        dataset_dirs = []
        for item in os.listdir(DATA_DIR):
            if item.startswith('processed_') and os.path.isdir(os.path.join(DATA_DIR, item)):
                dataset_dirs.append(item)
        
        # 按时间戳倒序排序（最新的在前面）
        dataset_dirs.sort(reverse=True)
        
        datasets = []
        for dir_name in dataset_dirs:
            dataset_path = os.path.join(DATA_DIR, dir_name)
            info_path = os.path.join(dataset_path, 'data_info.txt')
            info = {}
            
            if os.path.exists(info_path):
                # 尝试多种编码格式
                encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1']
                
                for encoding in encodings:
                    try:
                        with open(info_path, 'r', encoding=encoding) as f:
                            lines = f.readlines()
                            for line in lines:
                                if ':' in line:
                                    key, value = line.strip().split(':', 1)
                                    info[key.strip()] = value.strip()
                        print(f"成功使用 {encoding} 编码读取数据集信息")
                        break  # 成功读取，跳出循环
                    except Exception as e:
                        print(f"尝试使用 {encoding} 编码读取失败: {str(e)}")
                        # 继续尝试下一种编码
            
            # 提取时间戳
            try:
                # 目录名格式应该是 processed_YYYYMMDD_HHMMSS
                if '_' in dir_name:
                    parts = dir_name.split('_')
                    if len(parts) >= 3:  # 应该至少有三部分: processed, date, time
                        date_part = parts[1]
                        time_part = parts[2]
                        # 组合成合法的时间戳格式
                        timestamp_str = f"{date_part}_{time_part}"
                        # 解析时间戳
                        date = pd.to_datetime(timestamp_str, format='%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date = dir_name
                else:
                    date = dir_name
            except Exception as e:
                print(f"日期解析失败: {str(e)}")
                date = dir_name
            
            datasets.append({
                'path': dataset_path,
                'name': dir_name,
                'date': date,
                'info': info
            })
        
        return jsonify({
            'success': True,
            'data': datasets
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'获取数据集列表失败: {str(e)}'
        })


# 模型训练接口
@app.route('/transformer/train', methods=['POST'])
def train_model_api():
    try:
        # 获取训练参数
        data_path = request.form.get('data_path')
        model_name = request.form.get('model_name', 'transformer')
        epochs = int(request.form.get('epochs', 300))
        batch_size = int(request.form.get('batch_size', 16))
        learning_rate = float(request.form.get('learning_rate', 0.0001))
        weight_decay = float(request.form.get('weight_decay', 0.0001))
        d_model = int(request.form.get('d_model', 256))
        nhead = int(request.form.get('nhead', 4))
        num_layers = int(request.form.get('num_layers', 2))
        dropout_value = request.form.get('dropout')
        dropout = 0.05  # 默认值
        if dropout_value and dropout_value != 'undefined':
            try:
                dropout = float(dropout_value)
            except ValueError:
                pass  # 如果转换失败，使用默认值
        model_save_path = request.form.get('model_save_path')
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'message': '无效的数据路径'
            })
            
        print(f"开始训练模型，参数：")
        print(f"- 数据路径: {data_path}")
        print(f"- 模型名称: {model_name}")
        print(f"- 训练轮次: {epochs}")
        print(f"- 批次大小: {batch_size}")
        print(f"- 学习率: {learning_rate}")
        print(f"- 权重衰减: {weight_decay}")
        print(f"- 模型维度: {d_model}")
        print(f"- 注意力头数: {nhead}")
        print(f"- Transformer层数: {num_layers}")
        print(f"- Dropout率: {dropout}")
        print(f"- 模型保存路径: {model_save_path}")
        
        # 调用模型训练器进行训练
        result = model_trainer.train(
            data_path=data_path,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            model_save_path=model_save_path
        )
        
        return jsonify({
            'success': True,
            'message': '模型训练成功',
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'模型训练失败: {str(e)}'
        })


# 模型预测接口
@app.route('/transformer/predict', methods=['POST'])
def predict():
    try:
        # 获取预测参数
        data_file = request.files.get('data_file')
        model_path = request.form.get('model_path')
        
        if not data_file:
            return jsonify({
                'success': False,
                'message': '请上传预测数据'
            })
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'message': '无效的模型路径'
            })
        
        # 调用模型训练器进行预测
        result = model_trainer.predict(
            data_file=data_file,
            model_path=model_path
        )
        
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


# 超参数优化接口
@app.route('/transformer/optimize', methods=['POST'])
def optimize_params():
    try:
        # 获取优化参数
        n_trials = int(request.form.get('n_trials', 50))
        epochs_per_trial = int(request.form.get('epochs_per_trial', 30))
        batch_size = int(request.form.get('batch_size', 16))
        data_path = request.form.get('data_path')
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'message': '无效的数据路径'
            })
        
        # 获取参数范围
        d_model_values = json.loads(request.form.get('d_model_values', '[64, 128, 256]'))
        nhead_values = json.loads(request.form.get('nhead_values', '[1, 2, 4]'))
        num_layers_min = int(request.form.get('num_layers_min', 1))
        num_layers_max = int(request.form.get('num_layers_max', 4))
        lr_min = float(request.form.get('lr_min', 1e-5))
        lr_max = float(request.form.get('lr_max', 1e-3))
        weight_decay_min = float(request.form.get('weight_decay_min', 1e-8))
        weight_decay_max = float(request.form.get('weight_decay_max', 1e-4))
        
        # 构建优化参数
        optimize_params = {
            'n_trials': n_trials,
            'epochs_per_trial': epochs_per_trial,
            'batch_size': batch_size,
            'data_path': data_path,
            'd_model_values': d_model_values,
            'nhead_values': nhead_values,
            'num_layers_min': num_layers_min,
            'num_layers_max': num_layers_max,
            'lr_min': lr_min,
            'lr_max': lr_max,
            'weight_decay_min': weight_decay_min,
            'weight_decay_max': weight_decay_max
        }
        
        # 调用优化器执行优化
        result = model_optimizer.optimize(optimize_params)
        
        return jsonify({
            'success': True,
            'message': '超参数优化成功',
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'超参数优化失败: {str(e)}'
        })


# 获取优化历史记录接口
@app.route('/transformer/optimization_history', methods=['GET'])
def get_optimization_history():
    try:
        # 调用优化器获取历史记录
        history = model_optimizer.get_optimization_history()
        
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'获取优化历史记录失败: {str(e)}'
        })


# 获取优化结果详情接口
@app.route('/transformer/optimization_result/<opt_id>', methods=['GET'])
def get_optimization_result(opt_id):
    try:
        # 调用优化器获取优化结果
        result = model_optimizer.get_optimization_result(opt_id)
        
        if not result:
            return jsonify({
                'success': False,
                'message': f'找不到优化记录: {opt_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'获取优化结果详情失败: {str(e)}'
        })


# 文件下载接口
@app.route('/download', methods=['GET'])
def download_file():
    try:
        file_path = request.args.get('path')
        return download_file_handler(file_path, allowed_dirs=[DATA_DIR, MODEL_DIR])
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'文件下载失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
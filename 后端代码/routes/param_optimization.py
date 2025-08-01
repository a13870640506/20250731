"""
超参数优化相关路由
对应前端ParamOptimization.vue
"""
from flask import request, jsonify
import os
import json
import traceback
import pandas as pd

from routes import param_optimization_bp
from optimizer import ModelOptimizer

# 获取模型优化器实例
model_optimizer = ModelOptimizer(model_dir='./models')

# 超参数优化接口
@param_optimization_bp.route('/optimize', methods=['POST'])
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
@param_optimization_bp.route('/optimization_history', methods=['GET'])
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
@param_optimization_bp.route('/optimization_result/<opt_id>', methods=['GET'])
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

# 获取数据集列表接口
@param_optimization_bp.route('/datasets', methods=['GET'])
def get_datasets():
    try:
        # 数据目录
        DATA_DIR = './data'
        
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
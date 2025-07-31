"""
超参数优化模块
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import base64
from torch.utils.data import DataLoader
from my_dataset import GeotechDataset
from my_model import TimeSeriesTransformer
from my_train import train_model
from my_inference import validate_model


class ModelOptimizer:
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def optimize(self, params):
        """
        执行超参数优化
        
        参数:
            params: 包含优化参数的字典
            
        返回:
            优化结果字典
        """
        # 解析参数
        n_trials = params.get('n_trials', 50)
        epochs_per_trial = params.get('epochs_per_trial', 30)
        batch_size = params.get('batch_size', 16)
        data_path = params.get('data_path')
        d_model_values = params.get('d_model_values', [64, 128, 256])
        nhead_values = params.get('nhead_values', [1, 2, 4])
        num_layers_min = params.get('num_layers_min', 1)
        num_layers_max = params.get('num_layers_max', 4)
        lr_min = params.get('lr_min', 1e-5)
        lr_max = params.get('lr_max', 1e-3)
        weight_decay_min = params.get('weight_decay_min', 1e-8)
        weight_decay_max = params.get('weight_decay_max', 1e-4)
        
        # 创建带时间戳的文件夹保存优化结果，使用标准格式
        now = pd.Timestamp.now()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        timestamp = f"{date_part}_{time_part}"
        
        optimization_dir = os.path.join(self.model_dir, f'optimization_{timestamp}')
        os.makedirs(optimization_dir, exist_ok=True)
        
        # 创建优化ID
        optimization_id = f'opt_{timestamp}'
        
        # 加载数据集
        train_data = np.load(os.path.join(data_path, 'train_data.npz'))
        val_data = np.load(os.path.join(data_path, 'val_data.npz'))
        
        # 创建数据集和数据加载器
        train_dataset = GeotechDataset(
            train_data['X_train'], 
            train_data['delta_train'], 
            train_data['mask_train'], 
            train_data['y_train']
        )
        
        val_dataset = GeotechDataset(
            val_data['X_val'], 
            val_data['delta_val'], 
            val_data['mask_val'], 
            val_data['y_val']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 计算输入维度
        input_dim = train_data['X_train'].shape[2] + 2  # +2是因为模型会在前向传播中添加time_deltas和abs_time
        
        # 定义贝叶斯优化的目标函数
        def objective(trial):
            # 从trial中获取超参数
            d_model = trial.suggest_categorical('d_model', d_model_values)
            nhead = trial.suggest_categorical('nhead', nhead_values)
            num_layers = trial.suggest_int('num_layers', num_layers_min, num_layers_max)
            lr = trial.suggest_loguniform('lr', lr_min, lr_max)
            weight_decay = trial.suggest_loguniform('weight_decay', weight_decay_min, weight_decay_max)
            
            print(f"Trial {trial.number}: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, lr={lr:.6e}, weight_decay={weight_decay:.6e}")
            
            # 创建模型和优化器
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TimeSeriesTransformer(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # 训练循环
            for epoch in range(epochs_per_trial):  # 较小轮次快速搜索
                train_model(model, train_loader, criterion, optimizer)
                val_loss, _, _ = validate_model(model, val_loader, criterion)
            
            return float(val_loss)
        
        # 创建Optuna研究对象
        study = optuna.create_study(direction="minimize")
        
        # 运行优化
        study.optimize(objective, n_trials=n_trials)
        
        # 获取最佳参数
        best_params = {
            'd_model': int(study.best_params['d_model']),
            'nhead': int(study.best_params['nhead']),
            'num_layers': int(study.best_params['num_layers']),
            'lr': float(study.best_params['lr']),
            'weight_decay': float(study.best_params['weight_decay']),
            # batch_size列已删除
            'dropout': float(0.05),  # 固定值
            'accumulate_grad_batches': int(1)  # 固定值
        }
        
        # 获取优化指标
        metrics = {
            'val_loss': float(study.best_value),
            'best_trial': int(study.best_trial.number)  # 前端会+1显示
        }
        
        # 获取所有试验数据
        trials_df = study.trials_dataframe()
        
        # 将DataFrame中的数据转换为Python原生类型
        trials_data = []
        for _, row in trials_df.iterrows():
            # 确保所有字段都有正确的值
            trial_data = {
                'number': int(row['number']),
                'value': float(row['value']),
                'd_model': int(row['params_d_model']) if 'params_d_model' in row else None,
                'nhead': int(row['params_nhead']) if 'params_nhead' in row else None,
                'num_layers': int(row['params_num_layers']) if 'params_num_layers' in row else None,
                'lr': float(row['params_lr']) if 'params_lr' in row else None,
                'weight_decay': float(row['params_weight_decay']) if 'params_weight_decay' in row else None,
                # batch_size列已删除
                'dropout': float(0.05),
                'accumulate_grad_batches': int(1)
            }
            trials_data.append(trial_data)
        
        # 保存优化结果
        result = {
            'id': optimization_id,
            'date': timestamp,
            'best_params': best_params,
            'metrics': metrics,
            'trials': trials_data
        }
        
        with open(os.path.join(optimization_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
        # 生成优化过程曲线图
        self._plot_optimization_curve(trials_df, n_trials, optimization_dir)
        
        # 保存CSV文件
        trials_df.to_csv(os.path.join(optimization_dir, 'optuna_trials.csv'), index=False, encoding='utf_8_sig')
        
        # 读取图像为base64，用于直接在前端显示
        with open(os.path.join(optimization_dir, 'optimization_curve_zh.png'), 'rb') as f:
            curve_image = base64.b64encode(f.read()).decode('utf-8')
        
        # 定义文件路径
        curve_path = os.path.join(optimization_dir, 'optimization_curve_zh.png')
        csv_path = os.path.join(optimization_dir, 'optuna_trials.csv')
        
        # 构造返回结果
        response = {
            'best_params': best_params,
            'metrics': metrics,
            'optimization_id': optimization_id,
            'optimization_dir': optimization_dir,
            'curve_image': curve_image,
            'curve_path': curve_path,
            'csv_path': csv_path,
            'trials': trials_data
        }
        
        return response
    
    def _plot_optimization_curve(self, trials_df, n_trials, save_dir):
        """绘制优化曲线"""
        # 中文版
        plt.figure(figsize=(14/2.54, 8/2.54))  # 宽14cm，高8cm
        
        # 轮次从1开始显示
        x_values = trials_df['number'] + 1
        y_values = trials_df['value']
        
        plt.plot(x_values, y_values, 'o-', label='验证集损失')
        
        # 获取最优点
        best_idx = y_values.idxmin()
        best_trial = trials_df.iloc[best_idx]
        best_x = best_trial['number'] + 1  # 从1开始
        best_y = best_trial['value']
        
        # 绘制最优点
        plt.scatter([best_x], [best_y], color='red', s=100, zorder=5, label='最优点')
        plt.axhline(y=best_y, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='最优MSE')
        
        # 设置x轴范围从1开始
        plt.xlim(0.5, n_trials + 0.5)
        
        # 添加注释
        plt.title('贝叶斯优化过程曲线')
        plt.xlabel('优化轮次')
        plt.ylabel('验证损失 (MSE)')
        best_text = f'最优轮次: {int(best_x)}\n最优MSE: {best_y:.6f}'
        plt.text(best_x, best_y + 0.01, best_text, ha='center', va='bottom', fontsize=9)
        
        # 设置图例和标题
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, 'optimization_curve_zh.png'), dpi=300)
        plt.close()
        
        # 英文版
        plt.figure(figsize=(14/2.54, 8/2.54))  # 宽14cm，高8cm
        
        # 轮次从1开始显示
        plt.plot(x_values, y_values, 'o-', label='Validation Loss')
        plt.scatter([best_x], [best_y], color='red', s=100, zorder=5, label='Best Point')
        plt.axhline(y=best_y, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Best MSE')
        
        # 设置x轴范围从1开始
        plt.xlim(0.5, n_trials + 0.5)
        
        plt.title('Bayesian Optimization Progress Curve')
        plt.xlabel('Trial Number')
        plt.ylabel('Validation Loss (MSE)')
        best_text = f'Best Trial: {int(best_x)}\nBest MSE: {best_y:.6f}'
        plt.text(best_x, best_y + 0.01, best_text, ha='center', va='bottom', fontsize=9)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'optimization_curve_en.png'), dpi=300)
        plt.close()
    
    def get_optimization_history(self):
        """获取所有优化历史"""
        # 查找所有优化目录
        optimization_dirs = []
        for item in os.listdir(self.model_dir):
            if item.startswith('optimization_') and os.path.isdir(os.path.join(self.model_dir, item)):
                optimization_dirs.append(item)

        # 按时间戳排序（最新的在前面）
        optimization_dirs.sort(reverse=True)

        history = []
        for opt_dir in optimization_dirs:
            result_path = os.path.join(self.model_dir, opt_dir, 'result.json')
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        result = json.load(f)

                    # 格式化日期
                    date_str = result.get('date', '')
                    if date_str:
                        try:
                            # 处理新格式的时间戳 YYYYMMDD_HHMMSS
                            if '_' in date_str:
                                parts = date_str.split('_')
                                if len(parts) >= 2:  # 应该有两部分: date, time
                                    date_part = parts[0]
                                    time_part = parts[1]
                                    # 解析时间戳
                                    date_obj = pd.to_datetime(f"{date_part}_{time_part}", format='%Y%m%d_%H%M%S')
                                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    # 尝试直接解析
                                    date_obj = pd.Timestamp(date_str)
                                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # 尝试直接解析
                                date_obj = pd.Timestamp(date_str)
                                formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            print(f"时间戳解析失败: {str(e)}")
                            formatted_date = date_str
                    else:
                        formatted_date = 'Unknown'

                    # 获取优化曲线图路径
                    curve_path = os.path.join(self.model_dir, opt_dir, 'optimization_curve_zh.png')
                    curve_exists = os.path.exists(curve_path)

                    # 获取CSV文件路径
                    csv_path = os.path.join(self.model_dir, opt_dir, 'optuna_trials.csv')
                    csv_exists = os.path.exists(csv_path)

                    # 构建历史记录项
                    history_item = {
                        'id': result.get('id', ''),
                        'date': formatted_date,
                        'best_params': result.get('best_params', {}),
                        'metrics': result.get('metrics', {}),
                        'dir_path': os.path.join(self.model_dir, opt_dir),
                        'curve_path': curve_path if curve_exists else None,
                        'csv_path': csv_path if csv_exists else None
                    }

                    history.append(history_item)
                except json.JSONDecodeError:
                    # 跳过无法解析的JSON文件
                    print(f"无法解析JSON文件: {result_path}")
                    continue

        return history
        
    def get_optimization_result(self, opt_id):
        """获取特定的优化结果"""
        # 查找对应的优化目录
        optimization_dir = None
        for item in os.listdir(self.model_dir):
            if item.startswith('optimization_') and os.path.isdir(os.path.join(self.model_dir, item)):
                result_path = os.path.join(self.model_dir, item, 'result.json')
                if os.path.exists(result_path):
                    try:
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                        if result.get('id') == opt_id:
                            optimization_dir = os.path.join(self.model_dir, item)
                            break
                    except json.JSONDecodeError:
                        continue

        if not optimization_dir:
            return None

        # 读取结果文件
        result_path = os.path.join(optimization_dir, 'result.json')
        with open(result_path, 'r') as f:
            result = json.load(f)

        # 读取CSV文件
        csv_path = os.path.join(optimization_dir, 'optuna_trials.csv')
        if os.path.exists(csv_path):
            trials_df = pd.read_csv(csv_path)
            print(f"CSV文件列名: {trials_df.columns.tolist()}")
            
            # 处理数据类型，确保转换为正确的Python类型
            trials_data = []
            for i, row in trials_df.iterrows():
                try:
                    trial_data = {
                        'number': int(row['number']) if 'number' in row and not pd.isna(row['number']) else i,
                        'value': float(row['value']) if 'value' in row and not pd.isna(row['value']) else 0.0,
                        'd_model': int(row['params_d_model']) if 'params_d_model' in row and not pd.isna(row['params_d_model']) else None,
                        'nhead': int(row['params_nhead']) if 'params_nhead' in row and not pd.isna(row['params_nhead']) else None,
                        'num_layers': int(row['params_num_layers']) if 'params_num_layers' in row and not pd.isna(row['params_num_layers']) else None,
                        'lr': float(row['params_lr']) if 'params_lr' in row and not pd.isna(row['params_lr']) else None,
                        'weight_decay': float(row['params_weight_decay']) if 'params_weight_decay' in row and not pd.isna(row['params_weight_decay']) else None,
                        # batch_size列已删除
                        'dropout': float(0.05)
                    }
                    trials_data.append(trial_data)
                    if i < 3:  # 只打印前几条记录作为调试
                        print(f"Trial {i} data: {trial_data}")
                except Exception as e:
                    print(f"Error processing trial {i}: {str(e)}")
                    # 出错时使用默认值
                    trial_data = {
                        'number': i,
                        'value': 0.0,
                        'd_model': None,
                        'nhead': None,
                        'num_layers': None,
                        'lr': None,
                        'weight_decay': None,
                        'dropout': float(0.05)
                    }
                    trials_data.append(trial_data)
            print(f"Total trials loaded from CSV: {len(trials_data)}")
        else:
            trials_data = result.get('trials', [])
            print(f"Using trials from JSON result: {len(trials_data)}")

        # 读取优化曲线图
        curve_path = os.path.join(optimization_dir, 'optimization_curve_zh.png')
        curve_base64 = None
        if os.path.exists(curve_path):
            with open(curve_path, 'rb') as f:
                curve_bytes = f.read()
                curve_base64 = base64.b64encode(curve_bytes).decode('utf-8')

        # 构建响应数据
        response_data = {
            'id': result.get('id', ''),
            'date': result.get('date', ''),
            'best_params': result.get('best_params', {}),
            'metrics': result.get('metrics', {}),
            'trials': trials_data,
            'curve_image': curve_base64,
            'dir_path': optimization_dir,
            'curve_path': curve_path if os.path.exists(curve_path) else None,
            'csv_path': csv_path if os.path.exists(csv_path) else None
        }

        return response_data
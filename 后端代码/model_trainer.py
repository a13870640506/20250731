"""
模型训练模块，用于训练Transformer模型
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 在导入matplotlib之前设置后端为Agg（非交互式后端，不需要图形界面）
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
import base64
from io import BytesIO
from sklearn.metrics import r2_score, mean_squared_error

# 导入模型和训练相关组件
from my_model import TimeSeriesTransformer
from my_train import train_model
from my_inference import validate_model

# 设置matplotlib中文字体
plt.rcParams.update({
    'font.size': 10.5,
    'axes.labelsize': 10.5,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.5,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.5,
    'axes.linewidth': 1.0,
    'figure.constrained_layout.use': True,
    'font.family': 'Microsoft YaHei'
})

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self, model_dir='./models'):
        """初始化模型训练器"""
        self.model_dir = model_dir
        
        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, data_path, model_name='transformer', epochs=300, batch_size=16, learning_rate=0.0001, weight_decay=0.0001, dropout=0.05, d_model=256, nhead=4, num_layers=2, model_save_path=None):
        """
        训练Transformer模型，与app.py的train_model_api接口对应
        
        参数:
            data_path: 处理后的数据路径
            model_name: 模型名称
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            dropout: Dropout率
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            model_save_path: 模型保存路径（可选）
            
        返回:
            训练结果字典
        """
        # 创建模型参数和训练参数字典
        model_params = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dropout': dropout,
            'input_dim': None,  # 将根据数据设置
            'output_dim': None  # 将根据数据设置
        }
        
        training_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
        
        return self.train_model(data_path, model_params, training_params, model_save_path)
        
    def train_model(self, data_path, model_params, training_params, model_save_path=None):
        """
        训练Transformer模型
        
        参数:
            data_path: 处理后的数据路径
            model_params: 模型参数字典
            training_params: 训练参数字典
            model_save_path: 模型保存路径（可选）
            
        返回:
            训练结果字典
        """
        # 创建或使用指定的模型保存目录
        if model_save_path:
            model_save_dir = os.path.join(self.model_dir, model_save_path)
        else:
            # 根据参数自动生成保存路径，与前端格式完全一致
            d_model = model_params["d_model"]
            lr = f'{training_params["learning_rate"]:.6f}'
            bs = training_params["batch_size"]
            # 使用models/前缀，与前端保持一致
            model_path = f'models/model_c{d_model}_lr{lr}_bs{bs}'
            model_save_dir = os.path.join(self.model_dir, model_path)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 加载数据
        train_data = np.load(os.path.join(data_path, 'train_data.npz'))
        val_data = np.load(os.path.join(data_path, 'val_data.npz'))
        test_data = np.load(os.path.join(data_path, 'test_data.npz'))
        
        X_train, y_train = train_data['X_train'], train_data['y_train']
        X_val, y_val = val_data['X_val'], val_data['y_val']
        X_test, y_test = test_data['X_test'], test_data['y_test']
        
        # 设置输入和输出维度
        model_params['input_dim'] = X_train.shape[2]
        model_params['output_dim'] = y_train.shape[1]
        
        # 读取数据信息文件，获取参数名
        param_names = self._get_param_names(data_path)
        if not param_names:
            # 如果无法获取参数名，使用默认值
            param_names = [f'参数{i+1}' for i in range(model_params['output_dim'])]
        
        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # 创建GeotechDataset数据集，按照main.py中的方式
        # 注意：这里我们已经加载了处理好的npz文件，我们假设它们已经包含了正确处理的values和deltas
        # 如果npz文件中的数据结构不一致，这里需要调整
        
        # 打印数据形状，用于调试
        print(f"数据形状: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        print(f"数据形状: y_train={y_train.shape}, y_val={y_val.shape}, y_test={y_test.shape}")
        
        # 创建掩码数据 - 默认所有数据点都是有效的
        if 'mask_train' not in locals() or 'mask_val' not in locals() or 'mask_test' not in locals():
            print("创建默认掩码数据...")
            mask_train = np.ones((X_train.shape[0], X_train.shape[1]), dtype=bool)
            mask_val = np.ones((X_val.shape[0], X_val.shape[1]), dtype=bool)
            mask_test = np.ones((X_test.shape[0], X_test.shape[1]), dtype=bool)
        
        # 创建数据集
        from my_dataset import GeotechDataset
        train_dataset = GeotechDataset(X_train, delta_train if 'delta_train' in locals() else None, mask_train, y_train)
        val_dataset = GeotechDataset(X_val, delta_val if 'delta_val' in locals() else None, mask_val, y_val)
        test_dataset = GeotechDataset(X_test, delta_test if 'delta_test' in locals() else None, mask_test, y_test)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False)
        
        # 创建模型 - TimeSeriesTransformer不接受dropout参数
        model = TimeSeriesTransformer(
            input_dim=model_params['input_dim'],
            d_model=model_params['d_model'],
            nhead=model_params['nhead'],
            num_layers=model_params['num_layers'],
            num_outputs=model_params['output_dim']
        ).to(device)
        
        # 注意：虽然我们保留dropout参数用于路径命名，但TimeSeriesTransformer本身不使用它
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=training_params['learning_rate'],
            weight_decay=training_params['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # 训练循环 - 严格按照main.py中的逻辑
        epochs = training_params['epochs']
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_path = os.path.join(model_save_dir, 'best_model.pth')
        
        for epoch in range(epochs):
            # 训练一个epoch - 使用my_train.py中的train_model
            train_loss = train_model(model, train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            
            # 验证 - 使用my_inference.py中的validate_model
            val_loss, _, _ = validate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch + 1}: 保存新最佳模型，验证损失={val_loss:.6f}")
            
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 绘制损失曲线（按main.py逻辑）
        loss_curve_base64 = self._plot_loss_history(train_losses, val_losses)
        
        # 加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        
        # 在训练集、验证集和测试集上进行预测 - 使用imported的validate_model
        _, train_preds, train_labels = validate_model(model, train_loader, criterion)
        _, val_preds, val_labels = validate_model(model, val_loader, criterion)
        _, test_preds, test_labels = validate_model(model, test_loader, criterion)
        
        # 计算详细评估指标
        metrics = self._calculate_metrics(test_labels, test_preds, param_names)
        
        # 生成各种对比图
        combined_plots = self._plot_predictions_combined(train_labels, train_preds, test_labels, test_preds, param_names, metrics)
        train_plots = self._plot_predictions_individual(train_labels, train_preds, param_names, metrics, 'train')
        test_plots = self._plot_predictions_individual(test_labels, test_preds, param_names, metrics, 'test')
        
        # 简化指标供前端展示
        simple_train_metrics = {'loss': float(train_losses[-1]), 'r2': float(metrics[-1]['r2'])}
        simple_val_metrics = {'loss': float(val_losses[-1]), 'r2': float(metrics[-1]['r2'])}
        simple_test_metrics = {'loss': float(metrics[-1]['mse']), 'r2': float(metrics[-1]['r2'])}
        
        # 保存训练参数和结果
        training_result = {
            'model_params': model_params,
            'training_params': training_params,
            'best_epoch': train_losses.index(min(train_losses)) + 1,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # 保存结果到JSON文件
        with open(os.path.join(model_save_dir, 'training_result.json'), 'w') as f:
            import json
            json.dump(training_result, f, indent=4)
        
        # 返回训练结果
        return {
            'model_path': best_model_path,
            'train_metrics': simple_train_metrics,
            'val_metrics': simple_val_metrics,
            'test_metrics': simple_test_metrics,
            'loss_curve': loss_curve_base64,
            'combined_plots': combined_plots,
            'train_plots': train_plots,
            'test_plots': test_plots
        }
        
    def _get_param_names(self, data_path):
        """从data_info.txt文件中获取参数名称"""
        data_info_path = os.path.join(data_path, 'data_info.txt')
        param_names = []
        
        if os.path.exists(data_info_path):
            # 尝试多种编码读取文件
            for encoding in ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1']:
                try:
                    with open(data_info_path, 'r', encoding=encoding) as f:
                        for line in f.readlines():
                            if line.startswith('输出标签:'):
                                # 提取参数名
                                params_str = line.split(':', 1)[1].strip()
                                param_names = [p.strip() for p in params_str.split(',')]
                                return param_names
                except Exception as e:
                    continue
                    
        return param_names
    
    def _train_epoch(self, model, dataloader, criterion, optimizer):
        """训练一个epoch - 严格按照my_train.py中的train_model实现"""
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            values = batch['values'].to(device)
            deltas = batch['deltas'].to(device)
            labels = batch['label'].to(device)
            mask = batch['mask'].to(device) # (batch_size, seq_len)

            # 转换为Transformer需要的格式: True表示填充位置
            transformer_mask = ~mask  # 取反: 真实数据->False, 填充位置->True
            
            # 前向传播(传入注意力掩码)
            outputs = model(values, deltas, transformer_mask)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * values.size(0)
        
        return total_loss / len(dataloader.dataset)
    
    def _validate(self, model, dataloader, criterion):
        """验证模型 - 严格按照my_inference.py中的validate_model实现"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                values = batch['values'].to(device)
                deltas = batch['deltas'].to(device)
                labels = batch['label'].to(device)
                mask = batch['mask'].to(device)
                transformer_mask = ~mask  # 转换掩码格式
                
                outputs = model(values, deltas, transformer_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * values.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        return avg_loss, all_preds, all_labels
    
    def _evaluate_model(self, model, dataloader, criterion):
        """评估模型"""
        loss, preds, targets = self._validate(model, dataloader, criterion)
        
        # 计算R2分数
        r2 = r2_score(targets, preds)
        
        # 计算均方根误差
        rmse = np.sqrt(mean_squared_error(targets, preds))
        
        return {
            'loss': loss,
            'r2': r2,
            'rmse': rmse
        }
    
    def _plot_loss_history(self, train_losses, val_losses):
        """绘制训练和验证损失曲线，与main.py中的plot_loss_history保持一致"""
        plt.figure(figsize=(14 / 2.54, 8 / 2.54))  # 宽14cm
        
        # 绘制损失曲线
        train_line, = plt.plot(train_losses, label=f'训练集均方误差 (最小值: {min(train_losses):.4f})')
        val_line, = plt.plot(val_losses, label=f'验证集均方误差 (最小值: {min(val_losses):.4f})')
        
        # 找到最小损失值及其位置
        min_train_loss = min(train_losses)
        min_train_epoch = train_losses.index(min_train_loss)
        min_val_loss = min(val_losses)
        min_val_epoch = val_losses.index(min_val_loss)
        
        # 标记最小损失点
        plt.scatter(min_train_epoch, min_train_loss, color=train_line.get_color(), s=100, zorder=5)
        plt.scatter(min_val_epoch, min_val_loss, color=val_line.get_color(), s=100, zorder=5)
        
        # 添加文本标注
        plt.text(min_train_epoch, min_train_loss + 0.03,
                 f'训练集最小损失轮次: {min_train_epoch + 1}',
                 ha='center', va='bottom')
        plt.text(min_val_epoch, min_val_loss + 0.1,
                 f'验证集最小损失轮次: {min_val_epoch + 1}',
                 ha='center', va='bottom')
        
        # 更新图例标签以包含最小损失值
        plt.legend(loc='upper right')
        
        plt.title('训练与验证阶段的均方误差演化曲线')
        plt.xlabel('轮次')
        plt.ylabel('均方误差')
        plt.grid(True)
        
        # 将图表保存到文件和生成base64编码
        model_save_dir = os.path.join(self.model_dir, 'train_results')
        os.makedirs(model_save_dir, exist_ok=True)
        plt.savefig(os.path.join(model_save_dir, 'loss_history_zh.png'), dpi=300)
        
        # 将图表转换为base64编码的字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
        
    def _plot_predictions_combined(self, true_train, pred_train, true_test, pred_test, param_names, metrics):
        """每个参数绘图，训练集与测试集一起展示，与main.py中的plot_predictions_combined保持一致"""
        result_images = []
        model_save_dir = os.path.join(self.model_dir, 'train_results')
        os.makedirs(model_save_dir, exist_ok=True)
        
        for i, param in enumerate(param_names):
            title = f'{param} - 训练与测试集'
            xlabel = '真实值 (mm)'
            ylabel = '预测值 (mm)'
            legend_labels = ['训练集样本', '测试集样本']
            metrics_text = (f'测试集指标：\n'
                           f'R² = {metrics[i]["r2"]:.4f}\n'
                           f'MSE = {metrics[i]["mse"]:.4f}\n'
                           f'MAPE = {metrics[i]["mape"]:.2f}%')

            fig, ax = plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54), dpi=300)

            # 提取当前参数的值
            t_train = true_train[:, i]
            p_train = pred_train[:, i]
            t_test = true_test[:, i]
            p_test = pred_test[:, i]

            # 绘制训练集（蓝色方框）和测试集（橙色圆圈）
            ax.scatter(t_train, p_train, alpha=0.5, s=10, marker='s', color='#1F77B4', label=legend_labels[0])
            ax.scatter(t_test, p_test, alpha=0.6, s=10, marker='o', color='#FF7F0E', label=legend_labels[1])

            # 对角线
            min_val = min(t_train.min(), p_train.min(), t_test.min(), p_test.min())
            max_val = max(t_train.max(), p_train.max(), t_test.max(), p_test.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper left', frameon=True)

            # 指标信息
            ax.text(0.95, 0.05, metrics_text,
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.7))

            # 保存
            fig_path = os.path.join(model_save_dir, f'combined_prediction_{param}.png')
            fig.savefig(fig_path, dpi=300)
            
            # 转换为base64并添加到结果中
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            result_images.append({
                'param': param,
                'type': 'combined',
                'image': image_base64,
                'path': fig_path
            })
            
            plt.close(fig)
            
        return result_images

    def _plot_predictions_individual(self, true, pred, param_names, metrics, dataset_type='test'):
        """每个参数单独绘图，训练集和测试集分别展示，与main.py中的plot_predictions_individual保持一致"""
        result_images = []
        model_save_dir = os.path.join(self.model_dir, 'train_results')
        os.makedirs(model_save_dir, exist_ok=True)
        
        for i, param in enumerate(param_names):
            title = f"{param} - {'测试' if dataset_type == 'test' else '训练'}集"
            xlabel = '真实值 (mm)'
            ylabel = '预测值 (mm)'
            legend_label = '训练集样本' if dataset_type == 'train' else '测试集样本'
            metrics_text = (f'R² = {metrics[i]["r2"]:.4f}\n'
                           f'MSE = {metrics[i]["mse"]:.4f}\n'
                           f'MAPE = {metrics[i]["mape"]:.2f}%')

            fig, ax = plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54), dpi=300)

            param_true = true[:, i]
            param_pred = pred[:, i]

            # 设置 marker
            marker = 's' if dataset_type == 'train' else 'o'  # 训练集用方形，测试集用圆形

            # 散点图
            ax.scatter(param_true, param_pred, alpha=0.6, s=10, marker=marker, label=legend_label)

            # 对角线
            min_val = min(param_true.min(), param_pred.min())
            max_val = max(param_true.max(), param_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

            # 图例
            ax.legend(loc='upper left', frameon=True)

            # 标题、标签
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # 添加评估指标文本
            ax.text(0.95, 0.05, metrics_text,
                   transform=plt.gca().transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.7))

            # 保存图像
            fig_path = os.path.join(model_save_dir, f'{dataset_type}_prediction_{param}.png')
            fig.savefig(fig_path, dpi=300)
            
            # 转换为base64并添加到结果中
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            result_images.append({
                'param': param,
                'type': dataset_type,
                'image': image_base64,
                'path': fig_path
            })
            
            plt.close(fig)
            
        return result_images
        
    def _calculate_metrics(self, true, pred, param_names):
        """计算评估指标，与main.py中的calculate_metrics保持一致"""
        metrics = []
        
        for i, param in enumerate(param_names):
            param_true = true[:, i]
            param_pred = pred[:, i]
            
            # 计算相对误差
            relative_error = np.abs((param_pred - param_true) / (param_true + 1e-10)) * 100
            
            param_metrics = {
                'r2': r2_score(param_true, param_pred),
                'mse': mean_squared_error(param_true, param_pred),
                'mape': np.mean(relative_error),
                'mean_relative_error': np.mean(relative_error),
                'max_relative_error': np.max(relative_error)
            }
            
            metrics.append(param_metrics)
        
        # 整体指标
        overall_r2 = r2_score(true.ravel(), pred.ravel())
        overall_mse = mean_squared_error(true.ravel(), pred.ravel())
        overall_relative_error = np.abs((pred - true) / (true + 1e-10)) * 100
        overall_mape = np.mean(overall_relative_error)
        
        overall_metrics = {
            'r2': overall_r2,
            'mse': overall_mse,
            'mape': overall_mape,
            'mean_relative_error': np.mean(overall_relative_error)
        }
        
        metrics.append(overall_metrics)  # 最后一个为整体指标
        
        return metrics
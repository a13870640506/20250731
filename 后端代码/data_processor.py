"""
数据处理模块，用于处理上传的数据集
参考main.py中的数据处理逻辑
"""

import os
import numpy as np
import pandas as pd
# 在导入matplotlib之前设置后端为Agg（非交互式后端，不需要图形界面）
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import base64
from io import BytesIO

# 设置matplotlib中文字体
plt.rcParams.update({
    # 文本默认字号（刻度、注释、图例文本等）
    'font.size': 10.5,  # 12pt → 对应论文正文
    'axes.labelsize': 10.5,  # x/y 轴标签
    'xtick.labelsize': 10.5,  # x 刻度数字
    'ytick.labelsize': 10.5,  # y 刻度数字
    'legend.fontsize': 10.5,  # 图例文字

    # 网格样式
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.5,
    # 坐标轴线宽
    'axes.linewidth': 1.0,

    # 紧凑布局
    'figure.constrained_layout.use': True,
    # 字体
    'font.family': 'Microsoft YaHei'  # 中文字体
})

def preprocess_data(data_list, max_len=4, input_scaler=None, output_scaler=None):
    """处理变长时序数据"""
    padded_values = []
    padded_deltas = []
    masks = []
    labels = []

    for time_steps, values, deltas, label in data_list:
        seq_len = len(time_steps)

        # 输入标准化
        if input_scaler:
            input_features = np.hstack([values, deltas])
            input_norm = input_scaler.transform(input_features)
            values_norm = input_norm[:, :values.shape[1]]
            deltas_norm = input_norm[:, values.shape[1]:]
        else:
            values_norm = values
            deltas_norm = deltas

        # 填充时序数据
        pad_len = max_len - seq_len
        padded_values.append(np.pad(values_norm, [(0, pad_len), (0, 0)], 'constant'))
        padded_deltas.append(np.pad(deltas_norm, [(0, pad_len), (0, 0)], 'constant'))

        # 创建注意力掩码
        mask = np.ones(max_len, dtype=bool)
        if seq_len < max_len:
            mask[seq_len:] = False
        masks.append(mask)

        # 应用输出标准化
        if output_scaler:
            label_norm = output_scaler.transform(label.reshape(1, -1))[0]
            labels.append(label_norm)
        else:
            labels.append(label)

    return (
        np.array(padded_values, dtype=np.float32),
        np.array(padded_deltas, dtype=np.float32),
        np.array(masks, dtype=bool),
        np.array(labels, dtype=np.float32)
    )

class DataProcessor:
    def __init__(self, data_dir='./data', scaler_dir='./scalers'):
        """初始化数据处理器"""
        self.data_dir = data_dir
        self.scaler_dir = scaler_dir
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)
    
    def process_data(self, input_file, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        处理上传的数据文件，参考main.py中的get_csv_data函数
        
        参数:
            input_file: 输入数据文件
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        返回:
            处理结果字典
        """
        print("\n" + "="*50)
        print("开始处理数据文件")
        print("="*50)
        
        # 读取CSV文件
        df = pd.read_csv(input_file)
        samples_num = df.shape[0]
        print(f"读取CSV文件成功，共 {samples_num} 条样本")
        
        # 分离输入于输出参数，参考main.py中的get_csv_data函数
        output_params = df.iloc[:, :5]  # 前5列为输出参数
        input_params = df.iloc[:, 5:]   # 后面的列为输入参数
        
        input_features = input_params.values
        output_labels = output_params.values
        
        print(f"数据分离完成:")
        print(f"- 输入特征: {input_params.shape[1]} 维")
        print(f"- 输出标签: {output_params.shape[1]} 维")
        
        # 获取列名
        input_columns = input_params.columns.tolist()
        output_columns = output_params.columns.tolist()
        
        # 生成模拟时序数据，参考main.py中的get_csv_data函数
        time_deltas = np.array([0, 1, 1, 1]).reshape(-1, 1)
        time_steps = np.cumsum(time_deltas).reshape(-1)
        max_steps = 4
        
        # 构建数据列表
        data = []
        for i in range(samples_num):
            values = np.tile(input_features[i], (max_steps, 1))
            labels = output_labels[i]
            data.append((time_steps, values, time_deltas, labels))
        
        # 准备标准化
        all_input_features = []
        all_labels = []
        
        for time_steps, values, deltas, label in data:
            input_features = np.hstack([values, deltas])
            all_input_features.append(input_features)
            all_labels.append(label)
        
        all_input_features = np.vstack(all_input_features)
        all_labels = np.vstack(all_labels)
        
        # 创建标准化器
        input_scaler = StandardScaler()
        input_scaler.fit(all_input_features)
        
        output_scaler = StandardScaler()
        output_scaler.fit(all_labels)
        
        # 创建带时间戳的文件夹，使用标准格式
        now = pd.Timestamp.now()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        timestamp = f"{date_part}_{time_part}"
        
        processed_data_dir = os.path.join(self.data_dir, f'processed_{timestamp}')
        os.makedirs(processed_data_dir, exist_ok=True)
        print(f"\n创建数据处理目录: {processed_data_dir}")
        
        # 创建图表保存目录
        plots_dir = os.path.join(self.data_dir, f'plots_{timestamp}')
        os.makedirs(plots_dir, exist_ok=True)
        print(f"创建图表保存目录: {plots_dir}")
        
        # 生成标准化前后对比图
        standardization_plots = self._generate_standardization_plots(
            all_labels, output_scaler, output_columns, plots_dir
        )
        
        # 预处理数据
        padded_values, padded_deltas, masks, labels = preprocess_data(
            data, input_scaler=input_scaler, output_scaler=output_scaler
        )
        
        # 划分数据集: 7:1:2 (训练:验证:测试)
        print(f"\n数据集划分比例: 训练集={train_ratio}, 验证集={val_ratio}, 测试集={test_ratio}")
        
        # 先分训练+验证(80%)和测试(20%)
        X_train_val, X_test, delta_train_val, delta_test, mask_train_val, mask_test, y_train_val, y_test = train_test_split(
            padded_values, padded_deltas, masks, labels, test_size=test_ratio, random_state=42)
        
        # 再分训练(70%)和验证(10%)
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, delta_train, delta_val, mask_train, mask_val, y_train, y_val = train_test_split(
            X_train_val, delta_train_val, mask_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42)
            
        print(f"数据集划分完成:")
        print(f"- 训练集: {len(y_train)} 样本")
        print(f"- 验证集: {len(y_val)} 样本")
        print(f"- 测试集: {len(y_test)} 样本")
        
        # 保存标准化器
        scaler_path = os.path.join(self.scaler_dir, 'scalers.pkl')
        joblib.dump((input_scaler, output_scaler), scaler_path)
        
        # 保存训练集、验证集和测试集
        np.savez(
            os.path.join(processed_data_dir, 'train_data.npz'),
            X_train=X_train,
            delta_train=delta_train,
            mask_train=mask_train,
            y_train=y_train
        )
        
        np.savez(
            os.path.join(processed_data_dir, 'val_data.npz'),
            X_val=X_val,
            delta_val=delta_val,
            mask_val=mask_val,
            y_val=y_val
        )
        
        np.savez(
            os.path.join(processed_data_dir, 'test_data.npz'),
            X_test=X_test,
            delta_test=delta_test,
            mask_test=mask_test,
            y_test=y_test
        )
        
        # 保存原始数据副本
        df.to_csv(os.path.join(processed_data_dir, 'original_data.csv'), index=False)
        
        # 保存数据集信息，确保使用UTF-8编码
        with open(os.path.join(processed_data_dir, 'data_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"输入特征: {', '.join(input_columns)}\n")
            f.write(f"输出标签: {', '.join(output_columns)}\n")
            f.write(f"训练集样本数: {len(y_train)}\n")
            f.write(f"验证集样本数: {len(y_val)}\n")
            f.write(f"测试集样本数: {len(y_test)}\n")
            f.write(f"输入维度: {input_features.shape[1]}\n")
            f.write(f"输出维度: {output_labels.shape[1]}\n")
        
        # 返回处理结果
        print("\n" + "="*50)
        print("数据处理完成")
        print(f"生成了 {len(standardization_plots)} 个标准化对比图")
        print("="*50 + "\n")
        
        return {
            'train_size': len(y_train),
            'val_size': len(y_val),
            'test_size': len(y_test),
            'input_columns': input_columns,
            'output_columns': output_columns,
            'input_dim': input_features.shape[1],
            'output_dim': output_labels.shape[1],
            'standardization_plots': standardization_plots,
            'data_path': processed_data_dir
        }
    
    def _generate_standardization_plots(self, raw_labels, scaler, param_names, plots_dir):
        """
        生成标准化前后对比图，参考main.py中的plot_label_sequence_comparison函数
        使用main.py中的参数名称：['拱顶下沉', '拱顶下沉2', '周边收敛1', '周边收敛2', '拱脚下沉']
        
        参数:
            raw_labels: 原始标签数据
            scaler: 标准化器
            param_names: 参数名称列表
            plots_dir: 图片保存目录（带时间戳）
        """
        plots = []
        norm_labels = scaler.transform(raw_labels)
        samples = np.arange(raw_labels.shape[0])
        
        # 使用main.py中定义的参数名称
        param_display_names = ['拱顶下沉', '拱顶下沉2', '周边收敛1', '周边收敛2', '拱脚下沉']
        
        for i, name in enumerate(param_names):
            # 使用显示名称
            display_name = param_display_names[i] if i < len(param_display_names) else name
            
            fig, ax = plt.subplots(figsize=(20/2.54, 10/2.54), dpi=300)
            
            # 标准化前
            ax.plot(
                samples,
                raw_labels[:, i],
                marker='o', markersize=3,
                linewidth=0.8, alpha=0.7,
                label='标准化前'
            )
            
            # 标准化后（空心点）
            ax.plot(
                samples,
                norm_labels[:, i],
                marker='o', markersize=4,
                markerfacecolor='none', markeredgewidth=0.8,
                linewidth=0.8, alpha=0.7,
                label='标准化后'
            )
            
            # 设置图表属性
            ax.set_title(f'{display_name} 标准化前后序列对比')
            ax.set_xlabel('样本编号')
            ax.set_ylabel(f'{display_name} (mm)')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # 使用之前创建的带时间戳的图片保存目录
            # plots_dir 变量已在 process_data 方法中定义
            
            # 保存图片到文件，不使用时间戳命名单个图片
            image_filename = f'{display_name}_standardization_comparison.png'
            image_path = os.path.join(plots_dir, image_filename)
            fig.savefig(image_path, format='png', dpi=300, bbox_inches='tight')
            
            # 将图表转换为base64编码的字符串
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            print(f"保存图片: {image_path}")
            
            # 添加到结果列表
            plots.append({
                'title': f'{display_name} 标准化前后序列对比',
                'name': display_name,
                'image': image_base64,
                'image_path': image_path
            })
        
        return plots
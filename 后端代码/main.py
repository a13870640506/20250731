"""
正演分析: 围岩参数 —> 围岩位移

1.可视化输出图表（筛走误差大的样本点）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_squared_error

from my_model import TimeSeriesTransformer
from my_train import train_model
from my_dataset import GeotechDataset
from my_inference import validate_model

import pandas as pd
import os
# import seaborn as sns

# 添加 optuna 以支持贝叶斯优化
import optuna

csv_path = './data/fea_data_正演1.csv'
scaler_dir = './scalers'
MODEL_DIR = './models_正演'

# 创建目录
os.makedirs(scaler_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def calculate_metrics(true, pred, param_names):
    """计算评估指标"""
    metrics = {}
    for i, param in enumerate(param_names):
        param_true = true[:, i]
        param_pred = pred[:, i]

        # 计算相对误差
        relative_error = np.abs((param_pred - param_true) / (param_true + 1e-10)) * 100

        metrics[param] = {
            'r2': r2_score(param_true, param_pred),
            'mse': mean_squared_error(param_true, param_pred),
            'mape': np.mean(relative_error),
            'mean_relative_error': np.mean(relative_error),
            'max_relative_error': np.max(relative_error),
            'relative_errors': relative_error
        }

    # 整体指标
    overall_r2 = r2_score(true.ravel(), pred.ravel())
    overall_mse = mean_squared_error(true.ravel(), pred.ravel())
    overall_relative_error = np.abs((pred - true) / (true + 1e-10)) * 100
    overall_mape = np.mean(overall_relative_error)

    metrics['overall'] = {
        'r2': overall_r2,
        'mse': overall_mse,
        'mape': overall_mape,
        'mean_relative_error': np.mean(overall_relative_error),
        'relative_errors': overall_relative_error
    }

    return metrics


# ── 一次性设置全局样式 ─────────────────────────────────────────────────────
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
    'font.family': 'Microsoft YaHei'  # 中
    # 'font.family': 'Times New Roman' # 英
})


def plot_loss_history(train_losses, val_losses, lang='zh'):
    """绘制训练和验证损失曲线，并在图例中显示最小损失值"""
    # 设置中英文文本
    if lang == 'zh':
        title = '训练与验证阶段的均方误差演化曲线'
        xlabel = '轮次'
        ylabel = '均方误差'
        legend_labels = [
            f'训练集均方误差 (最小值: {min(train_losses):.4f})',
            f'验证集均方误差 (最小值: {min(val_losses):.4f})'
        ]
        train_text = f'训练集最小损失轮次: {train_losses.index(min(train_losses)) + 1}'
        val_text = f'验证集最小损失轮次: {val_losses.index(min(val_losses)) + 1}'
    else:
        title = 'Training and Validation Loss History (MSE)'
        xlabel = 'Epoch'
        ylabel = 'Loss'
        legend_labels = [
            f'Training MSE (min: {min(train_losses):.4f})',
            f'Validation MSE (min: {min(val_losses):.4f})'
        ]
        train_text = f'Min Train Epoch: {train_losses.index(min(train_losses)) + 1}'
        val_text = f'Min Val Epoch: {val_losses.index(min(val_losses)) + 1}'

    plt.figure(figsize=(14 / 2.54, 8 / 2.54))  # 宽14cm

    # 绘制损失曲线
    train_line, = plt.plot(train_losses, label=legend_labels[0])
    val_line, = plt.plot(val_losses, label=legend_labels[1])

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
             train_text,
             ha='center', va='bottom')
    plt.text(min_val_epoch, min_val_loss + 0.1,
             val_text,
             ha='center', va='bottom')

    # 更新图例标签以包含最小损失值
    plt.legend(loc='upper right')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, f'loss_history_{lang}.png'), dpi=300)
    plt.close()


def plot_predictions_combined(true_train, pred_train, true_test, pred_test, param_names, metrics, lang='zh'):
    """每个参数绘图，训练集与测试集一起展示"""

    for i, param in enumerate(param_names):
        if lang == 'zh':
            title = f'{param} - 训练与测试集'
            xlabel = '真实值 (mm)'
            ylabel = '预测值 (mm)'
            legend_labels = ['训练集样本', '测试集样本']
            metrics_text = (f'测试集指标：\n'
                            f'R² = {metrics[param]["r2"]:.4f}\n'
                            f'MSE = {metrics[param]["mse"]:.4f}\n'
                            f'MAPE = {metrics[param]["mape"]:.2f}%')
        else:
            title = f'{param} - Train & Test Sets'
            xlabel = 'True Values (mm)'
            ylabel = 'Predictions (mm)'
            legend_labels = ['Train', 'Test']
            metrics_text = (f'Test Metrics:\n'
                            f'R² = {metrics[param]["r2"]:.4f}\n'
                            f'MSE = {metrics[param]["mse"]:.4f}\n'
                            f'MAPE = {metrics[param]["mape"]:.2f}%')

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

        # 自动同步刻度
        plt.draw()
        y_ticks = ax.get_yticks()
        if len(y_ticks) > 1:
            interval = y_ticks[1] - y_ticks[0]
            ax.xaxis.set_major_locator(MultipleLocator(interval))

        # 指标信息
        ax.text(0.95, 0.05, metrics_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7))

        # 保存
        fig.savefig(os.path.join(MODEL_DIR, f'combined_prediction_{param}_{lang}.png'), dpi=300)
        plt.close(fig)


def plot_predictions_individual(true, pred, param_names, metrics, dataset_type='test', lang='zh'):
    """每个参数单独绘图，训练集和测试集分别展示"""

    for i, param in enumerate(param_names):
        # 修复字符串格式化问题
        if lang == 'zh':
            title = f"{param} - {'测试' if dataset_type == 'test' else '训练'}集"
            xlabel = '真实值 (mm)'
            ylabel = '预测值 (mm)'
            legend_label = '训练集样本' if dataset_type == 'train' else '测试集样本'
            metrics_text = (f'R² = {metrics[param]["r2"]:.4f}\n'
                            f'MSE = {metrics[param]["mse"]:.4f}\n'
                            f'MAPE = {metrics[param]["mape"]:.2f}%')
        else:
            title = f'{param} - {dataset_type.capitalize()} Set'
            xlabel = 'True Values (mm)'
            ylabel = 'Predictions (mm)'
            legend_label = 'Train' if dataset_type == 'train' else 'Test'
            metrics_text = (f'R² = {metrics[param]["r2"]:.4f}\n'
                            f'MSE = {metrics[param]["mse"]:.4f}\n'
                            f'MAPE = {metrics[param]["mape"]:.2f}%')

        fig, ax = plt.subplots(figsize=(7.5 / 2.54, 7.5 / 2.54), dpi=300)

        param_true = true[:, i]
        param_pred = pred[:, i]

        # 设置 marker
        if dataset_type == 'train':
            marker = 's'  # 方形
        else:
            marker = 'o'  # 圆形

        # 散点图
        ax.scatter(param_true, param_pred, alpha=0.6, s=10, marker=marker, label=legend_label)

        # 对角线
        min_val = min(param_true.min(), param_pred.min())
        max_val = max(param_true.max(), param_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        # 图例
        ax.legend(loc='upper left', frameon=True)

        # 标题、标签、刻度字号
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # 自动刻度同步
        plt.draw()
        y_ticks = ax.get_yticks()
        if len(y_ticks) > 1:
            interval = y_ticks[1] - y_ticks[0]
            ax.xaxis.set_major_locator(MultipleLocator(interval))

        # 添加评估指标文本
        ax.text(0.95, 0.05, metrics_text,
                transform=plt.gca().transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7))

        # 保存图像
        fig.savefig(os.path.join(MODEL_DIR, f'{dataset_type}_prediction_{param}_{lang}.png'), dpi=300)
        plt.close(fig)


def plot_relative_errors_per_param(metrics, param_names, dataset_type='test', lang='zh'):
    """为每个参数单独绘制相对误差折线图"""

    for param in param_names:
        # 修复字符串格式化问题
        if lang == 'en':
            title = f'Relative Errors for {param} - {dataset_type.capitalize()} Set'
            xlabel = 'Sample Index'
            ylabel = 'Relative Error (%)'
            legend_label = '10% Error Threshold'
        else:
            title = f"{param}相对误差 - {'测试' if dataset_type == 'test' else '训练'}集"
            xlabel = '样本编号'
            ylabel = '相对误差 (%)'
            legend_label = '10%误差阈值'

        errors = metrics[param]['relative_errors']

        # 筛除相对误差大于18%的样本
        if param == '拱顶下沉':
            mask = errors <= 10
        else:
            mask = errors <= 18
        filtered_errors = errors[mask]
        filtered_indices = np.arange(len(errors))[mask]

        # 创建新图形
        plt.figure(figsize=(12 / 2.54, 8 / 2.54))

        # 绘制误差
        plt.plot(filtered_indices, filtered_errors, 'o-', label=f'{param}')

        # 添加阈值线
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label=legend_label)

        # 设置标题和标签
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # 添加图例
        plt.legend(loc='upper right')
        plt.grid(True)

        # 保存图形
        plt.savefig(os.path.join(MODEL_DIR, f'{param}_{dataset_type}_errors_{lang}.png'), dpi=300)
        plt.close()


def plot_predictions_vs_index(true, pred, param_names, dataset_type='test', lang='zh', max_samples=100):
    """绘制真实值与预测值随样本索引变化的折线图"""
    # 限制样本数量以避免图表过于密集
    if len(true) > max_samples:
        indices = np.random.choice(len(true), max_samples, replace=False)
        true = true[indices]
        pred = pred[indices]
        sample_indices = np.arange(max_samples)
    else:
        sample_indices = np.arange(len(true))

    # 设置中英文文本
    if lang == 'en':
        title_suffix = f' - {dataset_type.capitalize()} Set'
        true_label = 'True Values'
        pred_label = 'Predictions'
        xlabel = 'Sample Index'
        ylabel = 'Value (mm)'
    else:
        title_suffix = f' - {"测试" if dataset_type == "test" else "训练"}集'
        true_label = '真实值'
        pred_label = '预测值'
        xlabel = '样本编号'
        ylabel = '数值 (mm)'

    # 为每个参数单独绘图
    for i, param in enumerate(param_names):
        param_true = true[:, i]
        param_pred = pred[:, i]

        # 创建新图形
        plt.figure(figsize=(12 / 2.54, 8 / 2.54))

        # 绘制真实值和预测值
        plt.plot(sample_indices, param_true, 'b-', linewidth=1.5, label=true_label)
        plt.plot(sample_indices, param_pred, 'r--', linewidth=1.5, label=pred_label)

        # 设置标题和标签
        plt.title(f'{param}{title_suffix}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # 添加图例
        plt.legend(loc='upper right')
        plt.grid(True)

        # 保存图形
        plt.savefig(os.path.join(MODEL_DIR, f'{param}_{dataset_type}_vs_index_{lang}.png'), dpi=300)
        plt.close()


def print_metrics(metrics, param_names, dataset_type='test'):
    """打印评估指标"""
    print(f"\n{'=' * 50}")
    print(f"Evaluation Metrics - {dataset_type.capitalize()} Set")
    print('=' * 50)

    # 打印各参数指标
    for param in param_names:
        m = metrics[param]
        print(f"{param}:")
        print(f"  R²: {m['r2']:.6f}")
        print(f"  MSE: {m['mse']:.6f}")
        print(f"  MAPE: {m['mape']:.2f}%")
        print(f"  Mean Relative Error: {m['mean_relative_error']:.2f}%")
        print(f"  Max Relative Error: {m['max_relative_error']:.2f}%")
        print('-' * 50)

    # 打印整体指标
    m = metrics['overall']
    print("Overall:")
    print(f"  R²: {m['r2']:.6f}")
    print(f"  MSE: {m['mse']:.6f}")
    print(f"  MAPE: {m['mape']:.2f}%")
    print(f"  Mean Relative Error: {m['mean_relative_error']:.2f}%")
    print('=' * 50)


def export_test_results(true, pred, param_names, param_labels, model_dir):
    """
    导出测试集预测结果表格
    true: 真实值数组 (n_samples, n_params)
    pred: 预测值数组 (n_samples, n_params)
    param_names: 参数名称列表 (英文)
    param_labels: 参数中文标签字典
    model_dir: 保存目录
    """
    # 确保参数名称和标签匹配
    assert len(param_names) == len(param_labels)

    # 计算相对误差百分比
    errors = np.abs((pred - true) / (true + 1e-10)) * 100

    # 创建结果DataFrame
    results = []

    # 添加表头
    header = ["测试样本"]
    for param in param_names:
        header.extend([
            f"{param_labels[param]}真实值",
            f"{param_labels[param]}预测值",
            f"{param_labels[param]}误差(%)"
        ])
    results.append(header)

    # 添加每个样本的数据
    for i in range(len(true)):
        row = [f"样本 {i + 1}"]

        for j, param in enumerate(param_names):
            row.extend([
                f"{true[i, j]:.6f}",
                f"{pred[i, j]:.6f}",
                f"{errors[i, j]:.2f}%"
            ])

        results.append(row)

    # 添加整体统计行
    avg_row = ["平均值"]
    max_row = ["最大值"]
    min_row = ["最小值"]

    for j, param in enumerate(param_names):
        # 平均误差
        avg_error = np.mean(errors[:, j])
        avg_row.extend(["", "", f"{avg_error:.2f}%"])

        # 最大误差
        max_error = np.max(errors[:, j])
        max_row.extend(["", "", f"{max_error:.2f}%"])

        # 最小误差
        min_error = np.min(errors[:, j])
        min_row.extend(["", "", f"{min_error:.2f}%"])

    results.append(avg_row)
    results.append(max_row)
    results.append(min_row)

    # 转换为DataFrame
    df = pd.DataFrame(results[1:], columns=results[0])

    # 保存为CSV
    csv_path = os.path.join(model_dir, "test_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf_8_sig')  # 使用utf_8_sig支持中文

    print(f"测试结果已保存到: {csv_path}")
    return df


# 贝叶斯优化函数，设置优化参数范围，返回验证损失
def objective(trial, train_loader, val_loader, input_dim, device):
    import torch.nn as nn
    from my_model import TimeSeriesTransformer
    from my_train import train_model
    from my_inference import validate_model

    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    nhead = trial.suggest_categorical('nhead', [1, 2, 4])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)

    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(30):  # 较小轮次快速搜索
        train_model(model, train_loader, criterion, optimizer)
        val_loss, _, _ = validate_model(model, val_loader, criterion)

    return val_loss


def plot_optuna_optimization_history(study, save_path, lang='zh'):
    """
    绘制贝叶斯优化过程中验证损失的演化曲线，包含：
    - 验证集损失随轮次变化的折线图
    - 最优点位置标记
    - 最优点横线
    - MSE注释
    """
    trials = study.trials_dataframe()

    plt.figure(figsize=(14 / 2.54, 8 / 2.54))  # 宽14cm，高8cm
    plt.plot(trials['number'], trials['value'], 'o-', label='验证集损失')

    # 获取最优点
    best_idx = trials['value'].idxmin()
    best_trial = trials.iloc[best_idx]
    best_x = best_trial['number']
    best_y = best_trial['value']

    # 绘制最优点
    plt.scatter([best_x], [best_y], color='red', s=100, zorder=5, label='最优点')
    plt.axhline(y=best_y, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='最优MSE')

    # 添加注释
    if lang == 'zh':
        title = '贝叶斯优化过程曲线'
        xlabel = '优化轮次'
        ylabel = '验证损失 (MSE)'
        best_text = f'最优轮次: {int(best_x)}\n最优MSE: {best_y:.6f}'
    else:
        title = 'Bayesian Optimization Progress Curve'
        xlabel = 'Trial Number'
        ylabel = 'Validation Loss (MSE)'
        best_text = f'Best Trial: {int(best_x)}\nBest MSE: {best_y:.6f}'

    # 显示最优点注释文本
    plt.text(best_x, best_y + 0.01, best_text, ha='center', va='bottom', fontsize=9)

    # 设置图例和标题
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_split_data_csv_v2(output_dir, data_splits, scalers, original_order=True):
    os.makedirs(output_dir, exist_ok=True)

    def save_array_as_csv(arr, filename):
        """
        将三维数组 (samples, time_steps, features) 展平为二维保存为 CSV
        每行 = 一个样本展开后的时间序列
        """
        reshaped = arr.reshape(arr.shape[0], -1)
        df = pd.DataFrame(reshaped)
        df.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf_8_sig')

    # === 标准化后的数据：按划分保存 === #
    for split in ['train', 'val', 'test']:
        save_array_as_csv(data_splits[f"X_{split}"], f"inputs_{split}_std.csv")
        save_array_as_csv(data_splits[f"delta_{split}"], f"inputs_{split}_delta_std.csv")
        save_array_as_csv(data_splits[f"mask_{split}"], f"inputs_{split}_mask.csv")
        save_array_as_csv(data_splits[f"y_{split}"], f"outputs_{split}_std.csv")

    print("✅ 已保存标准化后的输入/输出数据（按划分）")

    # === 反标准化后的数据：还原原始10条样本（inputs_raw / outputs_raw） === #
    input_scaler, output_scaler = scalers

    # 合并全部划分回原始顺序（注意这里要按原始顺序合并）
    X_all = np.concatenate([data_splits['X_train'], data_splits['X_val'], data_splits['X_test']], axis=0)
    delta_all = np.concatenate([data_splits['delta_train'], data_splits['delta_val'], data_splits['delta_test']],
                               axis=0)
    y_all = np.concatenate([data_splits['y_train'], data_splits['y_val'], data_splits['y_test']], axis=0)

    # 拼接 delta 与 inputs，一起反标准化
    n_samples, max_len, n_input = X_all.shape
    input_concat = np.concatenate([X_all, delta_all], axis=-1)  # shape: (samples, time, features+1)
    input_concat_reshaped = input_concat.reshape(-1, input_concat.shape[-1])

    input_raw_reshaped = input_scaler.inverse_transform(input_concat_reshaped)
    input_raw = input_raw_reshaped.reshape(n_samples, max_len, -1)
    input_features_raw = input_raw[:, :, :-1]  # 去掉 delta

    output_raw = output_scaler.inverse_transform(y_all)

    # 保存反标准化后的完整原始输入与输出（即10条样本）
    save_array_as_csv(input_features_raw, "inputs_raw.csv")
    save_array_as_csv(output_raw, "outputs_raw.csv")

    print("✅ 已保存反标准化后的完整原始输入/输出数据（共10条样本）")


# ── 在全局样式设置之后插入这段函数 ─────────────────────────
def plot_label_sequence_comparison(raw_labels, scaler, param_names, save_dir):
    """
    为每个输出标签绘制标准化前（raw_labels）和标准化后序列对比。
    折线+圆点，marker 小、线细、半透明，保持全局风格。
    """
    os.makedirs(save_dir, exist_ok=True)
    norm_labels = scaler.transform(raw_labels)
    samples = np.arange(raw_labels.shape[0])

    for i, name in enumerate(param_names):
        fig, ax = plt.subplots(
            figsize=(14/2.54, 7/2.54), dpi=300
        )
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

        ax.set_title(f'{name} 标准化前后序列对比')
        ax.set_xlabel('样本编号')
        ax.set_ylabel(f'{name} (mm)')
        ax.legend(loc='upper right')
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{name}_sequence_comparison.png'))
        plt.close(fig)


# 主函数定义
def main():
    # 加载数据
    def get_csv_data(csv_path: str, max_steps=4):
        df = pd.read_csv(csv_path)
        samples_num = df.shape[0]

        # 分离输入于输出参数
        input_params = df.iloc[:, 5:]
        output_params = df.iloc[:, :5]

        input_params = input_params.values
        output_params = output_params.values

        # 生成模拟时序数据
        time_deltas = np.array([0, 1, 1, 1]).reshape(-1, 1)
        time_steps = np.cumsum(time_deltas).reshape(-1)
        # param_names = ['GD1', 'GD2', 'SL1', 'SL2', 'GJ1']
        param_names = ['拱顶下沉', '拱顶下沉2', '周边收敛1', '周边收敛2', '拱脚下沉']

        data = []
        for i in range(samples_num):
            values = np.tile(input_params[i], (max_steps, 1))
            labels = output_params[i]
            data.append((time_steps, values, time_deltas, labels))

        return data, param_names

    data, param_names = get_csv_data(csv_path)

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

    plot_label_sequence_comparison(
        raw_labels=all_labels,
        scaler=output_scaler,
        param_names=param_names,
        save_dir=MODEL_DIR
    )

    # 保存标准化器
    joblib.dump(input_scaler, os.path.join(scaler_dir, 'input_scaler.pkl'))
    joblib.dump(output_scaler, os.path.join(scaler_dir, 'output_scaler.pkl'))
    print("标准化器已保存到", scaler_dir)

    # 预处理数据
    padded_values, padded_deltas, masks, labels = preprocess_data(
        data, input_scaler=input_scaler, output_scaler=output_scaler
    )

    # 划分数据集: 7:1:2 (训练:验证:测试)
    # 先分训练+验证(80%)和测试(20%)
    X_train_val, X_test, delta_train_val, delta_test, mask_train_val, mask_test, y_train_val, y_test = train_test_split(
        padded_values, padded_deltas, masks, labels, test_size=0.2, random_state=42)

    # 再分训练(70%)和验证(10%)
    X_train, X_val, delta_train, delta_val, mask_train, mask_val, y_train, y_val = train_test_split(
        X_train_val, delta_train_val, mask_train_val, y_train_val, test_size=0.125, random_state=42)  # 0.125*0.8=0.1

    print(f"数据集大小: 训练集={len(y_train)}, 验证集={len(y_val)}, 测试集={len(y_test)}")

    save_split_data_csv_v2(
        output_dir=os.path.join(MODEL_DIR, "split_data_csv"),
        data_splits={
            'X_train': X_train,
            'delta_train': delta_train,
            'mask_train': mask_train,
            'y_train': y_train,
            'X_val': X_val,
            'delta_val': delta_val,
            'mask_val': mask_val,
            'y_val': y_val,
            'X_test': X_test,
            'delta_test': delta_test,
            'mask_test': mask_test,
            'y_test': y_test,
        },
        scalers=(input_scaler, output_scaler)
    )

    # 创建数据集和数据加载器
    train_dataset = GeotechDataset(X_train, delta_train, mask_train, y_train)
    val_dataset = GeotechDataset(X_val, delta_val, mask_val, y_val)
    test_dataset = GeotechDataset(X_test, delta_test, mask_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 损失函数和优化器
    criterion = nn.MSELoss()

    # 控制是否使用贝叶斯优化，True为使用，False为不使用
    use_bayesian_optimization = False  # 设置为True以启用贝叶斯优化

    if use_bayesian_optimization:
        print("🔍 开始贝叶斯优化超参数...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, input_dim=7, device=device),
                       n_trials=50)
        print("✅ 贝叶斯优化完成！最佳参数为：")
        print(study.best_params)

        best_params = study.best_params

        model = TimeSeriesTransformer(
            input_dim=7,
            d_model=best_params['d_model'],
            nhead=best_params['nhead'],
            num_layers=best_params['num_layers']
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

        # 可选保存参数
        import json
        with open(os.path.join(MODEL_DIR, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)

        # 📊 绘制优化历史图
        plot_optuna_optimization_history(
            study,
            save_path=os.path.join(MODEL_DIR, 'optuna_loss_curve_zh.png'),
            lang='zh'
        )
        plot_optuna_optimization_history(
            study,
            save_path=os.path.join(MODEL_DIR, 'optuna_loss_curve_en.png'),
            lang='en'
        )

        # 💾 保存CSV
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(MODEL_DIR, 'optuna_trials.csv'), index=False, encoding='utf_8_sig')
        print("✅ 已保存优化历史为 CSV 和曲线图。")


    else:
        # 若不使用贝叶斯优化，则使用默认参数
        model = TimeSeriesTransformer(
            input_dim=7,
            d_model=256,
            nhead=4,
            num_layers=2
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=8.564825241340346e-05, weight_decay=2.3600012291720694e-07)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # 训练循环
    num_epochs = 300
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, _, _ = validate_model(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1}: 保存新最佳模型，验证损失={val_loss:.6f}")

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 绘制损失曲线
    plot_loss_history(train_losses, val_losses, lang='zh')
    plot_loss_history(train_losses, val_losses, lang='en')

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    print(f"加载最佳模型，验证损失={best_val_loss:.6f}")

    # 在训练集、验证集和测试集上进行预测
    def evaluate_set(loader, output_scaler):
        loss, preds, labels = validate_model(model, loader, criterion)
        preds_orig = output_scaler.inverse_transform(preds)
        labels_orig = output_scaler.inverse_transform(labels)
        return loss, preds_orig, labels_orig

    # 评估所有数据集
    train_loss, train_preds, train_true = evaluate_set(train_loader, output_scaler)
    val_loss, val_preds, val_true = evaluate_set(val_loader, output_scaler)
    test_loss, test_preds, test_true = evaluate_set(test_loader, output_scaler)

    print(f'\n最终损失: 训练集={train_loss:.6f}, 验证集={val_loss:.6f}, 测试集={test_loss:.6f}')

    # 计算评估指标
    train_metrics = calculate_metrics(train_true, train_preds, param_names)
    val_metrics = calculate_metrics(val_true, val_preds, param_names)
    test_metrics = calculate_metrics(test_true, test_preds, param_names)

    # 打印指标
    print_metrics(train_metrics, param_names, 'train')
    print_metrics(val_metrics, param_names, 'val')
    print_metrics(test_metrics, param_names, 'test')

    # 可视化结果
    plot_predictions_individual(train_true, train_preds, param_names, train_metrics, 'train', lang='zh')
    # plot_predictions_individual(val_true, val_preds, param_names, val_metrics, 'val', lang='zh')
    plot_predictions_individual(test_true, test_preds, param_names, test_metrics, 'test', lang='zh')

    # plot_predictions_vs_index(train_true, train_preds, param_names, 'train', lang='zh')
    # plot_predictions_vs_index(val_true, val_preds, param_names, 'val', lang='zh')
    plot_predictions_vs_index(test_true, test_preds, param_names, 'test', lang='zh')

    plot_predictions_combined(train_true, train_preds, test_true, test_preds, param_names, test_metrics, lang='zh')

    plot_relative_errors_per_param(train_metrics, param_names, 'train', lang='zh')
    # plot_relative_errors_per_param(val_metrics, param_names, 'val', lang='zh')
    plot_relative_errors_per_param(test_metrics, param_names, 'test', lang='zh')

    plot_predictions_individual(train_true, train_preds, param_names, train_metrics, 'train', lang='en')
    # plot_predictions_individual(val_true, val_preds, param_names, val_metrics, 'val', lang='en')
    plot_predictions_individual(test_true, test_preds, param_names, test_metrics, 'test', lang='en')

    # plot_predictions_vs_index(train_true, train_preds, param_names, 'train', lang='en')
    # plot_predictions_vs_index(val_true, val_preds, param_names, 'val', lang='en')
    plot_predictions_vs_index(test_true, test_preds, param_names, 'test', lang='en')

    plot_predictions_combined(train_true, train_preds, test_true, test_preds, param_names, test_metrics, lang='en')

    plot_relative_errors_per_param(train_metrics, param_names, 'train', lang='en')
    # plot_relative_errors_per_param(val_metrics, param_names, 'val', lang='en')
    plot_relative_errors_per_param(test_metrics, param_names, 'test', lang='en')

    # 定义参数中文标签
    param_labels = {
        '拱顶下沉': '拱顶下沉',
        '拱顶下沉2': '拱顶下沉2',
        '周边收敛1': '周边收敛1',
        '周边收敛2': '周边收敛2',
        '拱脚下沉': '拱脚下沉',
    }
    # param_labels = {
    #     'GD': '拱顶下沉',
    #     'GD2': '拱顶下沉2',
    #     'SL1': '周边收敛2',
    #     'SL2': '周边收敛1',
    #     'GJ': '拱脚下沉',
    # }

    # 导出测试结果
    test_results_df = export_test_results(
        test_true, test_preds, param_names, param_labels, MODEL_DIR
    )

    # 打印表格前几行
    print("\n测试集预测结果预览:")
    print(test_results_df.head())


if __name__ == "__main__":
    main()

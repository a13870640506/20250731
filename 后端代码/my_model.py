import torch
import torch.nn as nn

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Transformer模型定义
class TimeSeriesTransformer(nn.Module):
    """
    输入: (batch_size, seq_len, input_dim)
    输出: (batch_size, num_outputs)
    """
    def __init__(self, input_dim=5, d_model=128, nhead=8, num_layers=4, num_outputs=5):
        super().__init__()
        
        # 输入嵌入层: 将原始特征映射到高维空间
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码: 为序列添加位置信息 (使用可学习的位置编码)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 4, d_model)) 
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局平均池化: 聚合序列信息
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层: 预测五个岩土参数
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, num_outputs)
        )

    def forward(self, x, time_deltas, src_key_padding_mask=None):
        """
        x: 输入时序数据 (batch_size, seq_len, 3)
        time_deltas: 时间间隔 (batch_size, seq_len, 1)
        src_key_padding_mask: 注意力掩码 (batch_size, seq_len)
        """
        # 拼接原始数据和相对时间差 [拱顶下沉, 周边收敛, 拱脚下沉, 相对时间差, 绝对时间]
        batch_size, seq_len, _ = x.shape
        
        # 计算绝对时间 (累积时间)
        abs_time = torch.cumsum(time_deltas, dim=1)
        
        # 组合特征: [原始3特征, 相对时间差, 绝对时间]
        x = torch.cat([x, time_deltas, abs_time], dim=-1)  # (batch, seq, input_size)
        
        # 输入嵌入
        x = self.input_embedding(x)  # (batch, seq, d_model)

        # 添加位置编码 (截取到实际序列长度)
        x = x + self.positional_encoding[:, :seq_len, :] # (batch, seq, d_model)

        # Transformer编码器, 忽略padding部分
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        
        # 全局平均池化
        x = x.permute(0, 2, 1)  # (batch, d_model, seq)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)       # (batch, d_model)
        
        # 输出层
        return self.output_layer(x)  # (batch, 5)

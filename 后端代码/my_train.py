import torch

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 训练函数
def train_model(model, dataloader, criterion, optimizer):
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



import torch
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 验证函数
def validate_model(model, dataloader, criterion):
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

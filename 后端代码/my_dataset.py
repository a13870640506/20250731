import torch
from torch.utils.data import Dataset

# 3. 自定义数据集
class GeotechDataset(Dataset):
    def __init__(self, values, deltas, masks, labels):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.deltas = torch.tensor(deltas, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'values': self.values[idx],
            'deltas': self.deltas[idx],
            'mask': self.masks[idx],
            'label': self.labels[idx]
        }

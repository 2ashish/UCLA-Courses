import torch

from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class ModalityMatchingDataset(Dataset):
    def __init__(
        self, df_modality1, df_modality2, ch_dim=True
    ):
        super().__init__()
        self.df_modality1 = df_modality1
        self.df_modality2 = df_modality2
        self.ch_dim = ch_dim
    def __len__(self):
        return self.df_modality1.shape[0]
    
    def __getitem__(self, index: int):
        x = self.df_modality1.iloc[index].values
        y = self.df_modality2.iloc[index].values
        if self.ch_dim:
            return torch.from_numpy(x).unsqueeze(0),torch.from_numpy(y)
        else:
            return x, y
        
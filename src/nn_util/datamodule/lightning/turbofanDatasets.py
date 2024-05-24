import numpy as np
from torch.utils.data import Dataset


class TFDataset(Dataset):

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        """

        Args:
            data:
            targets:
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


class TFPredictionDataset(Dataset):

    def __init__(self, data: np.ndarray):
        """

        Args:
            data:
            labels:
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
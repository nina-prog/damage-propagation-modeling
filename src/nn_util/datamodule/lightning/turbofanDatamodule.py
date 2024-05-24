import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.nn_util.datamodule.lightning.turbofanDatasets import TFDataset, TFPredictionDataset


class TurbofanDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()

        self.batch_size = batch_size

        self.test_dataset = None
        self.predict_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def set_test_dataset(self, X_test: np.ndarray, y_test: np.ndarray):
        self.test_dataset = TFDataset(data=X_test, targets=y_test)

    def set_predict_dataset(self, X_predict: np.ndarray):
        self.predict_dataset = TFPredictionDataset(data=X_predict)

    def set_train_dataset(self, X_train: np.ndarray, y_train: np.ndarray):
        self.train_dataset = TFDataset(data=X_train, targets=y_train)

    def set_val_dataset(self, X_val: np.ndarray, y_val: np.ndarray):
        self.val_dataset = TFDataset(data=X_val, targets=y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
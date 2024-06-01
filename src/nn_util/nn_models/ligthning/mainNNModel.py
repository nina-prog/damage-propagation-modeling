import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod


class MainNNModel(pl.LightningModule, ABC):

    def __init__(self, lr=0.001, momentum=0.9, beta_1=0.9, beta_2=0.999, window_size=30, features=24, optimizer='Adam'):
        """

        :param lr: The learning rate from the optimizer.
        :param momentum: The momentum of the optimizer, only relevant for SGD.
        :param beta_1: The first beta term for Adam.
        :param beta_2: The second beta term for Adam.
        :param window_size: The window size of the sliding window.
        :param features: The number of features in the input data.
        :param optimizer: The optimizer to use. Can be either 'Adam' or 'SGD'.
        """
        super().__init__()
        # General variables
        self.lr = lr
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.optimizer = optimizer

        self.loss = torch.nn.MSELoss()

        self.window_size = window_size
        self.features = features

    @abstractmethod
    def forward(self, x):
        pass

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self.forward(x)
        loss = self.loss(pred_y, y)
        self.log('train_loss', loss)
        return {"loss": loss, "preds": pred_y, "targets": y}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self.forward(x)
        loss = self.loss(pred_y, y)
        self.log('val_loss', loss)
        return {"loss": loss, "preds": pred_y, "targets": y}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self.forward(x)
        loss = self.loss(pred_y, y)
        self.log('test_loss', loss)
        return {"loss": loss, "preds": pred_y, "targets": y}

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))

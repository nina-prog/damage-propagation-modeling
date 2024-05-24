import torch
import pytorch_lightning as pl


class MainLSTMModel(pl.LightningModule):

    def __init__(self, lr=0.001, momentum=0.9, beta_1=0.9, beta_2=0.999, window_size=30, features=24,
                 dropout_rate=0.5, optimizer='Adam', default=True):
        """

        :param lr: The learning rate from the optimizer.
        :param momentum: The momentum of the optimizer, only relevant for SGD.
        :param beta_1: The first beta term for Adam.
        :param beta_2: The second beta term for Adam.
        :param window_size: The window size of the sliding window.
        :param features: The number of features in the input data.
        :param dropout_rate: The dropout rate.
        :param optimizer: The optimizer to use. Can be either 'Adam' or 'SGD'.
        :param default: Has to be True if the main class is used and False for all the subclasses.
        """
        super().__init__()
        # General variables
        self.lr = lr
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.window_size = window_size
        self.loss = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Architecture
        if default:
            self.lstm_layer1 = torch.nn.LSTM(input_size=features, hidden_size=128)
            self.lstm_layer2 = torch.nn.LSTM(input_size=128, hidden_size=64)
            self.lstm_layer3 = torch.nn.LSTM(input_size=64, hidden_size=16)

            input_from_conv_layers = window_size * 16
            self.fc1 = torch.nn.Linear(input_from_conv_layers, 64)
            self.fc2 = torch.nn.Linear(64, 64)
            self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm_layer1(x)
        x, _ = self.lstm_layer2(x)
        x, _ = self.lstm_layer3(x)

        x = x.flatten(start_dim=1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self.forward(x)
        loss = self.loss(pred_y, y)
        self.log('train_loss', loss)
        # return loss
        return {"loss": loss, "preds": pred_y, "targets": y}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self.forward(x)
        loss = self.loss(pred_y, y)
        self.log('val_loss', loss)
        # return loss
        return {"loss": loss, "preds": pred_y, "targets": y}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self.forward(x)
        loss = self.loss(pred_y, y)
        print(loss)
        self.log('test_loss', loss)
        # return loss
        return {"loss": loss, "preds": pred_y, "targets": y}

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))

from src.nn_util.nn_models.ligthning.mainNNModel import MainNNModel

import torch


class ExampleLSTMModel(MainNNModel):

    def __init__(self, lr=0.001, momentum=0.9, beta_1=0.9, beta_2=0.999, window_size=30, features=24,
                 dropout_rate=0.5, optimizer='Adam'):
        """

        :param lr: The learning rate from the optimizer.
        :param momentum: The momentum of the optimizer, only relevant for SGD.
        :param beta_1: The first beta term for Adam.
        :param beta_2: The second beta term for Adam.
        :param window_size: The window size of the sliding window.
        :param features: The number of features in the input data.
        :param dropout_rate: The dropout rate.
        :param optimizer: The optimizer to use. Can be either 'Adam' or 'SGD'.
        """
        super().__init__(lr=lr, momentum=momentum, beta_1=beta_1, beta_2=beta_2, window_size=window_size,
                         features=features, optimizer=optimizer)
        # General variables
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Architecture
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

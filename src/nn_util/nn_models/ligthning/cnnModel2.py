from src.nn_util.nn_models.ligthning.mainNNModel import MainNNModel

import torch


class CNNModel2(MainNNModel):

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
        self.layer1_conv = torch.nn.Conv1d(in_channels=features, out_channels=40, kernel_size=5, padding=2)
        self.layer2_conv = torch.nn.Conv1d(in_channels=40, out_channels=40, kernel_size=5, padding=2)
        self.layer3_conv = torch.nn.Conv1d(in_channels=40, out_channels=40, kernel_size=5, padding=2)
        self.layer4_conv = torch.nn.Conv1d(in_channels=40, out_channels=40, kernel_size=5, padding=2)

        input_from_conv_layers = window_size * 40
        self.fc1 = torch.nn.Linear(input_from_conv_layers, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1_conv(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.layer2_conv(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.layer3_conv(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.layer4_conv(x)
        x = torch.relu(x)

        x = x.flatten(start_dim=1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.dropout(x)
        x = self.fc3(x)
        x, _ = torch.max(x, 1)
        return x

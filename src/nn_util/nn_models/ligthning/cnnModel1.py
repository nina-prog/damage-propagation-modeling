from src.nn_util.nn_models.ligthning.mainNNModel import MainNNModel

import torch


class CNNModel1(MainNNModel):

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
        self.layer1_conv = torch.nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, padding=1)
        self.batchnorm_2 = torch.nn.BatchNorm2d(num_features=40)
        self.layer2_conv = torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=1)
        self.batchnorm_3 = torch.nn.BatchNorm2d(num_features=40)
        self.layer3_conv = torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=1)
        self.batchnorm_4 = torch.nn.BatchNorm2d(num_features=40)
        self.layer4_conv = torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=1)
        self.batchnorm_5 = torch.nn.BatchNorm2d(num_features=40)
        self.layer5_conv = torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=1)

        input_from_conv_layers = window_size * features * 40
        self.batchnorm_6 = torch.nn.BatchNorm1d(num_features=input_from_conv_layers)
        self.fc1 = torch.nn.Linear(input_from_conv_layers, 256)
        self.batchnorm_7 = torch.nn.BatchNorm1d(num_features=256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.batchnorm_8 = torch.nn.BatchNorm1d(num_features=128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.batchnorm_9 = torch.nn.BatchNorm1d(num_features=64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, x):

        x = self.layer1_conv(x)
        x = torch.relu(x)

        x = self.batchnorm_2(x)
        x = self.dropout(x)
        x = self.layer2_conv(x)
        x = torch.relu(x)

        x = self.batchnorm_3(x)
        x = self.dropout(x)
        x = self.layer3_conv(x)
        x = torch.relu(x)

        x = self.batchnorm_4(x)
        x = self.dropout(x)
        x = self.layer4_conv(x)
        x = torch.relu(x)

        x = self.batchnorm_5(x)
        x = self.dropout(x)
        x = self.layer5_conv(x)
        x = torch.relu(x)

        x = x.flatten(start_dim=1)

        x = self.batchnorm_6(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.batchnorm_7(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.batchnorm_8(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.relu(x)

        x = self.batchnorm_9(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = torch.relu(x)
        return x

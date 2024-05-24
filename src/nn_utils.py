from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_data(df):
    """
    Scales the numerical columns in the DataFrame using MinMaxScaler.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Scaled DataFrame.
    """
    scaler = MinMaxScaler()
    
    # Select float columns
    float_columns = df.select_dtypes(include=float).columns.tolist()
    
    # Scale the data
    scaled_data = scaler.fit_transform(df[float_columns])
    
    # Update the DataFrame with scaled data
    df[float_columns] = scaled_data

    return df

def create_sliding_window(df, window_size=30, drop_columns=["UnitNumber", "Cycle", "RUL"]):
    """
    Creates a sliding window of data for time series prediction.

    Args:
        df (pandas.DataFrame): Input DataFrame containing time series data.
        window_size (int): Size of the sliding window.
        drop_columns (list): List of columns to drop from the input DataFrame.

    Returns:
        tuple: A tuple containing X (input) and y (output) arrays.
    """
    number_engines = df["UnitNumber"].unique()
    X, y = [], []

    for engine in number_engines:
        # Get data for the current engine
        temp = df[df["UnitNumber"] == engine]
        assert temp["UnitNumber"].unique() == engine

        for i in range(len(temp) - window_size + 1):
            # Extract windowed data and RUL for each window
            X_temp = temp.iloc[i : (i + window_size)].drop(columns=drop_columns)
            Y_temp = temp.iloc[(i + window_size - 1)]["RUL"]
            assert len(X_temp) == window_size
            X.append(X_temp.to_numpy())
            y.append(Y_temp)
            if i == (len(temp) - window_size):
                assert Y_temp == 1

    X = np.array(X)
    y = np.array(y)

    return X, y


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, feature_size)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, feature_size: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.feature_size = feature_size
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(feature_size, dropout)
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(feature_size, 1) 
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, feature_size)
            mask: Optional mask of shape (seq_len, seq_len)
            
        Returns:
            out: Tensor of shape (batch_size, 1) for regression
        """
        # Add positional encoding
        x = self.positional_encoding(x)# (seq_len, batch_size, feature_size)
        x = x.to(torch.float32)

        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask)  # (seq_len, batch_size, feature_size)
    
        # Take the mean across the sequence length dimension
        x = torch.mean(x, dim=0)  # (batch_size, feature_size)
        
        # Output layer
        out = self.fc_out(x)  # (batch_size, 1)
    
        return out
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Example dataset class
class TurbofanDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Training function
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    count = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if count % 44 == 0:
            print(f"--> {count}/{len(dataloader)}")
        count += 1
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
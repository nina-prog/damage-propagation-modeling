import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR


# Example dataset class
class TurbofanDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.from_numpy(data).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    def __init__(self, feature_size: int, num_heads: int, num_layers: int, project_dim : int, window_size: int = 30, dropout: float = 0.05):
        super(TransformerModel, self).__init__()
        self.feature_size = feature_size
        self.project_dim = project_dim
        #self.intermediate_dim = intermediate_dim
        # pseudo emb
        self.project_emb = nn.Linear(feature_size, project_dim)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(project_dim, dropout)
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=project_dim, nhead=num_heads, dropout=dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.bn = nn.BatchNorm1d(window_size)
        # Fully Connected layers to output
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(window_size * project_dim, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 64)       # Second fully connected layer 
        self.fc3 = nn.Linear(64, 1) 
        self.act = nn.ReLU()
        
        #self.intermediate = nn.Linear(project_dim, intermediate)
        # Output layer
        #self.fc_out = nn.Linear(intermediate_dim, 1) 
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, feature_size)
            mask: Optional mask of shape (seq_len, seq_len)
            
        Returns:
            out: Tensor of shape (batch_size, 1) for regression
        """
        #print(f"Input dim: {x.shape}")
        # Pseudo projection
        x = self.bn(x)
        x = self.project_emb(x)
        #print(f"Projection dim: {x.shape}")
        x = self.bn(x)
        # Add positional encoding
        x = self.positional_encoding(x)# (seq_len, batch_size, feature_size)
        x = x.to(torch.float32)
        #print(f"Positional dim: {x.shape}")
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask)  # (seq_len, batch_size, feature_size)
        #print(x.var())
        #print(f"Transformer dim: {x.shape}")

        x = self.flatten(x)
        #print(f"Flatten: {x.shape}")
        x = self.fc1(x)
        x = self.act(x)
        #print(f"FC 1: {x.shape}")
        x = self.fc2(x)
        x = self.act(x)
        #print(f"FC2: {x.shape}")
        out = self.fc3(x)
        
        # Take the mean across the sequence length dimension
        #x = torch.mean(x, dim=1)# (batch_size, feature_size)
        #print(x.var())
        #print(f"Mean dim: {x.shape}")
        #print(f"After Median: {x.shape}")
        # Output layer
        #out = self.fc_out(x)  # (batch_size, 1)
        #print(f"Out dim: {out.shape}")
        #print(out.var())

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
        
        #if count % 44 == 0:
        #    print(f"--> {count}/{len(dataloader)}")
        count += 1
        optimizer.zero_grad()
        
        outputs = model(inputs)
        targets = targets.view(-1, 1)
        
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

    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)
            outputs = model(inputs)
            outputs = torch.round(outputs)
            outputs = torch.round(outputs)
            
            # Set minimal value to 1
            min_value = 1
            outputs = torch.where(outputs < min_value, torch.tensor(min_value), outputs)


            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            #if count == 1 or count == 10:
                #print(count)
                #print(outputs[:10], targets[:10])
            count += 1
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
import torch
import torch.nn as nn

class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size = 128, num_layers = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
#CNN-1d
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        out = self.conv(x)
        out = out.mean(dim=2)  # Global average pooling
        out = self.fc(out)
        return out
    
# TCN
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out
    
class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[64, 128], kernel_size=3):
        super().__init__()
        layers = []
        in_channels = input_size
        for out_channels in num_channels:
            layers.append(TCNBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        out = self.tcn(x)
        out = out.mean(dim=2)  # Global average pooling
        out = self.fc(out)
        return out

#Tranformers
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        self.transformer = nn.Transformer(d_model=input_size, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        out = self.transformer(x)
        out = out.mean(dim=0)  # Global average pooling
        out = self.fc(out)
        return out
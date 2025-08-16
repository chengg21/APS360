import torch
import torch.nn as nn
from cnn_model import CNN

class CRNN_WRN(nn.Module):
    def __init__(self, feature_c=2048, hidden_size=256, num_layers=2):
        super().__init__()
        self.name = "test13"
        self.num_classes = ord("~") - ord(" ") + 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_in = feature_c

        cnn = CNN() # from Layer 0 (conv1) to Layer 7 (layer 4)
        checkpoint = torch.load('cnn_models/CNN_lr0.0001_bs128_decay1e-10_epoch200/CNN_epoch_20.pth')
        cnn.load_state_dict(checkpoint['model_cnn_state_dict'])

        self.cnn = nn.Sequential(
            cnn.cnn[7],  # ResNet layer4
            cnn.proj
        )

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=0.2)
        self.rnn = nn.LSTM(
            input_size = 512,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2) 
        x = x.permute(0, 2, 1) 
        x = self.dropout(x)
        # x: (B, W, C) = (batch_size, 148, 2048)
        rnn_out, _ = self.rnn(x)         # (batch size, timesteps, 2*hidden_size)
        logits = self.fc(rnn_out)      # (batch size, timesteps, NUM_CLASSES)
        return logits.permute(1, 0, 2)   # (timesteps, batch size, NUM_CLASSES)
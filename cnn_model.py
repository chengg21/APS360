import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
class CNN(nn.Module):
    """
    Input:  (B, 1, H, W)  (H will be normalized to 32 in transforms)
    Output: (B, W', C)    where C=out_channels (default 512)
    """
    def __init__(self, embed_dim = 512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2]) #removes last two layers (pooling and fc)

        for name, param in self.cnn.named_parameters():
            # Freeze all layers except last block (layer4)
            if "layer4" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)


    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.cnn(x)           # (B, 512, H', W')
        x = self.proj(x)   # (B, embed_dim, H', W')
        # Collapse height to 1, keep width as sequence
        x = x.mean(dim=2)       # (B, embed_dim, W')

        # Permute to (B, W', embed_dim) for RNN input
        x = x.permute(0, 2, 1)       # (B, W', embed_dim)
        return x               # sequence for RNN

class EMNISTClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(-1)  # (B, C)
        x = self.head(x)         # (B, num_classes)
        return x
    
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        self.name = "CNN"
        self.cnn = CNN(embed_dim)
        self.classifier = EMNISTClassifier(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x
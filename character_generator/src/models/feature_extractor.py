import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super(FeatureExtractor, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, output_dim)
        
    def forward(self, x):
        # Ensure input is in the correct format (B, C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Extract features using ResNet
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
            
            # Normalize features to ensure consistency
            x = F.normalize(x, p=2, dim=1)
            
            # Project to lower dimension while preserving feature relationships
            x = self.fc(x)
            
            # Additional normalization to ensure stable parameter generation
            x = torch.tanh(x)
        
        return x

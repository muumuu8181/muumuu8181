import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterGenerator(nn.Module):
    def __init__(self, feature_dim=512, attr_dim=3, skill_dim=1):
        super(CharacterGenerator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_features = nn.Linear(256, feature_dim)
        self.fc_attr = nn.Linear(feature_dim, attr_dim)
        self.fc_skill = nn.Linear(feature_dim, skill_dim)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = self.fc_features(x)
        
        attributes = F.relu(self.fc_attr(features))  # HP, ATK, DEF
        skill = torch.sigmoid(self.fc_skill(features))  # Skill probability
        
        return attributes, skill

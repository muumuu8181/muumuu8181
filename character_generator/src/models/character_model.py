import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterGenerator(nn.Module):
    def __init__(self, feature_dim=512, attr_dim=3, skill_dim=1):
        super(CharacterGenerator, self).__init__()
        self.fc_attr = nn.Linear(feature_dim, attr_dim)
        self.fc_skill = nn.Linear(feature_dim, skill_dim)
    
    def forward(self, x):
        attributes = F.relu(self.fc_attr(x))  # HP, ATK, DEF
        skill = torch.sigmoid(self.fc_skill(x))  # Skill probability
        
        return attributes, skill

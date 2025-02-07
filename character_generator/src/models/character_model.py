import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterGenerator(nn.Module):
    def __init__(self, feature_dim=512):
        super(CharacterGenerator, self).__init__()
        # 基本パラメータ
        self.fc_hp = nn.Linear(feature_dim, 1)
        self.fc_mp = nn.Linear(feature_dim, 1)
        # 戦闘パラメータ
        self.fc_attack = nn.Linear(feature_dim, 1)
        self.fc_defense = nn.Linear(feature_dim, 1)
        self.fc_speed = nn.Linear(feature_dim, 1)
        # 魔法パラメータ
        self.fc_magic = nn.Linear(feature_dim, 1)
        # その他
        self.fc_luck = nn.Linear(feature_dim, 1)
        # 特技（5つまで）
        self.fc_skills = nn.Linear(feature_dim, 5)
    
    def forward(self, x):
        # パラメータを生成（HPは最大10000、他は最大1000）
        hp = torch.sigmoid(self.fc_hp(x)) * 10000
        mp = torch.sigmoid(self.fc_mp(x)) * 1000
        attack = torch.sigmoid(self.fc_attack(x)) * 1000
        defense = torch.sigmoid(self.fc_defense(x)) * 1000
        speed = torch.sigmoid(self.fc_speed(x)) * 1000
        magic = torch.sigmoid(self.fc_magic(x)) * 1000
        luck = torch.sigmoid(self.fc_luck(x)) * 1000
        
        # 特技の確率（上位5つを選択）
        skills = torch.sigmoid(self.fc_skills(x))
        
        return {
            'base': {'hp': hp, 'mp': mp},
            'battle': {'attack': attack, 'defense': defense, 'speed': speed},
            'magic': {'magic': magic},
            'other': {'luck': luck},
            'skills': skills
        }

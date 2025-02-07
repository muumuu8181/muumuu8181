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
        with torch.no_grad():
            # 特徴量から一貫したパラメータを生成
            feature_norm = torch.norm(x, p=2, dim=1, keepdim=True)
            
            # HPは特徴量の強さに基づいて生成（5000-10000の範囲）
            hp = 5000 + torch.abs(self.fc_hp(x)) * 5000
            
            # その他のパラメータは特徴量の方向性に基づいて生成（0-1000の範囲）
            mp = torch.abs(self.fc_mp(x)) * 1000
            attack = torch.abs(self.fc_attack(x)) * 1000
            defense = torch.abs(self.fc_defense(x)) * 1000
            speed = torch.abs(self.fc_speed(x)) * 1000
            magic = torch.abs(self.fc_magic(x)) * 1000
            luck = torch.abs(self.fc_luck(x)) * 1000
            
            # 特技の確率（上位5つを選択）
            skills = torch.sigmoid(self.fc_skills(x))
            
            return {
                'base': {'hp': hp, 'mp': mp},
                'battle': {'attack': attack, 'defense': defense, 'speed': speed},
                'magic': {'magic': magic},
                'other': {'luck': luck},
                'skills': skills
            }

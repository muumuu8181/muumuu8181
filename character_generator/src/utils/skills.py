from enum import Enum
import torch

class Skill(Enum):
    # 攻撃魔法
    FIRE_MAGIC = 1     # 炎魔法
    ICE_MAGIC = 2      # 氷魔法
    THUNDER_MAGIC = 3  # 雷魔法
    
    # 回復・支援魔法
    HEALING = 4        # 回復魔法
    SHIELD_MAGIC = 5   # 防御魔法
    HASTE_MAGIC = 6    # 加速魔法
    
    # 物理技
    DOUBLE_ATTACK = 7  # 二段攻撃
    COUNTER = 8        # カウンター
    SHIELD_WALL = 9    # シールドウォール
    CRITICAL_HIT = 10  # 会心の一撃
    
    # 特殊技
    LIFE_STEAL = 11    # HP吸収
    MP_DRAIN = 12      # MP吸収
    LUCKY_STRIKE = 13  # 幸運の一撃
    BERSERK = 14      # 狂戦士化
    MEDITATION = 15    # 瞑想（MP回復）

def get_top_skills(skill_probs: torch.Tensor, num_skills: int = 5) -> list[Skill]:
    """確率の高い順に特技を選択する"""
    # 確率の高い順にインデックスを取得
    _, indices = torch.sort(skill_probs, descending=True)
    skills = []
    
    # 上位num_skills個の特技を選択
    for idx in indices[:num_skills]:
        skill_index = idx.item()
        if skill_index < len(Skill):
            skills.append(list(Skill)[skill_index])
    
    return skills

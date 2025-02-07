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

def get_top_skills(skill_probs: list[float], num_skills: int = 5) -> list[Skill]:
    """確率の高い順に特技を選択する"""
    # インデックスと確率のペアを作成
    indexed_probs = list(enumerate(skill_probs))
    # 確率の高い順にソート
    sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
    skills = []
    
    # 上位num_skills個の特技を選択
    for idx, _ in sorted_probs[:num_skills]:
        if idx < len(Skill):
            skills.append(list(Skill)[idx])
    
    return skills

from enum import Enum

class Skill(Enum):
    FIRE_MAGIC = 1     # 炎魔法
    ICE_MAGIC = 2      # 氷魔法
    HEALING = 3        # 回復魔法
    DOUBLE_ATTACK = 4  # 二段攻撃
    COUNTER = 5        # カウンター
    SHIELD_WALL = 6    # シールドウォール
    CRITICAL_HIT = 7   # 会心の一撃
    SPEED_BOOST = 8    # 俊敏強化

def get_skill_by_value(value: float) -> Skill:
    """値を特技にマッピングする（0-1の値を8種類の特技に変換）"""
    skill_index = min(int(value * 8), 7)  # 0-7のインデックスに変換
    return list(Skill)[skill_index]

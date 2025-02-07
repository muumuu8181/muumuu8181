import random
from src.utils.skills import Skill

def generate_random_character(id: int):
    # 基本パラメータ（平均250程度）
    hp = random.randint(4000, 10000)  # HP
    mp = random.randint(200, 1000)    # MP
    attack = random.randint(100, 1000) # 攻撃力
    defense = random.randint(100, 1000) # 防御力
    speed = random.randint(100, 1000)  # 素早さ
    magic = random.randint(100, 1000)  # 魔力
    luck = random.randint(100, 1000)   # 運
    
    # ランダムに5つの特技を選択
    available_skills = list(Skill)
    skills = random.sample(available_skills, 5)
    
    return {
        'id': id,
        'base': {'hp': hp, 'mp': mp},
        'battle': {'attack': attack, 'defense': defense, 'speed': speed},
        'magic': {'magic': magic},
        'other': {'luck': luck},
        'skills': [skill.name for skill in skills]
    }

def main():
    characters = []
    for i in range(10):
        character = generate_random_character(i + 1)
        characters.append(character)
        
        # キャラクター情報の表示
        print(f"\nキャラクター {character['id']}:")
        print("-------------------")
        print("基本パラメータ:")
        print(f"HP: {character['base']['hp']}")
        print(f"MP: {character['base']['mp']}")
        
        print("\n戦闘パラメータ:")
        print(f"攻撃力: {character['battle']['attack']}")
        print(f"防御力: {character['battle']['defense']}")
        print(f"素早さ: {character['battle']['speed']}")
        
        print("\n魔法パラメータ:")
        print(f"魔力: {character['magic']['magic']}")
        
        print("\nその他:")
        print(f"運: {character['other']['luck']}")
        
        print("\n特技:")
        for skill in character['skills']:
            print(f"- {skill}")
            
        # 平均値の計算（HPを除く）
        params = [
            character['base']['mp'],
            character['battle']['attack'],
            character['battle']['defense'],
            character['battle']['speed'],
            character['magic']['magic'],
            character['other']['luck']
        ]
        avg = sum(params) / len(params)
        print(f"\n平均値（HP除く）: {avg:.1f}")

if __name__ == "__main__":
    main()

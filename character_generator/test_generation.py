import sys
import os

# モジュールパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from character_generator.src.pipeline import CharacterGenerationPipeline
from character_generator.src.utils.skills import get_top_skills

def main():
    # パイプラインの初期化
    pipeline = CharacterGenerationPipeline()
    
    # テスト画像のパス
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "character_generator", "data", "test_images", "test_character.png")
    
    try:
        # キャラクター生成
        result = pipeline.process_image(image_path)
        
        # 結果の表示
        print("\nキャラクター生成結果:")
        print("-------------------")
        print("基本パラメータ:")
        print(f"HP: {result['base']['hp']:.1f}")
        print(f"MP: {result['base']['mp']:.1f}")
        
        print("\n戦闘パラメータ:")
        print(f"攻撃力: {result['battle']['attack']:.1f}")
        print(f"防御力: {result['battle']['defense']:.1f}")
        print(f"素早さ: {result['battle']['speed']:.1f}")
        
        print("\n魔法パラメータ:")
        print(f"魔力: {result['magic']['magic']:.1f}")
        
        print("\nその他:")
        print(f"運: {result['other']['luck']:.1f}")
        
        print("\n特技:")
        for skill in get_top_skills(result['skills']):
            print(f"- {skill.name}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()

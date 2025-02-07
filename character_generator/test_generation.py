from src.pipeline import CharacterGenerationPipeline
from src.utils.skills import get_skill_by_value

def main():
    # パイプラインの初期化
    pipeline = CharacterGenerationPipeline()
    
    # テスト画像のパス
    image_path = "data/test_images/test_character.png"
    
    try:
        # キャラクター生成
        result = pipeline.process_image(image_path)
        
        # 結果の表示
        print("\nキャラクター生成結果:")
        print("-------------------")
        print(f"HP: {result['attributes']['hp']:.1f}")
        print(f"攻撃力: {result['attributes']['attack']:.1f}")
        print(f"防御力: {result['attributes']['defense']:.1f}")
        
        # 特技の取得と表示
        skill = get_skill_by_value(result['skill'])
        print(f"特技: {skill.name}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()

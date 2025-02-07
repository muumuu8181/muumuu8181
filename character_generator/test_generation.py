import os
import torch
from src.pipeline import CharacterGenerationPipeline

def test_character_generation():
    pipeline = CharacterGenerationPipeline()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "character_generator", "data", "test_images", "test_character.png")
    
    try:
        # 同じ画像から2回キャラクターを生成
        result1 = pipeline.process_image(image_path)
        result2 = pipeline.process_image(image_path)
        
        # パラメータの一貫性をテスト
        assert result1['base']['hp'] == result2['base']['hp']
        assert result1['base']['mp'] == result2['base']['mp']
        assert result1['battle']['attack'] == result2['battle']['attack']
        assert result1['battle']['defense'] == result2['battle']['defense']
        assert result1['battle']['speed'] == result2['battle']['speed']
        assert result1['magic']['magic'] == result2['magic']['magic']
        assert result1['other']['luck'] == result2['other']['luck']
        assert result1['skills'] == result2['skills']
        
        # パラメータの範囲をテスト
        assert 5000 <= result1['base']['hp'] <= 10000
        assert 0 <= result1['base']['mp'] <= 1000
        assert 0 <= result1['battle']['attack'] <= 1000
        assert 0 <= result1['battle']['defense'] <= 1000
        assert 0 <= result1['battle']['speed'] <= 1000
        assert 0 <= result1['magic']['magic'] <= 1000
        assert 0 <= result1['other']['luck'] <= 1000
        assert len(result1['skills']) == 5
        
        print("Character generation test passed!")
        print(f"Generated character parameters: {result1}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_character_generation()

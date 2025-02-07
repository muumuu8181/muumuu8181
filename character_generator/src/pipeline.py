import torch
from .models.object_detector import ObjectDetector
from .models.feature_extractor import FeatureExtractor
from .models.character_model import CharacterGenerator
from .utils.image_processor import ImageProcessor
from .utils.skills import get_top_skills

class CharacterGenerationPipeline:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.feature_extractor = FeatureExtractor()
        self.character_generator = CharacterGenerator()
        self.image_processor = ImageProcessor()
        
        # 一貫性を保証するため評価モードに設定
        self.feature_extractor.eval()
        self.character_generator.eval()
    
    @torch.no_grad()
    def process_image(self, image_path):
        # 画像の読み込みと前処理
        image_tensor = self.image_processor.preprocess_image(image_path)
        
        # 特徴抽出（正規化済み）
        features = self.feature_extractor(image_tensor)
        
        # キャラクター属性と特技の生成
        character_data = self.character_generator(features)
        
        # パラメータの整形
        result = {
            'base': {
                'hp': int(character_data['base']['hp'].item()),
                'mp': int(character_data['base']['mp'].item())
            },
            'battle': {
                'attack': int(character_data['battle']['attack'].item()),
                'defense': int(character_data['battle']['defense'].item()),
                'speed': int(character_data['battle']['speed'].item())
            },
            'magic': {
                'magic': int(character_data['magic']['magic'].item())
            },
            'other': {
                'luck': int(character_data['other']['luck'].item())
            }
        }
        
        # 特技の選択（確率の高い順に5つ）
        skill_probs = character_data['skills'].squeeze().tolist()
        result['skills'] = get_top_skills(skill_probs)
        
        return result

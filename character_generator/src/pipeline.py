import torch
from .models.object_detector import ObjectDetector
from .models.feature_extractor import FeatureExtractor
from .models.character_model import CharacterGenerator
from .utils.image_processor import ImageProcessor

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
        # 1. 画像の読み込みと前処理
        image_tensor = self.image_processor.preprocess_image(image_path)
        
        # 2. 特徴抽出
        features = self.feature_extractor(image_tensor)
        
        # 3. キャラクター属性と特技の生成
        features = features.view(features.size(0), -1)
        attributes, skill = self.character_generator(features)
        
        # バッチの最初の要素を取得
        attributes = attributes[0]  # バッチサイズ1なので最初の要素を取得
        skill = skill[0]  # バッチサイズ1なので最初の要素を取得
        
        return {
            'attributes': {
                'hp': float(attributes[0].item() * 100),  # 0-100のスケールに変換
                'attack': float(attributes[1].item() * 100),
                'defense': float(attributes[2].item() * 100)
            },
            'skill': float(skill.item())
        }

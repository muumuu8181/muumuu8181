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
        image = self.image_processor.preprocess_image(image_path)
        
        # 2. 物体検出
        bbox = self.object_detector.detect_object(image)
        if bbox is None:
            raise ValueError("No object detected in the image")
            
        # 3. 物体の切り出しと正規化
        x1, y1, x2, y2 = bbox
        object_image = image[:, :, y1:y2, x1:x2]
        object_image = self.image_processor.normalize(object_image)
        
        # 4. 特徴抽出
        features = self.feature_extractor(object_image)
        
        # 5. キャラクター属性と特技の生成
        attributes, skill = self.character_generator(features)
        
        return {
            'attributes': {
                'hp': attributes[0].item(),
                'attack': attributes[1].item(),
                'defense': attributes[2].item()
            },
            'skill': skill.item()
        }

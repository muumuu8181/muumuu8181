from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import torch
from typing import Tuple, List
from app.utils.logger import logger

class ImageRecognitionService:
    def __init__(self):
        model_name = "microsoft/resnet-50"
        logger.log("INFO", "initializing_model", model=model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)
        logger.log("INFO", "model_initialized", model=model_name)

    def predict(self, image: Image.Image) -> List[Tuple[str, float]]:
        try:
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predictions = [
                (self.model.config.id2label[i], float(p))
                for i, p in enumerate(probs[0].tolist())
                if float(p) > 0.1
            ]
            
            logger.log("INFO", "prediction_completed", 
                      prediction_count=len(predictions),
                      top_prediction=predictions[0] if predictions else None)
            
            return predictions
        except Exception as e:
            logger.log("ERROR", "prediction_failed", error=str(e))
            raise

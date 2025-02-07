import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
    def detect_object(self, image):
        # Convert image to tensor and normalize
        image_tensor = F.to_tensor(image)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Filter predictions based on confidence
        mask = predictions['scores'] > self.confidence_threshold
        boxes = predictions['boxes'][mask]
        scores = predictions['scores'][mask]
        
        if len(boxes) == 0:
            return None
            
        # Return the box with highest confidence
        best_box = boxes[scores.argmax()].int().tolist()
        return best_box  # [x1, y1, x2, y2]

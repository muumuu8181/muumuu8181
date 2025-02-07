import cv2
import numpy as np
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = self.transform(image).numpy()
        return image
        
    def normalize(self, image_tensor):
        return torch.nn.functional.interpolate(
            image_tensor,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

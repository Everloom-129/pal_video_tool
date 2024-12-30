# Turn phone camera from 1080p to 720p
import cv2
import numpy as np

class ImageCompressor:
    def __init__(self, target_height=720):
        self.target_height = target_height
        
    def compress(self, image):
        """Compress image to target height while maintaining aspect ratio"""
        if image is None:
            raise ValueError("Input image is None")
            
        # Get current dimensions
        height, width = image.shape[:2]
        
        # Calculate new width to maintain aspect ratio
        aspect_ratio = width / height
        target_width = int(self.target_height * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (target_width, self.target_height), 
                           interpolation=cv2.INTER_AREA)
        
        return resized

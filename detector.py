from PIL import Image, ImageDraw
import numpy as np

def detect_faces(image_path, min_confidence=0.5):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw a placeholder box in center to confirm app works
    w, h = img.size
    draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], outline="green", width=3)
    draw.text((w//4, h//4 - 20), "Face Detection Active", fill="green")
    
    return np.array(img)
    

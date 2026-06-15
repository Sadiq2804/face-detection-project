from PIL import Image, ImageDraw
import numpy as np

def detect_faces(image_path, min_confidence=0.5):
    from ultralytics import YOLO
    model = YOLO("yolov8n-face.pt")
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    results = model(image_path)
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence >= min_confidence:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1 - 15), f"{confidence:.0%}", fill="green")
    return np.array(img)
    

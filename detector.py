from PIL import Image, ImageDraw
import numpy as np
from deepface import DeepFace

def detect_faces(image_path, min_confidence=0.5):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        results = DeepFace.extract_faces(image_path, enforce_detection=False)
        for face in results:
            region = face["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            confidence = face.get("confidence", 0)
            if confidence >= min_confidence:
                draw.rectangle([x, y, x + w, y + h], outline="green", width=3)
                draw.text((x, y - 15), f"{confidence:.0%}", fill="green")
    except Exception as e:
        draw.text((10, 10), f"Error: {str(e)}", fill="red")

    return np.array(img)

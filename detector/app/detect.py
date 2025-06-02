from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

def detect_objects(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img)[0]
    
    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name in ["car", "person"]:
            detections.append({
                "label": cls_name,
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
    
    return {"detections": detections}
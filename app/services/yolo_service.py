from ultralytics import YOLO
import cv2

# =====================================================
# LOAD MODEL YOLO
# =====================================================

model = YOLO("app/models/best.pt")


# =====================================================
# DETEKSI OBJECT
# =====================================================

def detect_objects(image_path):

    results = model(image_path)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    return detections
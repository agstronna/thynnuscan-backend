import os
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
from keras.models import load_model
from keras.applications.mobilenet_v3 import preprocess_input

yolo_model = None
classification_model = None

target_names = ['Busuk', 'Mulai Busuk', 'Cukup Segar', 'Segar']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, classification_model
    yolo_path = "./models/best.pt"
    keras_path = "./models/model_ikan_final.keras"

    if not os.path.exists(yolo_path):
        print(f"⚠️ WARNING: YOLO model not found at {yolo_path}")
    else:
        yolo_model = YOLO(yolo_path)
        print(f"✅ YOLOv8 Loaded: {yolo_path}")

    if not os.path.exists(keras_path):
        print(f"⚠️ WARNING: MobileNet model not found at {keras_path}")
    else:
        classification_model = load_model(keras_path)
        print(f"✅ MobileNetV3 Loaded: {keras_path}")

    yield

app = FastAPI(title="Deteksi Ikan API", version="1.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "API Deteksi Ikan is running. Use POST /predict to analyze an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if yolo_model is None or classification_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")

    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Run YOLO Detection (dengan batas minimal confidence 78% agar sangat ketat menyaring false positive)
        results = yolo_model(img_bgr, conf=0.78, verbose=False)
        
        list_hasil = []

        if len(results[0].boxes) == 0:
            return JSONResponse({
                "status": "success",
                "data": {
                    "conclusion": {
                        "status": "Tidak Ditemukan",
                        "final_status_score": -1,
                        "remaining_hours": 0.0,
                        "message": "Objek mata atau insang tidak ditemukan"
                    },
                    "detections": []
                }
            })

        h_img, w_img, _ = img_rgb.shape

        for i, box in enumerate(results[0].boxes):
            # Coordinates from YOLO
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label_objek = results[0].names[int(box.cls[0])]

            # 2. Crop with Padding & Predict
            pad = 15
            y1_pad, y2_pad = max(0, y1 - pad), min(h_img, y2 + pad)
            x1_pad, x2_pad = max(0, x1 - pad), min(w_img, x2 + pad)

            # Crop padded area
            crop = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]

            # Resize with INTER_AREA
            img_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

            # Preprocessing MobileNetV3 (Range -1 to 1)
            img_prep = np.array(img_resized, dtype=np.float32)
            img_input = np.expand_dims(img_prep, axis=0)
            img_input = preprocess_input(img_input)

            # Multi-task Prediction
            pred_klass, pred_regr = classification_model.predict(img_input, verbose=0)

            # Classification Result
            idx_stat = int(np.argmax(pred_klass[0]))
            status = target_names[idx_stat]
            conf = float(np.max(pred_klass[0])) * 100

            # Regression Result (Max 18 hours)
            jam = float(pred_regr[0][0]) * 18.0

            list_hasil.append({
                'objek': label_objek.upper(),
                'status': status,
                'idx': idx_stat,
                'conf': conf,
                'jam': max(0.0, jam),
                'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
            })

        # 4. Summary & Final Conclusion
        final_status_score = 4 # Start higher than max index
        final_status_name = ""
        final_jam = 99.0

        for hasil in list_hasil:
            # Update status if worse (smaller index)
            if hasil['idx'] < final_status_score:
                final_status_score = hasil['idx']
                final_status_name = hasil['status']

            # Update hours if shorter
            if hasil['jam'] < final_jam:
                final_jam = hasil['jam']

        return JSONResponse({
            "status": "success",
            "data": {
                "conclusion": {
                    "status": final_status_name.upper(),
                    "final_status_score": final_status_score,
                    "remaining_hours": final_jam
                },
                "detections": list_hasil
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
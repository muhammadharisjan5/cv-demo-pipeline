from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List, Tuple
import cv2
import numpy as np
import io
from cv_pipeline import ObjectDetector, process_video

app = FastAPI()

model = ObjectDetector()

@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image"}
    
    detections = model.predict(image)
    return {"detections": detections}

@app.post("/detect_video")
async def detect_video(rtsp_url: str):
    stop_signal = process_video(rtsp_url, model)
    return {"message": "Video processing started", "stop_signal": str(stop_signal)}

from fastapi import FastAPI
import os
import io
import base64
import numpy as np
from PIL import Image
import json
from pydantic import BaseModel


from eyeq.inference_engine import InferenceEngine
from eyeq import Yolov5onnxDet
from eyeq import Yolov6onnxDet
from eyeq import Yolov7onnxDet
from eyeq import Yolov8onnxDet


class InferData(BaseModel):
    data: str
    model_id: str


WEIGHTS_DIR = os.path.join(os.getcwd(), "data", "weights")


inference_engine = InferenceEngine()
rest_app = FastAPI()


@rest_app.get("/")
def read_root():
    return {"Hello": "World"}


@rest_app.post("/load_model")
def load_model(model_path: str, model_type: str):
    detection_model = None
    model_id = None
    if model_type == "yolov5_detector_onnx":
        detection_model = Yolov5onnxDet()
    elif model_type == "yolov6_detector_onnx":
        detection_model = Yolov6onnxDet()
    elif model_type == "yolov7_detector_onnx":
        detection_model = Yolov7onnxDet()
    if detection_model:
        inference_engine.register_model(model=detection_model)
        model_id = detection_model.model_id
        model_path = os.path.join(WEIGHTS_DIR, model_path)
        inference_engine.load_network(model_id=model_id, model_path=model_path)
    return {"status": True, "model_id": model_id, "model_path": model_path}


@rest_app.get("/active_models")
def get_active_models():
    active_models = inference_engine.get_active_models()
    return {"status": True, "active_models": active_models}


@rest_app.get("/available_models")
def get_available_models():
    models = os.listdir(WEIGHTS_DIR)
    return {"status": True, "models": models}


@rest_app.post("/detect")
def detect(data: InferData):
    data = data.dict()
    model_id = data['model_id']
    buf = io.BytesIO(base64.b64decode(data['data']))
    image = Image.open(buf)
    image = np.array(image)
    image = image[:, :, ::-1].copy()
    detections = inference_engine.forward(model_id=model_id, img=image)

    results = []
    if detections:
        boxes = detections.xyxy
        labels = detections.class_id
        scores = detections.confidence

        for label, score, box in zip(labels, scores, boxes):
            xtl = int(box[0])
            ytl = int(box[1])
            xbr = int(box[2])
            ybr = int(box[3])

            results.append({
                "confidence": str(score),
                "label": str(label),
                "box": [xtl, ytl, xbr, ybr],
                "type": "rectangle"
            })
    res = json.dumps(results)
    return {"status": True, "detections": res}









from eyeq.utils.video_capture_extended import VideoCaptureExtended
from eyeq.utils.painter import draw_boxes, draw_tracklets
from eyeq.inference_engine import InferenceEngine
import cv2
import time

from eyeq import Yolov5onnxDet, Sort

video_path = "../data/images/pig.mp4"
model_path = "../data/weights/yolov5s_pig_counter.onnx"


cap = VideoCaptureExtended(video_path)


inference_engine = InferenceEngine()
detector = Yolov5onnxDet(conf_thresh=0.3, iou_thresh=0.5)

inference_engine.register_model(model=detector)
inference_engine.load_network(model_id=detector.model_id, model_path=model_path)


tracker = Sort()

time.sleep(2)
counter = 0
while True:
    ret, img = cap.read()

    if not ret:
        break

    detections = inference_engine.forward(model_id=detector.model_id, img=img)
    tracklets = tracker.update_public(detections)

    if tracklets:
        img = draw_tracklets(detections=tracklets, image=img)

    cv2.imshow("image", img)
    cv2.waitKey(1)

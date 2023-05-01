from eyeq.utils.video_capture_extended import VideoCaptureExtended
from eyeq.detectors.yolov5.yolov5_onnx import V5ONNX
from eyeq.utils.painter import draw_boxes, draw_tracklets
from eyeq.inference_engine import InferenceEngine

from eyeq.trackers.byte_track.byte_tracker import BYTETracker



import cv2
import time


class ByteTrackConfig:
    frame_rate = 25
    track_high_thresh = 0.4  # threshold for the first association
    track_low_thresh = 0.1  # threshold for the second association
    new_track_thresh = 0.3  # threshold for init new track if the detection does not match any tracks
    track_buffer = 60  # buffer to calculate the time when to remove tracks
    match_thresh = 0.8


video_path = "../data/images/2.mp4"
model_path = "../data/weights/yolov5s.onnx"


cap = VideoCaptureExtended(video_path)


inference_engine = InferenceEngine()
detector = V5ONNX(conf_thresh=0.1, iou_thresh=0.5)

tracker = BYTETracker(config=ByteTrackConfig)

inference_engine.register_model(model=detector)
inference_engine.load_network(model_id=detector.model_id, model_path=model_path)

time.sleep(2)
counter = 0
while True:
    ret, img = cap.read()

    if not ret:
        break
    detections = inference_engine.forward(model_id=detector.model_id, img=img)
    tracklets = tracker.update(results=detections, img=img)
    if tracklets:
        img = draw_tracklets(detections=tracklets, image=img)
    cv2.imshow("image", img)
    cv2.waitKey(1)


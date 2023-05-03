import time

import cv2
import numpy as np

from eyeq.detectors.yolov5.yolov5_onnx import Yolov5onnxDet
from eyeq.inference_engine import InferenceEngine
from eyeq.trackers.byte_track.byte_tracker import BYTETracker
from eyeq.utils.painter import draw_tracklets
from eyeq.utils.video_capture_extended import VideoCaptureExtended
from eyeq.utils.tracer import TraceAnnotator, Trace

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
detector = Yolov5onnxDet(conf_thresh=0.1, iou_thresh=0.5)

tracker = BYTETracker(config=ByteTrackConfig)

inference_engine.register_model(model=detector)
inference_engine.load_network(model_id=detector.model_id, model_path=model_path)

time.sleep(2)
counter = 0

trace = Trace()
trace_annotator = TraceAnnotator()

# frame x1, y1, x2, y2, class, track id

detection_list = np.zeros(shape=(0, 7))  # [[frame, x1, y1, x2, y2, class, object_id]]

counter = 0
while True:
    ret, img = cap.read()

    if not ret:
        break
    detections = inference_engine.forward(model_id=detector.model_id, img=img)
    tracklets = tracker.update(results=detections, img=img)
    if tracklets:
        trace.update(detections=tracklets, frame_counter=counter)
        boxes = tracklets.xyxy
        tracker_id = tracklets.tracker_id
        class_id = tracklets.class_id
        new_detections = np.zeros(shape=(boxes.shape[0], 7))
        new_detections[:, 0] = counter
        new_detections[:, 1:5] = boxes
        new_detections[:, 5] = class_id.astype(int)
        new_detections[:, 6] = tracker_id

        detection_list = np.append(detection_list, new_detections, axis=0)

        # img = draw_tracklets(detections=tracklets, image=img)
    img = trace_annotator.annotate(scene=img, trace=trace)
    cv2.imshow("image", img)
    cv2.waitKey(1)
    counter += 1

np.save("tracklets.npy", detection_list)




import cv2
import numpy as np
import supervision as sv
from eyeq.utils.painter import draw_tracklets
from eyeq.utils.video_capture_extended import VideoCaptureExtended
from eyeq.utils.tracer import TraceAnnotator, Trace

video_path = "../data/images/pig.mp4"
cap = VideoCaptureExtended(video_path)

counter = 0
detection_list = np.load("tracklets.npy")

trace = Trace()
trace_annotator = TraceAnnotator()
box_annotator = sv.BoxAnnotator()


counter = 0
while True:
    ret, img = cap.read()

    if not ret:
        break
    detections = sv.Detections.empty()
    tracklets = detection_list[detection_list[:, 0] == counter]
    if tracklets.shape[0] >0:
        detections = sv.Detections(xyxy=tracklets[:, 1:5], class_id=tracklets[:, 5].astype(int), tracker_id=tracklets[:, -1])


    trace.update(detections=detections, frame_counter=counter)
        # img = box_annotator.annotate(detections=detections, scene=img)

    trace_annotator.annotate(scene=img, trace=trace)
        # img = draw_tracklets(detections=tracklets, image=img)
    cv2.imshow("image", img)
    key = cv2.waitKey(20)
    counter += 1

    if key == ord('q'):
        quit()


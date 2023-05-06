import glob
import time

import cv2
import numpy as np
import supervision as sv
from eyeq.utils.painter import draw_boxes, draw_tracklets
from eyeq.utils.general import get_image_list, get_file_list
from eyeq.utils.namespaces import FileTypes

from eyeq import Yolov5onnxDet, BYTETracker, NmsSort, Sort, MoTDataset, MOTBenchmark, OCSort
from eyeq.utils.fps_monitor import FpsMonitor

mot_sequence = "../data/MOT20/train/MOT20-01"
image_dir = f"{mot_sequence}/img1"
gt_path = f"{mot_sequence}/gt/gt.txt"

run_evaluation = False

fps_monitor = FpsMonitor()
# tracker = BYTETracker()
# tracker = Sort()
# tracker = OCSort()

mot_data = MoTDataset(video_sequence_name=mot_sequence, tracker_name=tracker.tracker_name)
image_list = sorted(get_image_list(image_dir))
groundtruths = mot_data.read_mot_gt()

for f in range(len(image_list)):
    img = cv2.imread(image_list[f])
    valid = np.where(groundtruths[:, 0] == f)
    tracklets_frame = groundtruths[valid]

    detections = sv.Detections.empty()
    if tracklets_frame.shape[0] > 0:
        # frame, id, x1, y1, w, h, score, -1, -1, -1
        xywh = tracklets_frame[:, 2:6]
        xywh[:, 2] += xywh[:, 0]
        xywh[:, 3] += xywh[:, 1]
        tracker_id = tracklets_frame[:, 1]
        confidence = tracklets_frame[:, 6]
        class_id = np.full(xywh.shape[0], 1)
        detections = sv.Detections(xyxy=xywh, class_id=class_id, tracker_id=tracker_id, confidence=confidence)

    tracklets_track = tracker.update_public(detections=detections)
    # if tracklets_track:
    #     img = draw_tracklets(img, tracklets_track)
    mot_data.add_tracklets(frame=f, tracklets=tracklets_track)

    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("MOT", img)
    cv2.waitKey(1)

mot_data.write()
import os
import numpy as np
import supervision as sv


class MoTDataset:

    def __init__(self, video_sequence_name: str, tracker_name: str=None):
        self.video_sequence_name = video_sequence_name
        self.tracker_name = tracker_name
        if not os.path.exists(f"{video_sequence_name}/{tracker_name}"):
            os.mkdir(f"{video_sequence_name}/{tracker_name}")
        self.det_file = f"{video_sequence_name}/{tracker_name}/det.txt"
        self.gt_file = f"{video_sequence_name}/gt/gt.txt"
        # frame, id, x1, y1, w, h, score, -1, -1, -1
        self.tracker_tracklets = np.zeros((0, 10))

    def add_tracklets(self, frame: int, tracklets: sv.Detections):
        n_tracks = tracklets.xyxy.shape[0]
        new_tracklets = np.zeros((n_tracks, 10))
        if n_tracks:
            new_tracklets[:, 0] = frame
            boxes = tracklets.xyxy
            boxes = self.xyxy2xywh(boxes)
            new_tracklets[:, 1] = tracklets.tracker_id
            new_tracklets[:, 2:6] = boxes
            new_tracklets[:, 6] = 1
            new_tracklets[:, 7] = -1
            new_tracklets[:, 8] = -1
            new_tracklets[:, 9] = -1

            valid = np.where(new_tracklets[:, 1] > 0)
            new_tracklets = new_tracklets[valid]
            self.tracker_tracklets = np.append(self.tracker_tracklets, new_tracklets, axis=0)

    def write(self):
        # self.tracker_tracklets = self.tracker_tracklets[self.tracker_tracklets[:, 1].argsort()]
        np.savetxt(self.det_file, self.tracker_tracklets, delimiter=',', fmt='%d')

    @staticmethod
    def xyxy2xywh(boxes):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        boxes[:, 2] = w
        boxes[:, 3] = h
        return boxes

    def read_mot_gt(self):
        return self.read_mot_file(self.gt_file)

    def read_mot_det(self):
        return self.read_mot_file(self.det_file)

    @staticmethod
    def read_mot_file(source):
        data = np.loadtxt(source, delimiter=',')
        return data



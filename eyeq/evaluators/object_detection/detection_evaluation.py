import os
import supervision as sv
import numpy as np
from tqdm import tqdm
from eyeq.evaluators.object_detection.metrics import ap_per_class_torch, ConfusionMatrix
from eyeq.utils.box_utils import iou_matrix


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = iou_matrix(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            _X1 = np.vstack([x[0], x[1]]).transpose()
            _x2 = iou[x[0], x[1]][:, None]
            matches = np.concatenate([_X1, _x2], axis=1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


class DetectionEvaluator:

    def __init__(self, plot=False):
        self.groundtruths = []
        self.predictions = []
        self.n_c = 1
        self.iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.size
        self.plot = plot
        # self.confusion_matrix = ConfusionMatrix(nc=self.n_c)

    def set_classes(self, class_maps: dict):
        self.class_maps = class_maps
        self.n_c = len(class_maps.keys())
        if self.plot:
            self.confusion_matrix = ConfusionMatrix(nc=self.n_c)  # overwrite confusion matrix

    def add_groundtruth(self, gt: sv.Detections) -> None:
        self.groundtruths.append(gt)

    def add_prediction(self, det: sv.Detections):
        self.predictions.append(det)

    def evaluate(self):
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        jdict, stats, ap, ap_class = [], [], [], []
        seen = 0
        pbar = tqdm(self.groundtruths, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for batch_i, (_) in enumerate(pbar):
            gt_target = self.groundtruths[batch_i]
            det_target = self.predictions[batch_i]

            det_class_ids = det_target.class_id
            det_xyxy = det_target.xyxy
            det_confidence = det_target.confidence
            det_targets = np.concatenate([det_xyxy, det_confidence[:, None], det_class_ids[:, None]], axis=1)

            gt_class_ids = gt_target.class_id
            gt_xyxy = gt_target.xyxy
            gt_targets = np.concatenate([gt_class_ids[:, None], gt_xyxy], axis=1)

            nl, npr = gt_targets.shape[0], det_targets.shape[0]  # number of labels, predictions
            correct = np.zeros((npr, self.niou), dtype=np.bool)  # init
            if npr == 0:
                if nl:
                    stats.append((correct, *np.zeros((2, 0)), gt_targets[:, 0]))
                    if self.plot:
                        self.confusion_matrix.process_batch(detections=None, labels=gt_targets[:, 0])
            det_targetsn = det_targets.copy()
            if nl:
                correct = process_batch(det_targetsn, gt_targets, self.iouv)

                if self.plot:
                    self.confusion_matrix.process_batch(det_targetsn, gt_targets)
                stats.append((correct, det_targetsn[:, 4], det_targetsn[:, 5], gt_targets[:, 0]))  # (correct, conf, pcls, tcls)
            seen += 1

        stats = [np.concatenate(x, 0) for x in zip(*stats)]

        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class_torch(*stats, plot=True, save_dir=os.getcwd(), names=self.class_maps)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.n_c)  # number of targets per class

        # Print results
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        if self.plot:
            self.confusion_matrix.plot(normalize=True, save_dir=os.getcwd(), names=self.class_maps.values())
        # total image, total instances, precision, recall, map50, map50:95
        return seen, nt.sum(), mp, mr, map50, map


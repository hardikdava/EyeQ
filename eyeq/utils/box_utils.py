import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def denormalize(detections: np.ndarray, img_w: int, img_h: int):
    detections[:, 0] *= img_w
    detections[:, 1] *= img_h
    detections[:, 2] *= img_w
    detections[:, 3] *= img_h
    return detections


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy




def iou(box1: np.ndarray, box2: np.ndarray):
    """ Intersection over union

    :param box1: array (n, 4) -> n x (x1, y1, x2, y2)
    :param box2: array (n, 4) -> n x (x1, y1, x2, y2)
    :return (n) -> n x iou
    """
    xA = np.maximum(box1[:, 0], box2[:, 0])
    yA = np.maximum(box1[:, 1], box2[:, 1])
    xB = np.minimum(box1[:, 2], box2[:, 2])
    yB = np.minimum(box1[:, 3], box2[:, 3])
    interArea = np.abs((np.maximum((xB - xA), 0)) * np.maximum((yB - yA), 0))
    boxAArea = np.abs((box1[:, 0] - box1[:, 2]) * (box1[:, 1] - box1[:, 3]))
    boxBArea = np.abs((box2[:, 0] - box2[:, 2]) * (box2[:, 1] - box2[:, 3]))
    iou = interArea / np.maximum((boxAArea + boxBArea - interArea), 1)
    return iou


def iou_matrix(box1: np.ndarray, box2: np.ndarray):
    """
    Returns an n_box1 x n_box2 matrix with calculated ious between all box pairs
    :param box1: array (n, 4) -> n x (x1, y1, x2, y2)
    :param box2: array (m, 4) -> m x (x1, y1, x2, y2)
    :return (n, m) -> n x m x iou
    """
    if len(box2.shape) == 3:
        box2 = box2[:, :, 0]
    _box1 = np.repeat(box1, box2.shape[0], axis=0)
    _box2 = np.tile(box2, (box1.shape[0], 1))
    ious = iou(_box1, _box2)
    ious = np.reshape(ious, (box1.shape[0], box2.shape[0]))
    return ious

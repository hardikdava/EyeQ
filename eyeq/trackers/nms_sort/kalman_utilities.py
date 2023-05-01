import numpy as np


def invert_covariance_matrix(x):
    """
    Performing the Gauss-Jordan algorithm on a matrix with non-zero elements on the diagonal axis and on elements
    [0,4], [1,5], [2,6], [4,0], [5,1], [6,2] in a fast way.
    :param x: matrix of size [n, 7, 7]
    :return: matrix of size [n, 7, 7] with
    """
    inv = np.zeros_like(x)
    inv[:, :, :] = np.eye(7)[None]
    # First Step
    multiplier = x[:, [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]][:, :, None]
    multiplier = 1 / np.maximum(0.00001, multiplier)
    inv = inv * multiplier
    x = x * multiplier
    # Step two
    multiplier = x[:, [4, 5, 6], [0, 1, 2]][:, :, None]
    inv[:, -3:] -= inv[:, :3] * multiplier
    x[:, -3:] -= x[:, :3] * multiplier
    # Step three
    multiplier = x[:, [4, 5, 6], [4, 5, 6]][:, :, None]
    multiplier = 1 / np.maximum(0.00001, multiplier)
    inv[:, -3:] *= multiplier
    x[:, -3:] *= multiplier
    # Step four
    multiplier = x[:, [0, 1, 2], [4, 5, 6]][:, :, None]
    inv[:, :3] -= inv[:, -3:] * multiplier
    return inv


def convert_bboxes_to_z(bboxes, initialized=False):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]] where x1,y1 is the top left and x2,y2 is the bottom right for each boxes
    """
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    # Try not to divide by zero
    w = w + 0.000001
    h = h + 0.000001
    x = bboxes[:, 0] + w / 2.
    y = bboxes[:, 1] + h / 2.
    s = w * h  # scale is just area
    r = w / h
    if initialized:
        _extra =  [[0] * 3] * x.shape[0]
        _tmp = np.column_stack((x, y, s, r))
        return np.hstack([_tmp, _extra])
    else:
        return np.column_stack((x, y, s, r))


def convert_x_to_bboxes(x, score=None):
    """
    Takes a bounding box in the centre form [[x,y,s,r],[x,y,s,r],[x,y,s,r]] and returns it in the form
      [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    if x.ndim == 3:
        x = x[:, 0]

    w = np.sqrt(np.abs(x[:, 2]) * np.abs(x[:, 3]))

    # if np.max(np.isnan(w) * 1) == 1)
    # h = x[:, 2] / np.maximum(1, w)
    h = x[:, 2] /  w
    x1 = x[:, 0] - w / 2.
    y1 = x[:, 1] - h / 2.
    x2 = x[:, 0] + w / 2.
    y2 = x[:, 1] + h / 2.
    if (score == None):
        return np.column_stack((x1, y1, x2, y2))
    else:
        return np.column_stack((x1, y1, x2, y2, score))



def iom(box1: np.ndarray, box2: np.ndarray):
    """ Intersection over minimum

    :param box1: array (n, 4) -> n x (x1, y1, x2, y2)
    :param box2: array (n, 4) -> n x (x1, y1, x2, y2)
    :return (n) -> n x iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box1[:, 0], box2[:, 0])
    yA = np.maximum(box1[:, 1], box2[:, 1])
    xB = np.minimum(box1[:, 2], box2[:, 2])
    yB = np.minimum(box1[:, 3], box2[:, 3])
    # compute the area of intersection rectangle
    interArea = np.abs((np.maximum((xB - xA), 0)) * np.maximum((yB - yA), 0))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.abs((box1[:, 0] - box1[:, 2]) * (box1[:, 1] - box1[:, 3]))
    boxBArea = np.abs((box2[:, 0] - box2[:, 2]) * (box2[:, 1] - box2[:, 3]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iom = interArea / np.maximum(np.minimum(boxAArea, boxBArea), 1)
    return iom


def iom_matrix(box1: np.ndarray, box2: np.ndarray):
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
    ioms = iom(_box1, _box2)
    ioms = np.reshape(ioms, (box1.shape[0], box2.shape[0]))
    return ioms



def iou(box1: np.ndarray, box2: np.ndarray):
    """ Intersection over union

    :param box1: array (n, 4) -> n x (x1, y1, x2, y2)
    :param box2: array (n, 4) -> n x (x1, y1, x2, y2)
    :return (n) -> n x iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box1[:, 0], box2[:, 0])
    yA = np.maximum(box1[:, 1], box2[:, 1])
    xB = np.minimum(box1[:, 2], box2[:, 2])
    yB = np.minimum(box1[:, 3], box2[:, 3])
    # compute the area of intersection rectangle
    interArea = np.abs((np.maximum((xB - xA), 0)) * np.maximum((yB - yA), 0))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.abs((box1[:, 0] - box1[:, 2]) * (box1[:, 1] - box1[:, 3]))
    boxBArea = np.abs((box2[:, 0] - box2[:, 2]) * (box2[:, 1] - box2[:, 3]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
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
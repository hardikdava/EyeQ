import numpy as np
from eyeq.utils.box_utils import xywh2xyxy


def prefilter_non_max_suppression(boxes, scores, threshold):
    """ Performs some steps of NMS for likely neighbours
    """
    # Sort boxes based on distance. This will order most likely ious as neighbours
    dist = np.power(boxes[:, 0] + boxes[:, 2], 2) + np.power(boxes[:, 1] + boxes[:, 3], 2)
    idxs = np.argsort(dist)
    # Calc IoUs
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inter_min = np.maximum(boxes[idxs[0:-1], 0:2], boxes[idxs[1:], 0:2])
    inter_max = np.minimum(boxes[idxs[0:-1], 2:4], boxes[idxs[1:], 2:4])
    intersection = np.maximum(0, inter_max - inter_min)
    intersection = intersection[:, 0] * intersection[:, 1]
    union = areas[idxs[0:-1]] + areas[idxs[1:]] - intersection
    iou = intersection / np.maximum(0.001, union)
    # Check if overlap is bigger than threshold and remove the one with less confidence. Check left and right neighbour
    is_overlap = iou > threshold
    left_is_low_score = scores[idxs[0:-1]] < scores[idxs[1:]]
    remove = np.zeros_like(areas)
    remove[:-1] = (is_overlap * left_is_low_score) * 1
    remove[1:] = np.maximum(remove[1:], (is_overlap * ~left_is_low_score) * 1)
    # Keep is the opposite of remove
    keep = np.where(remove == 0)[0]
    return idxs[keep]


def nms(boxes, scores, threshold):
    """Returns a list of indexes of objects passing the NMS.
    Args:
      objects: result candidates.
      threshold: the threshold of overlapping IoU to merge the boxes.
    Returns:
      A list of indexes containings the objects that pass the NMS.
    """
    if len(boxes) == 1:
        return [0]
    # return nms(boxes, scores, threshold)
    # ### Try to prefilter based on very simple heuristics ###
    keep = np.arange(boxes.shape[0])
    for i in range(5):
        if boxes.shape[0] <= 30:
            break
        _keep = prefilter_non_max_suppression(boxes, scores, threshold)
        if keep.size == _keep.size:
            break
        # return _keep
        boxes, scores, keep = boxes[_keep], scores[_keep], keep[_keep]

    if len(keep) <= 1:
        return keep

    # ### Run optimal NMS ###
    idxs = np.argsort(scores)[::-1]
    boxes = boxes[idxs]
    # Calculate IoUs for the upper right triangular matrix
    roi = np.triu(np.ones((boxes.shape[0], boxes.shape[0])), 1) == 1
    inter_xmin = np.maximum(boxes[:, 0][:, None], boxes[:, 0][None, :])[roi]
    inter_ymin = np.maximum(boxes[:, 1][:, None], boxes[:, 1][None, :])[roi]
    inter_xmax = np.minimum(boxes[:, 2][:, None], boxes[:, 2][None, :])[roi]
    inter_ymax = np.minimum(boxes[:, 3][:, None], boxes[:, 3][None, :])[roi]
    w = np.maximum(0, inter_xmax - inter_xmin)
    h = np.maximum(0, inter_ymax - inter_ymin)
    intersections = w * h
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = (areas[:, None] + areas[None, :])[roi] - intersections

    ious = np.zeros((len(boxes), len(boxes)))
    ious[roi] = intersections / np.maximum(0.001, union)

    if np.max(ious) < threshold:
        return keep
    # NMS
    rows, cols = np.where(ious > threshold)
    ids_to_keep = [i for i in range(len(boxes))]
    ids_to_remove = list()
    for r, c in zip(rows, cols):
        if r in ids_to_remove:
            continue
        if c not in ids_to_remove:
            ids_to_remove.append(c)
            ids_to_keep.remove(c)

    res = keep[idxs[ids_to_keep]]
    return res

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=300):

    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 640  # (pixels) minimum and maximum box width and height
    max_nms = 6000  # maximum number of boxes into torchvision.ops.nms()

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf = np.amax(x[:, 5:85], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:85], axis=1).reshape(conf.shape)
        x = np.concatenate((box, conf, j.astype(float)), axis=1)[conf.flatten() > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[:max_nms]]  # sort by confidence

        # Batched NMS
        if agnostic:
            c = x[:, 5:6] * 0  # classes
        else:
            c = x[:, 5:6] * max_wh  # classes

        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms(boxes, scores, iou_thres)  # NMS

        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
    return output

import uuid
import cv2
import numpy as np
import onnxruntime as ort
from eyeq.utils.non_maximum_suppression import nms
import supervision as sv
from eyeq.utils.onnx_utils import OnnxDetectors
from eyeq.utils.box_utils import xywh2xyxy


class V6ONNX(OnnxDetectors):

    def __init__(self, conf_thresh=0.4, iou_thresh=0.5, max_det=300):
        super().__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det

        self.labelmap = None
        self.is_inititated = False

    def load_network(self, model_path: str):
        self.model = ort.InferenceSession(model_path)
        self.get_input_details()
        self.get_output_details()
        self.is_inititated = True

    def infer(self, img: np.ndarray, agnostic=False):

        full_image, net_image, pad = self._get_image_tensor(img)
        net_image = net_image.transpose((2, 0, 1))

        net_image /= 255
        net_image = np.expand_dims(net_image, 0)
        output = self.model.run(self.output_names, {self.model.get_inputs()[0].name: net_image})[0]

        predictions = np.asarray(output)[0]

        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_thresh]
        obj_conf = obj_conf[obj_conf > self.conf_thresh]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_thresh]
        scores = scores[scores > self.conf_thresh]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = predictions[:, :4]
        boxes = xywh2xyxy(boxes)
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        valid = nms(boxes, scores, self.iou_thresh)

        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]

        boxes = self._process_predictions(boxes, full_image, pad)

        detections = sv.Detections(xyxy=boxes, class_id=class_ids.astype(int), confidence=scores)
        return detections
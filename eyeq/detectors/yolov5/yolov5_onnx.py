import uuid
import cv2
import numpy as np
import onnxruntime as ort
from eyeq.utils.non_maximum_suppression import non_max_suppression
import supervision as sv
from eyeq.utils.onnx_utils import OnnxDetectors


class V5ONNX(OnnxDetectors):

    def __init__(self, conf_thresh=0.4, iou_thresh=0.5, max_det=300):
        super().__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.labelmap = None
        self.is_inititated = False

    def set_labelmap(self, labelmap: dict):
        self.labelmap = labelmap

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

        output = np.asarray(output)

        pred = non_max_suppression(output, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, agnostic=agnostic)

        pred = np.array(pred[0])

        boxes = pred[:, 0:4]
        boxes = self._process_predictions(boxes, full_image, pad)

        class_ids = pred[:, -1].astype(int)
        confidence = pred[:, 4]
        detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidence)
        return detections



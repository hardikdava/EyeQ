import uuid
import numpy as np
import onnxruntime as ort
import cv2
import supervision as sv
from eyeq.utils.non_maximum_suppression import nms
from eyeq.utils.onnx_utils import OnnxDetectors


class V7ONNX(OnnxDetectors):
    def __init__(self,  conf_thresh=0.35, iou_thresh=0.5, max_det=300, use_cuda=False):
        super().__init__()
        self.model = None
        self.model_id = str(uuid.uuid4())
        self.is_inititated = False
        self.cuda = use_cuda
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det

    def load_network(self, model_path):
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model_path, providers=providers, sess_options=so)
            self.get_input_details()
            self.get_output_details()

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model_path}: {e}")

    def infer(self, img: np.ndarray):

        full_image, net_image, pad = self._get_image_tensor(img)
        net_image = net_image.transpose((2, 0, 1))

        net_image /= 255
        net_image = np.expand_dims(net_image, 0)
        inp = {self.input_names[0]: net_image}
        output = self.model.run(self.output_names, inp)[0]

        output = np.asarray(output)
        boxes = output[:, 1:5]
        boxes = self._process_predictions(boxes, full_image, pad)

        valid = nms(boxes, output[:, -1], self.iou_thresh)  # NMS

        output = output[valid]
        class_ids = output[:, 5].astype(int)
        confidences = output[:, -1]

        valid = np.where(confidences > self.conf_thresh)[0]

        boxes = boxes[valid]
        class_ids = class_ids[valid]
        confidences = confidences[valid]

        detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidences)
        return detections



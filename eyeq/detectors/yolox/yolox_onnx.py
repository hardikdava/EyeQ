import uuid
import cv2
import numpy as np
import onnxruntime as ort
from eyeq.utils.non_maximum_suppression import non_max_suppression, nms
import supervision as sv
from eyeq.utils.onnx_utils import OnnxDetectors
from eyeq.utils.box_utils import xywh2xyxy


class VXONNX(OnnxDetectors):

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
        net_image = net_image.transpose((2, 0, 1)).astype(np.float32)

        # net_image /= 255
        net_image = np.expand_dims(net_image, 0)
        output = self.model.run(self.output_names, {self.model.get_inputs()[0].name: net_image})[0]

        output = np.asarray(output)

        predictions = self._postprocess(output[0], self.input_shape)


        boxes = predictions[:, :4]

        scores = predictions[:, 4:5] * predictions[:, 5:]



        confidences = np.amax(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        valid = np.where(confidences > self.conf_thresh)[0]
        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid].astype(int)

        boxes = xywh2xyxy(boxes)
        valid = nms(boxes, confidences, self.iou_thresh)
        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid].astype(int)

        boxes = self._process_predictions(boxes, full_image, pad)

        detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidences)
        return detections

    def _postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs



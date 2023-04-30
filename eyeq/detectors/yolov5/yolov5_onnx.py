import uuid
import cv2
import numpy as np
import onnxruntime as ort
from eyeq.utils.non_maximum_suppression import non_max_suppression
import supervision as sv


class V5ONNX:

    def __init__(self, conf_thresh=0.4, iou_thresh=0.5, max_det=300):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.model_id = str(uuid.uuid4())
        self.input_shape = None
        self.input_names = None
        self.output_names = None
        self.labelmap = None
        self.is_inititated = False

    def set_labelmap(self, labelmap: dict):
        self.labelmap = labelmap

    def load_network(self, model_path: str):
        self.model = ort.InferenceSession(model_path)
        self.get_input_details()
        self.get_output_details()
        self.is_inititated = True

    def get_input_details(self):
        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_image_shape = model_inputs[0].shape
        self.input_shape = (input_image_shape[2], input_image_shape[3])

    def get_output_details(self):
        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def infer(self, img: np.ndarray, agnostic=False):

        full_image, net_image, pad = self._get_image_tensor(img)
        net_image = net_image.transpose((2, 0, 1))

        net_image /= 255
        net_image = np.expand_dims(net_image, 0)
        output = self.model.run(self.output_names, {self.model.get_inputs()[0].name: net_image})[0]

        output = np.asarray(output)

        pred = non_max_suppression(output, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, agnostic=agnostic)

        pred = np.array(pred[0])
        pred = self._process_predictions(pred, full_image, pad)

        boxes = pred[:, 0:4]
        class_ids = pred[:, -1].astype(int)
        confidence = pred[:, 4]
        detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidence)
        return detections

    def _get_scaled_coords(self, xyxy, output_image, pad):
        """
        Converts raw prediction bounding box to orginal
        image coordinates.

        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        pad_w, pad_h = pad
        in_h, in_w = self.input_shape
        out_h, out_w, _ = output_image.shape

        ratio_w = out_w / (in_w - pad_w)
        ratio_h = out_h / (in_h - pad_h)

        xyxy[:, 0] *= ratio_w
        xyxy[:, 1] *= ratio_h
        xyxy[:, 2] *= ratio_w
        xyxy[:, 3] *= ratio_h

        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, out_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, out_h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, out_w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, out_h)

        return xyxy.astype(int)

    def _process_predictions(self, det, output_image, pad):
        """
        Process predictions and optionally output an image with annotations
        """
        if len(det):
            det[:, :4] = self._get_scaled_coords(det[:, :4], output_image, pad)

        return det

    @staticmethod
    def _resize_and_pad(image, desired_size):
        old_size = image.shape[:2]
        ratio = float(desired_size / max(old_size))
        new_size = tuple([int(x * ratio) for x in old_size])
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]

        # new_size should be in (width, height) format
        image = cv2.resize(image, (new_size[1], new_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pad = (delta_w, delta_h)

        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
                                    value=color)
        return new_im, pad

    def _get_image_tensor(self, img):
        """
        Reshapes an input image into a square with sides max_size
        """
        new_im, pad = self._resize_and_pad(img, self.input_shape[0])
        new_im = np.asarray(new_im, dtype=np.float32)
        return img, new_im, pad

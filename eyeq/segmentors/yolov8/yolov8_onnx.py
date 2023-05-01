import cv2
import numpy as np
import onnxruntime as ort
from eyeq.utils.non_maximum_suppression import nms
import supervision as sv
from eyeq.utils.onnx_utils import OnnxDetectors
from eyeq.utils.box_utils import sigmoid, xywh2xyxy


class Yolov8onnxSeg(OnnxDetectors):

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
        original_shape = full_image.shape

        net_image = net_image.transpose((2, 0, 1))

        net_image /= 255
        net_image = np.expand_dims(net_image, 0)
        output = self.model.run(self.output_names, {self.model.get_inputs()[0].name: net_image})

        protos = np.asarray(output[1])
        predictions = np.asarray(output[0])

        predictions = np.squeeze(predictions).T
        nm = 32
        nc = predictions.shape[1] - nm - 4
        scores = np.max(predictions[:, 4:4+nc], axis=1)
        valid = scores > self.conf_thresh

        predictions = predictions[valid, :]
        scores = scores[valid]
        class_ids = np.argmax(predictions[:, 4:4+nc], axis=1)
        boxes = predictions[:, :4]
        masks_preds = predictions[:, 4+nc:]

        boxes = xywh2xyxy(boxes)

        valid = nms(boxes, scores, self.iou_thresh)
        boxes = boxes[valid]
        class_ids = class_ids[valid]
        confidence = scores[valid]
        mask_pred = masks_preds[valid]

        scaled_boxes = boxes.copy()
        boxes = self._process_predictions(boxes, full_image, pad)

        mask_maps = self.process_mask_output(boxes, scaled_boxes, mask_pred, protos, original_shape)

        detections = sv.Detections(xyxy=boxes, class_id=class_ids.astype(int), confidence=confidence, mask=mask_maps)
        return detections

    def process_mask_output(self, scaled_up_boxes, original_boxes, mask_predictions, mask_output, original_shape):

        if mask_predictions is None:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        scale_boxes = original_boxes * 0.25  ## mask protos are 1/4th of original size

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), original_shape[0], original_shape[1]))
        blur_size = (int(original_shape[1] / mask_width), int(original_shape[1]/ mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int((scale_boxes[i][0]))
            scale_y1 = int((scale_boxes[i][1]))
            scale_x2 = int((scale_boxes[i][2]))
            scale_y2 = int((scale_boxes[i][3]))

            x1 = int((scaled_up_boxes[i][0]))
            y1 = int((scaled_up_boxes[i][1]))
            x2 = int((scaled_up_boxes[i][2]))
            y2 = int((scaled_up_boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps


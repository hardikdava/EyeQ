import cv2
import numpy as np
import onnxruntime as ort
from eyeq.utils.non_maximum_suppression import non_max_suppression
import supervision as sv
from eyeq.utils.onnx_utils import OnnxDetectors
from eyeq.utils.box_utils import sigmoid


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
        original_shape = full_image.shape

        net_image = net_image.transpose((2, 0, 1))

        net_image /= 255
        net_image = np.expand_dims(net_image, 0)
        output = self.model.run(self.output_names, {self.model.get_inputs()[0].name: net_image})

        predictions = np.asarray(output[0])
        protos = np.asarray(output[1])

        pred = non_max_suppression(predictions, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, agnostic=agnostic, nm=32)

        pred = np.array(pred[0])

        boxes = pred[:, 0:4]
        class_ids = pred[:, 5].astype(int)
        confidence = pred[:, 4]
        mask_pred = pred[:, 6:]
        scaled_boxes = boxes.copy()
        boxes = self._process_predictions(boxes, full_image, pad)
        mask_maps = self.process_mask_output(boxes, scaled_boxes, mask_pred, protos, original_shape)

        detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidence, mask=mask_maps)
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


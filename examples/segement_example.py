import time

import cv2

from eyeq.inference_engine import InferenceEngine
from eyeq.utils.painter import draw_masks

model_config = dict()
model_config["yolov5_onnx"] = True
model_config["yolov7_onnx"] = False
model_config["yolov8_onnx"] = False


image_path = "../data/images/zidane.jpg"
img = cv2.imread(image_path)

detector = None
model_path = None

for key, item in model_config.items():
    if model_config[key] == True:
        if key == "yolov5_onnx":
            from eyeq.segmentors.yolov5.yolov5_onnx import V5ONNX
            model_path = "../data/weights/yolov5s-seg.onnx"
            detector = V5ONNX(conf_thresh=0.3, iou_thresh=0.4)
        break


inference_engine = InferenceEngine()
detector_id = detector.model_id
inference_engine.register_model(model=detector)


print("Active Models:", inference_engine.get_active_models())

inference_engine.load_network(model_id=detector_id, model_path=model_path)
counter = 0
# img = cv2.resize(img, (416, 416))
detections = inference_engine.forward(model_id=detector.model_id, img=img)

if detections:
    img = draw_masks(detections=detections, image=img)

cv2.imshow("image", img)
cv2.waitKey(0)

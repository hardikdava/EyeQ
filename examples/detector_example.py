import time

import cv2

from eyeq.inference_engine import InferenceEngine
from eyeq.utils.painter import draw_boxes

model_config = dict()
model_config["yolox_onnx"] = True
model_config["yolov5_onnx"] = False
model_config["yolov6_onnx"] = False
model_config["yolov7_onnx"] = False
model_config["yolov8_onnx"] = False
model_config["damoyolo_tinynasL18_Ns"] = False

image_path = "../data/images/zidane.jpg"
img = cv2.imread(image_path)

detector = None
model_path = None

for key, item in model_config.items():
    if model_config[key] == True:
        if key == "yolov5_onnx":
            from eyeq.detectors.yolov5.yolov5_onnx import V5ONNX
            model_path = "../data/weights/yolov5s.onnx"
            detector = V5ONNX(conf_thresh=0.2, iou_thresh=0.4)
        elif key == "yolov7_onnx":
            from eyeq.detectors.yolov7.yolov7_onnx import V7ONNX
            model_path = "../data/weights/yolov7-tiny.onnx"
            detector = V7ONNX(conf_thresh=0.2, iou_thresh=0.4)
        elif key == "yolov6_onnx":
            from eyeq.detectors.yolov6.yolov6_onnx import V6ONNX
            model_path = "../data/weights/yolov6s_base_bs1.onnx"
            detector = V6ONNX(conf_thresh=0.35, iou_thresh=0.4)
        elif key == "yolov8_onnx":
            from eyeq.detectors.yolov8.yolov8_onnx import V8ONNX
            model_path = "../data/weights/yolov8s.onnx"
            detector = V8ONNX(conf_thresh=0.35, iou_thresh=0.4)
        elif key == "damoyolo_tinynasL18_Ns":
            model_path = "../data/weights/damoyolo_tinynasL18_Ns.onnx"
            from eyeq.detectors.damoyolo.damoyolo_onnx import DamoOnnx
            detector = DamoOnnx(conf_thresh=0.5, iou_thresh=0.4)
        elif key == "yolox_onnx":
            model_path = "../data/weights/yolox_tiny.onnx"
            from eyeq.detectors.yolox.yolox_onnx import VXONNX
            detector = VXONNX(conf_thresh=0.3, iou_thresh=0.4)
        break


inference_engine = InferenceEngine()
detector_id = detector.model_id
inference_engine.register_model(model=detector)


print("Active Models:", inference_engine.get_active_models())

inference_engine.load_network(model_id=detector_id, model_path=model_path)
counter = 0

detections = inference_engine.forward(model_id=detector.model_id, img=img)

if detections:
    img = draw_boxes(detections=detections, image=img)
cv2.imshow("image", img)
cv2.waitKey(0)

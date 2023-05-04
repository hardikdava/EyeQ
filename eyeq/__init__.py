__version__ = "0.0.1"

# Import all detectors here
from eyeq.detectors.yolox.yolox_onnx import YoloXonnxDet
from eyeq.detectors.yolov5.yolov5_onnx import Yolov5onnxDet
from eyeq.detectors.yolov5u.yolov5u_onnx import Yolov5uonnxDet
from eyeq.detectors.yolov6.yolov6_onnx import Yolov6onnxDet
from eyeq.detectors.yolov7.yolov7_onnx import Yolov7onnxDet
from eyeq.detectors.yolov8.yolov8_onnx import Yolov8onnxDet
from eyeq.detectors.damoyolo.damoyolo_onnx import DamoYoloonnxDet

# import all segmentors here
from eyeq.segmentors.yolov5.yolov5_onnx import Yolov5onnxSeg
from eyeq.segmentors.yolov8.yolov8_onnx import Yolov8onnxSeg


# import dataset reader her
from eyeq.dataset.yolo_parser import YoloLoader

# import evaluatos reader her
from eyeq.evaluators.object_detection.detection_evaluation import DetectionEvaluator
# EyeQ


EyeQ is a minimal computer vision inference pakackge. Currently it supports following detectors in onnx runtime. It works with minimal dependencies. It is designed in such a manner to run on edge devices also.

--------------------
### Object Detection:
| Detector | onnx |
|--|--|
| [yolov5](https://github.com/ultralytics/yolov5) | ✅| 
| [yolov6](https://github.com/meituan/YOLOv6) | ✅ |
| [yolov7](https://github.com/WongKinYiu/yolov7) | ✅ | 
| [yolov8](https://github.com/ultralytics/ultralytics) | ✅ | 
| [yolov5u](https://github.com/ultralytics/ultralytics) | ✅ |
| [yoloX](https://github.com/Megvii-BaseDetection/YOLOX) | ✅ |
| [Damo-yolo](https://github.com/tinyvision/DAMO-YOLO) | ✅ |

--------------------
### Instance Segmentation:

| Detector Name | onnx |
|--|--|
| [yolov5](https://github.com/ultralytics/yolov5) | ✅ |
| [yolov7](https://github.com/WongKinYiu/yolov7) | #TODO |  
| [yolov8](https://github.com/ultralytics/ultralytics) | ✅ |  


--------------------
### Multi Object Tracker:

| Tracker Name | Integration |
|--|--|
| [SORT](https://github.com/ultralytics/yolov5) | ✅ |
| [ByteTrack](https://github.com/WongKinYiu/yolov7) | ✅ |
| [OcSort](https://github.com/ultralytics/ultralytics) | ✅ |
| [Norfair](https://github.com/ultralytics/ultralytics) | - |

--------------------
### Installation:

Installation can be done via pip using following argument
```
 pip3 install git+https://github.com/hardikdava/EyeQ.git
```
--------------------
#### TODO:
- Docker support
- RestAPI server ✅
- Multi object trackers ✅
- Instance segmentation ✅
- Yolo Dataset loading ✅
- COCO dataset loading
- Object detection evaluation ✅
- Multi object tracker evaluation ✅
- Automatic annotation support using clip, grounding dino and sam
- Introduce SAHI technique

#### Available APIs:

- Object Detection Inference using ONNX runtime
- Object Detction Evaluation API
- Model serving using RESTAPI using FastAPI based server
- Multi object Tracking for bounding boxes
- Multi object Tracking
- Instance segmentation support
- Data loading for yolo

Note: models are trained using [notebooks](https://github.com/roboflow/notebooks) prepared by roboflow but models are not included with codebase.

### References:




# EyeQ


EyeQ is a minimal computer vision inference pakackge. Currently it supports following detectors in onnx runtime. It works with minimal dependencies. It is designed in such a manner to run on edge devices also.

--------------------
### Object Detection:
| Detector | onnx |
|--|--|
| [yolov5](https://github.com/ultralytics/yolov5) | `:heavy_check_mark:`| 
| [yolov6](https://github.com/meituan/YOLOv6) | `:heavy_check_mark:` |
| [yolov7](https://github.com/WongKinYiu/yolov7) | `:heavy_check_mark:` | 
| [yolov8](https://github.com/ultralytics/ultralytics) | `:heavy_check_mark:` | 
| [yolov5u](https://github.com/ultralytics/ultralytics) | `:heavy_check_mark:` |
| [yoloX](https://github.com/Megvii-BaseDetection/YOLOX) | `:heavy_check_mark:` |
| [Damo-yolo](https://github.com/tinyvision/DAMO-YOLO) | `:heavy_check_mark:` |

--------------------
### Instance Segmentation:

| Detector Name | onnx |
|--|--|
| [yolov5](https://github.com/ultralytics/yolov5) | `:heavy_check_mark:` |
| [yolov7](https://github.com/WongKinYiu/yolov7) | #TODO |  
| [yolov8](https://github.com/ultralytics/ultralytics) | `:heavy_check_mark:` |  


--------------------
### Multi Object Tracker:

| Tracker Name | Integration |
|--|--|
| [SORT](https://github.com/ultralytics/yolov5) | - |
| [ByteTrack](https://github.com/WongKinYiu/yolov7) | - |
| [OcSort](https://github.com/ultralytics/ultralytics) | - |
| [Norfair](https://github.com/ultralytics/ultralytics) | - |

--------------------
### Installation:
```
 pip3 install git+https://github.com/hardikdava/EyeQ.git
```
--------------------
#### TODO:
- [ ] Docker support
- [X] RestAPI server
- [X] Multi object trackers
- [x] Instance segmentation
- [x] Yolo Dataset loading
- [ ] COCO dataset loading
- [ ] Fiftyone integration
- [x] Object detection evaluation
- [ ] Multi object tracker evaluation
- [ ] Automatic annotation support using clip, grounding dino and sam
- [ ] Introduce SAHI technique

#### Available APIs:

- Object Detection Inference using ONNX runtime
- Object Detction Evaluation API
- Model serving using RESTAPI using FastAPI based server
- Multi object Tracking for bounding boxes
- Instance segmentation support
- Data loading for yolo

### References:



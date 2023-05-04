import cv2
from tqdm import tqdm
from eyeq import YoloLoader, DetectionEvaluator, Yolov5onnxDet
from eyeq.utils.namespaces import TaskTypes

model_path = "../data/weights/yolov5s.onnx"
detector = Yolov5onnxDet(conf_thresh=0.2, iou_thresh=0.6)
detector.load_network(model_path=model_path)

task_type = TaskTypes.DETECTION
yaml_file_path = "../data/coco128.yaml"
data_reader = YoloLoader(yaml_file_path=yaml_file_path, task_type=task_type)

class_maps = data_reader.get_class_info()

image_paths = data_reader.val_images
label_paths = data_reader.val_labels

evaluator = DetectionEvaluator()
evaluator.set_classes(class_maps)


for f in tqdm(range(len(image_paths)), desc="Processing Inference"):
    image_path = image_paths[f]
    label_path = label_paths[f]

    img, groundtruth = data_reader.read_image_and_detections(image_path, label_path)
    evaluator.add_groundtruth(gt=groundtruth)
    detection = detector.infer(img)
    evaluator.add_prediction(det=detection)

seen, nt, mp, mr, map50, map = evaluator.evaluate()


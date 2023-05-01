from eyeq.utils.video_capture_extended import VideoCaptureExtended
from eyeq.detectors.damoyolo.damoyolo_onnx import DamoOnnx
from eyeq.utils.painter import draw_boxes
from eyeq.inference_engine import InferenceEngine
from eyeq.utils.fps_monitor import FpsMonitor
import cv2
import time


video_path = "../data/images/2.mp4"
model_path = "../data/weights/damoyolo_tinynasL18_Ns.onnx"


cap = VideoCaptureExtended(video_path)


inference_engine = InferenceEngine()
detector = DamoOnnx(conf_thresh=0.3, iou_thresh=0.5)

inference_engine.register_model(model=detector)
inference_engine.load_network(model_id=detector.model_id, model_path=model_path)

fps_monitor = FpsMonitor()
time.sleep(2)
counter = 0
while True:
    ret, img = cap.read()

    if not ret:
        break

    detections = inference_engine.forward(model_id=detector.model_id, img=img)
    fps_monitor.tick()
    if detections:
        img = draw_boxes(detections=detections, image=img)
    print(fps_monitor.get_fps())
    cv2.imshow("image", img)
    cv2.waitKey(1)

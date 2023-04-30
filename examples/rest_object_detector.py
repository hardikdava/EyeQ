import requests
import cv2
import base64
import json
import os
model_path = "yolov5s.onnx"
model_type = "yolov5_detector_onnx"

url = f'http://127.0.0.1:8000/load_model?model_path={model_path}&model_type={model_type}'
resp = requests.post(url=url)
response = resp.json()
path_img = "../data/images/zidane.jpg"
model_id = response["model_id"]


url = 'http://127.0.0.1:8000/detect'
with open(path_img, 'rb') as image_string:
  byte_string = base64.b64encode(image_string.read()).decode('utf-8')

res = requests.post(url, json={'data': byte_string, 'model_id': model_id})

print(res.json())




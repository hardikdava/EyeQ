import os
import typing
import cv2
import numpy as np
import supervision as sv
from eyeq.utils.namespaces import TaskTypes
from eyeq.utils.general import yaml_load, get_image_list
from eyeq.utils.box_utils import denormalize, xywh2xyxy


class YoloLoader:
    """
    Read Dataset in the form of Yolo
    """

    def __init__(self, yaml_file_path: str, task_type: TaskTypes):
        self.yaml_file_path = yaml_file_path
        self.task_type = task_type
        self.class_dict = {}
        self.yaml_data = {}
        self.read_yaml_file()

        self.train_directories = self.yaml_data.get('train', [])
        self.train_images = self.get_images_list(self.train_directories)
        self.train_labels = self.img2label_paths(self.train_images)
        self.val_directories = self.yaml_data.get('val', [])
        self.val_images = self.get_images_list(self.val_directories)
        self.val_labels = self.img2label_paths(self.val_images)

    def read_yaml_file(self) -> None:
        """
        :return: None but read all dataset paths
        """
        self.yaml_data = yaml_load(self.yaml_file_path)
        self.class_dict = self.yaml_data.get("names", {})

    def get_class_info(self) -> typing.Dict:
        """
        :return: labelmap as dict
        """
        return self.class_dict

    @staticmethod
    def img2label_paths(img_paths, label_type="labels") -> list:
        """
        :return: change extentions to txt and replace directory images to labels
        """
        # Define label paths as a function of image paths
        # /images/, /labels/ substrings
        s_a, s_b = f'{os.sep}images{os.sep}', f'{os.sep}{label_type}{os.sep}'
        return [s_b.join(x.rsplit(s_a, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    @staticmethod
    def read_image_and_detections(image_path: str, file_path: str):
        """
        :return: read and return detections of given filepath
        """
        xyxy = []
        class_id = []
        confidence = []
        detections = sv.Detections.empty()
        img = None
        if os.path.exists(file_path):
            img = cv2.imread(image_path)
            img_h, img_w, _ = img.shape
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f:
                    data = line.strip().split(' ')
                    bbox = [float(x) for x in data[1:5]]
                    xyxy.append(bbox)
                    class_id.append(int(data[0]))
                    confidence.append(1)

            xyxy = np.asarray(xyxy)
            class_id = np.asarray(class_id)
            confidence = np.asarray(confidence)
            if xyxy.shape[0] > 0:
                xyxy = xywh2xyxy(xyxy)
                xyxy = denormalize(detections=xyxy, img_h=img_h, img_w=img_w)
            detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)
        return img, detections

    def get_images_list(self, directory_list) -> list:
        """
        :return: return list of image list of given dataset type
        """
        image_paths = []
        if isinstance(directory_list, str):
            image_paths = get_image_list(directory_path=directory_list)
        elif isinstance(directory_list, list):
            for directory in directory_list:
                image_paths.extend(get_image_list(directory_path=directory))
        return image_paths

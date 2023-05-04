from enum import Enum


class DatasetTypes(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class TaskTypes(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    POSE_ESTIMATION = "pose_estimation"


class ImageFormats(Enum):
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    BMP = ".bmp"
    DNG = ".dng"
    MPO = ".mpo"
    TIFF = ".tiff"
    WEBP = ".webp"
    PFM = ".pfm"


class FileTypes(Enum):
    JSON = ".json"
    YAML = ".yaml"
    XML = ".xml"
    IMG = "image"
    TXT = ".txt"

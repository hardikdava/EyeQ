import yaml
import glob
import json
import os
from typing import Dict, List
from utils.namespaces import FileTypes, ImageFormats


def yaml_load(file='data.yaml') -> Dict:
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def json_load(file) -> Dict:
    with open(file, encoding="utf8") as f:
        return json.load(f)


def get_file_list(directory_path, file_type: FileTypes) -> List:
    print(file_type, type(file_type), file_type == FileTypes.IMG, directory_path)
    if file_type in [FileTypes.YAML, FileTypes.JSON, FileTypes.XML, FileTypes.TXT]:
        return glob.glob(f"{directory_path}/*{file_type.value}")


def get_image_list(directory_path) -> List:
    file_list = list()
    for ext in ImageFormats:
        _file_list = glob.glob(f"{directory_path}/*{ext.value}")
        file_list.extend(_file_list)
    return file_list


def is_exists(filepath) -> bool:
    return os.path.exists(filepath)


def create_dir(dir_path) -> bool:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return True

import numpy as np
import supervision as sv


def draw_detections(image: np.ndarray, detections: sv.Detections, label_maps: dict=None) -> np.ndarray:
    box_annotator = sv.BoxAnnotator()
    if label_maps:
        labels = [
            f"{label_maps[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
    else:
        labels = [
            f"{class_id} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    return annotated_frame


# TODO: Improve drawing method
def draw_tracklets(image: np.ndarray, detections: sv.Detections, label_maps: dict) -> np.ndarray:
    box_annotator = sv.BoxAnnotator()
    if label_maps:
        labels = [
            f"{label_maps[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
    else:
        labels = [
            f"{class_id} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    return annotated_frame
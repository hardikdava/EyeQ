import supervision as sv
from supervision.geometry.core import Point, Position
from typing import Dict, List, Optional, Union
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
import numpy as np
import cv2


class Trace:

    xy: np.ndarray
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None

    def __init__(
        self,
        position: Position = Position.CENTER,
        max_length: int = 10,
    ):
        self.trace: Dict[int, List[sv.Point]] = {}
        self.position = position
        self.max_length = max_length
        self.frame_counter = 0
        # Counter, x, y, class, trackid
        self.storage = np.zeros((0, 5))

    def update(self, frame_counter: int, detections: Detections) -> None:
        if detections.xyxy.shape[0]>0:
            xyxy = detections.xyxy
            x = (xyxy[:, 0] + xyxy[:, 2]) / 2
            y = (xyxy[:, 1] + xyxy[:, 3]) / 2
            new_detections = np.zeros(shape=(xyxy.shape[0], 5))
            new_detections[:, 0] = frame_counter
            new_detections[:, 1] = x
            new_detections[:, 2] = y
            new_detections[:, 3] = detections.class_id
            new_detections[:, 4] = detections.tracker_id
            self.storage = np.append(self.storage, new_detections, axis=0)
            self.frame_counter = frame_counter
        self._remove_previous()

    def _remove_previous(self):
        to_remove_frames = self.frame_counter - self.max_length
        if self.storage.shape[0]>0:
            valid = np.where(self.storage[:, 0] > to_remove_frames)
            self.storage = self.storage[valid]


class TraceAnnotator:

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.boundry_tolerance = 20

    def annotate(self, scene: np.ndarray, trace: Trace) -> np.ndarray:
        img_h, img_w, _ = scene.shape
        unique_ids = np.unique(trace.storage[:, -1])
        for unique_id in unique_ids:
            valid = np.where(trace.storage[:, -1] == unique_id)[0]

            frames =trace.storage[valid, 0]
            latest_frame = np.argmax(frames)
            points_to_draw = trace.storage[valid, 1:3]

            n_pts = points_to_draw.shape[0]
            headx, heady = int(points_to_draw[latest_frame][0]), int(points_to_draw[latest_frame][1])

            if headx > self.boundry_tolerance and heady > self.boundry_tolerance:
                class_id = trace.storage[0, -2]
                idx = int(unique_id)
                color = (
                    self.color.by_idx(idx)
                    if isinstance(self.color, ColorPalette)
                    else self.color
                )

                for i in range(n_pts-1):
                    px, py = int(points_to_draw[i][0]), int(points_to_draw[i][1])
                    qx, qy = int(points_to_draw[i+1][0]), int(points_to_draw[i+1][1])
                    cv2.line(scene, (px, py), (qx, qy), color.as_bgr(), self.thickness)
                    scene = cv2.circle(scene, (headx, heady), int(10), color.as_bgr(), thickness=-1)

        return scene




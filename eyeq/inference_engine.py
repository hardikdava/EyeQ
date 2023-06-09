import numpy as np
import supervision as sv


class InferenceEngine:

    def __init__(self):
        self.active_models = dict()
        self.active_trackers = dict()

    def load_network(self, model_id: str, model_path: str):
        if model_id in self.active_models:
            self.active_models[model_id]["model"].load_network(model_path=model_path)
            print(f"Model {model_path} is loaded with id {model_id}")
        return True

    def set_labelmap(self, model_id: str, labelmap: dict):
        if model_id in self.active_models:
            self.active_models[model_id]["labelmap"] = labelmap

    def register_model(self, model):
        if model.model_id not in self.active_models:
            self.active_models[model.model_id] = {"model": model}

    def assign_model_name(self, model_id: str, model_name: str):
        if model_id in self.active_models:
            self.active_models[model_id]["name"] = model_name

    def deregister_model(self, model_id: str):
        if model_id in self.active_models:
            del self.active_models[model_id]

    def get_active_models(self):
        return self.active_models

    def forward(self, model_id: str, img: np.ndarray):
        detections = sv.Detections.empty()
        if model_id in self.active_models:
            detections = self.active_models[model_id]["model"].infer(img=img)
        return detections

    def register_tracker(self, tracker):
        if tracker.tracker_id not in self.active_trackers:
            self.active_trackers[tracker.model_id] = {"tracker": tracker}

    def assign_tracker_name(self, tracker_id: str, model_name: str):
        if tracker_id in self.active_trackers:
            self.active_trackers[tracker_id]["name"] = model_name

    def deregister_tracker(self, tracker_id: str):
        if tracker_id in self.active_trackers:
            del self.active_trackers[tracker_id]

    def get_active_trackers(self):
        return self.active_trackers

    def track(self, tracker_id: str, detections: sv.Detections):
        tracklets = sv.Detections.empty()
        if tracker_id in self.active_trackers:
            tracklets = self.active_trackers[tracker_id]["tracker"].update(detections=detections)
        return tracklets

    def forward_and_track(self, model_id: str, tracker_id: str, img: np.ndarray):
        detections = self.forward(model_id=model_id, img=img)
        tracklets = self.track(tracker_id=tracker_id, detections=detections)
        return tracklets


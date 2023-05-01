import sys
from datetime import datetime
import supervision as sv
import numpy as np
from scipy.optimize import linear_sum_assignment


from eyeq.trackers.nms_sort.kalman_utilities import iou_matrix, iom_matrix, convert_bboxes_to_z, convert_x_to_bboxes
from eyeq.utils.non_maximum_suppression import nms


class NmsSortTracker:
    def __init__(self, parameter: dict = None):
        self.frame_count = 0
        self.traj, self.objects = dict(), dict()  # The currently tracked objects and all used objects accumulated
        self.frame_timestamps = list()  # A list of timestamps corresponding to all incoming frames
        self.min_confidence_threshold = None
        self.next_id = 0  # The next id for new trajectories
        self.iou_matching_threshold = 0.15  # The minimal value that object proposals and trajectories can be merged
        self.iou_removal_threshold = 0.5  # The maximal iou that lost trajectories are allowed to have to others
        self.min_lifetime = 6  # The minimal number of frames a trajectory needs to exists to be valid
        self.max_spatial_cue_time = 20
        self.max_lost_time = 60  # The maximal number of frames to be lost
        self.image_width, self.image_height = None, None
        self.matched_obj_removal_threshold = 0.5
        self.unmatched_obj_removal_threshold = 0.5
        self.matched_traj_removal_threshold = 0.5
        self.unmatched_obj_nms_threshold = 0.4
        self.matching_method = "iom"
        self.tracker_fps = 0
        self.wait_time = 1
        self.labels = {"pig": 0}
        # Update values from parameter dict
        if parameter is not None:
            for key, value in parameter.items():
                try:
                    self.__getattribute__(key)
                    self.__setattr__(key, value)
                except Exception as e:
                    raise Exception("Cannot set tracking parameter %s. %s" % (key, e))

        self.min_confidence_threshold = self.min_confidence_threshold if self.min_confidence_threshold else 0.4
        # Static Kalman variables
        self.dim_x = 7
        self.dim_z = 4
        self.kalman_A = np.array(  # State transition function      7x7 matrix
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.kalman_H = np.array(  # Measurement function      4x7 matrix
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])
        self.kalman_Q = np.eye(self.dim_x)  # process uncertainty (copied from pyfilter)
        self.kalman_Q[-1, -1] *= 0.01
        self.kalman_Q[4:, 4:] *= 0.01
        self.kalman_R = np.eye(self.dim_z)  # state uncertainty (copied from pyfilter)
        self.kalman_R[2:, 2:] *= 10.
        self.kalman_P_initialization = np.eye(self.dim_x)
        self.kalman_P_initialization[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kalman_P_initialization *= 10.

        self.reset()  # Initializes the objects and traj dictionary

    def get_start_time(self):
        """ Returns the timestamp of the first image tracked in until the start/last reset """
        ret = self.frame_timestamps[0] if self.frame_timestamps else None
        return ret

    def get_last_time(self):
        """ Returns the timestamp of the last image tracked """
        if self.frame_timestamps:
            return self.frame_timestamps[-1]
        return None

    def update(self, detections: sv.Detections, store_all_objects=False):
        try:
            ''' 
            Some preprocessing and meta stuff:
                - count up frame count 
                - add timestamps to internal timestamp list
                - copy boxes array, so no reference call is allowed
                - count up lost state of all trajectories, because till now they are not tracked
            '''
            self.frame_count += 1
            timestamp = datetime.now()
            self.frame_timestamps.append(timestamp)

            xyxy = detections.xyxy
            class_ids = detections.class_id
            confidences = detections.confidence
            n_new_detections = xyxy.shape[0]
            boxes = np.zeros((n_new_detections, 6))

            if n_new_detections>0:
                boxes[:, :4] = xyxy
                boxes[:, 4] = confidences
                boxes[:, -1] = class_ids

            boxes = np.copy(boxes)
            self._count_up_lost_state()
            ''' 
            The main tracker algorithm:
                1. Estimate new object states
                2. Match object proposals to existing trajectories based on local spatial cues
                3. Suppress 
                    a. object proposals by trajectories
                    b. object proposals by local object proposals with higher confidence
                    c. potential incorrect trajectories by valid trajectories with high overlap 
                    d. object proposals by confidence threshold 
                4. Match remaining object proposals to existing trajectories based on long term social feature
                5. Create new trajectories for remaining object proposals
                6. Remove outdated trajectories
            '''
            self._predict_kalman()  # 1
            boxes = self._match_by_motion_model(boxes)  # 2
            boxes = self._suppress_by_existing_trajecories(boxes)  # 3a
            boxes = self._suppress_by_local_maxima(boxes)  # 3b
            self._suppress_trajectories_by_trajectories()  # 3c

            boxes = self._suppress_boxes_by_confidence_threshold(boxes, self.min_confidence_threshold)  # 3d
            self._init_new_trajectories(boxes)  # 5
            self._remove_outdated_trajectories()  # 6
            if store_all_objects:
                self._store_current_trajectories_to_object_pool()

            return self.get_current_objects(remove_lost=True, remove_short_trajectories=True)

        except Exception as e:
            print('Online Tracker: Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def get_current_objects(self, remove_lost=False, remove_short_trajectories=True) -> dict:
        """
        Returns the current objects list.
        :param remove_lost: Remove all objects with lost state > 0
        :param remove_short_trajectories: Remove all objects that belong to short trajectories < self.min_lifetime
        :return: A dict with arrays containing information about the object
        """
        new_dict = self._create_traj_dict()
        if self.traj["x1"].size > 0:
            valid = np.ones_like(self.traj["lifespan"]) == 1
            if remove_lost:
                valid *= self.traj["lifespan"] > self.min_lifetime
            if remove_short_trajectories:
                valid *= self.traj["lost"] == 0
            if np.sum(valid) > 0:
                for key, item in self.traj.items():
                    new_dict[key] = np.copy(item[valid])

        tracklets = sv.Detections.empty()
        if new_dict['x1'].shape[0]>0:
            xyxy = np.zeros((new_dict['x1'].shape[0], 4))
            xyxy[:, 0] = new_dict['x1']
            xyxy[:, 1] = new_dict['y1']
            xyxy[:, 2] = new_dict['x2']
            xyxy[:, 3] = new_dict['y2']
            tracklets = sv.Detections(xyxy=xyxy, class_id=new_dict['label'], confidence=new_dict['confidence'], tracker_id=new_dict['id'])

        return tracklets

    def update_trajectories(self, indices, indices_boxes, boxes) -> np.ndarray:
        """
        Updates existing trajectories with new bounding box information
        :param indices: The indices of trajectories in self.traj that should be updated
        :param indices_boxes: The indices of the boxes that should be used for the update
        :param boxes: The boxes used for the update
        :return: The input array without the assigned boxes
        """
        # Update Kalman Filter

        z = convert_bboxes_to_z(boxes[indices_boxes])
        # Calculate residuum/gain
        HPHTR = self.traj["P"][indices, 0:4, 0:4] + self.kalman_R[:, :]  # Simplification of H * P^- * H^T
        HPHTR_inv = np.zeros_like(HPHTR)
        HPHTR_inv[:, [0, 1, 2, 3], [0, 1, 2, 3]] = 1 / np.maximum(0.0001, HPHTR[:, [0, 1, 2, 3], [0, 1, 2, 3]])
        PHT = self.traj["P"][indices, :, 0:4]
        K = np.matmul(PHT, HPHTR_inv)
        # Update state with gain and measurement error
        y = (z - self.traj["X"][indices, :4])[:, :, None]
        update = np.matmul(K, y)[:, :, 0]
        self.traj["X"][indices] = self.traj["X"][indices] + update
        # Update covariance matrix
        I = np.zeros_like(self.traj["P"][indices])
        I[:, [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]] = 1
        I[:, :, 0:4] -= K
        self.traj["P"][indices] = np.matmul(I, self.traj["P"][indices])
        # Update meta data
        self.traj["lost"][indices] = 0
        self.traj["frame"][indices] = len(self.frame_timestamps)
        self.traj["time"][indices] = self.frame_timestamps[-1].timestamp()
        self.traj["confidence"][indices] = np.copy(boxes[indices_boxes, 4])
        self.traj["lifespan"][indices] += 1
        # Remove the assigned object proposals from the set of object proposals
        if indices.size == boxes.shape[0]:
            boxes = np.zeros((0, 6))
        else:
            inds = np.arange(0, boxes.shape[0], dtype=int)
            boxes = boxes[inds[np.isin(inds, indices_boxes, invert=True)]]
        return boxes

    def _predict_kalman(self):
        """ Predicts the new state of trajectories with a kalman filter """
        if self.traj["x1"].size > 0:
            self.traj["X"][:, :3] = self.traj["X"][:, 0:3] + self.traj["X"][:, 4:7]
            self.traj["P"][:, 0:3, :] += self.traj["P"][:, 4:7, :]  # Simple implementation of A*P
            self.traj["P"][:, :, 0:3] += self.traj["P"][:, :, 4:7]  # Simple implementation of P*A^T
            self.traj["P"] += self.kalman_Q[:, :]  # Simple implementation of P+Q
            pr = convert_x_to_bboxes(self.traj["X"])
            pr[np.isnan(pr)] = 0
            self.traj["x1"], self.traj["y1"], self.traj["x2"], self.traj["y2"] = pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3]

    def _match_by_motion_model(self, boxes: np.ndarray) -> np.ndarray:
        if self.traj["x1"].size == 0 or boxes.size == 0:
            return boxes

        # Calculate ious between trajectories (i) and incoming boject proposals (j)
        boxes_i = np.stack([self.traj["x1"], self.traj["y1"], self.traj["x2"], self.traj["y2"]], axis=1)
        boxes_j = boxes[:, 0:4]
        conf_scores = boxes[:, 4]
        ious = iou_matrix(boxes_i, boxes_j)
        ious = ious * conf_scores * conf_scores  ## Eq-11
        # Calculate costs
        with np.errstate(divide='ignore'):
            costs = -np.log(ious / (1 - ious + 1e-10))  ## Eq-12
        costs[ious < self.iou_matching_threshold] = 10000000
        is_lost_too_long = self.traj["lost"][:, None] > self.max_spatial_cue_time
        label_mismatch = (self.traj["label"][:, None] - boxes[:, 5][None, :]) != 0
        costs += 1000000 * label_mismatch + 1000000 * is_lost_too_long
        # Find best matching pairs
        costs[np.isnan(costs)] = 10000000
        row, col = linear_sum_assignment(costs)
        # Check if pairs are valid and remove invalid pairs
        valid = ious[row, col] > self.iou_matching_threshold

        row, col = row[valid], col[valid]
        if col.size > 0:
            matched_obj_proposals = boxes_j[col]

            ious_matched_objs = iou_matrix(matched_obj_proposals, matched_obj_proposals)
            _indices = np.argwhere(ious_matched_objs > self.matched_obj_removal_threshold)
            to_del_row, to_del_col = [], []
            if _indices.size > 0:
                for indice in _indices:
                    if (indice[0] != indice[1]) and (indice[0] not in to_del_col) and (indice[1] not in to_del_col):
                        # Find cost at (row_id,indice[0]) and (row_id,indice[1])
                        if (costs[row[indice[0]], col[indice[0]]] < costs[row[indice[1]], col[indice[1]]]):
                            to_del_row.append(row[indice[1]])
                            to_del_col.append(col[indice[1]])
                        else:
                            to_del_row.append(row[indice[0]])
                            to_del_col.append(col[indice[0]])
                row = np.setdiff1d(row, to_del_row, assume_unique=True)
                col = np.setdiff1d(col, to_del_col, assume_unique=True)

        if row.size > 0:
            boxes = self.update_trajectories(row, col, boxes)

        return boxes

    def _suppress_by_existing_trajecories(self, boxes: np.ndarray) -> np.ndarray:
        if self.traj["x1"].size == 0 or boxes.size == 0:
            return boxes
        boxes_i = np.stack([self.traj["x1"], self.traj["y1"], self.traj["x2"],
                            self.traj["y2"]], axis=1)
        boxes_j = boxes[:, 0:4]
        if self.matching_method == "iou":
            ious = iou_matrix(boxes_i, boxes_j)
        elif self.matching_method == "iom":
            ious = iom_matrix(boxes_i, boxes_j)
        else:
            raise Exception()
        _indices = np.argwhere(ious > self.unmatched_obj_removal_threshold)
        labels_i = self.traj["label"]
        labels_j = boxes[:, 5]
        indices_overlap = []
        for indice in _indices:
            if labels_i[indice[0]] != labels_j[indice[1]]:
                continue
            ind_col = indice[1]
            if ind_col not in indices_overlap:
                indices_overlap.append(ind_col)
        indices_overlap = np.asarray(indices_overlap)
        if indices_overlap.size > 0:
            inds = np.arange(0, boxes.shape[0], dtype=int)
            main_list = np.setdiff1d(inds, indices_overlap)
            boxes = boxes[main_list]
        return boxes

    def _suppress_by_local_maxima(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.shape[0] == 0:
            return boxes
        valid_ids = []
        for key, value in self.labels.items():
            _boxes = boxes[np.where(boxes[:, 5] == int(value))]
            valid_ids = nms(_boxes[:, 0:4], _boxes[:, 4], self.unmatched_obj_nms_threshold)
        if len(valid_ids)>0:
            boxes = boxes[valid_ids]
        return boxes

    def _suppress_trajectories_by_trajectories(self):
        if self.traj["x1"].size == 0:
            return
            # Suppress Overlapping Trajectories to remove FP traj
            # Following lines make sure that bounding boxes are within range

        boxes_i = np.stack([self.traj["x1"], self.traj["y1"], self.traj["x2"], self.traj["y2"]], axis=1)
        to_del = []
        ious_traj = iou_matrix(boxes_i, boxes_i)
        _indices = np.argwhere(ious_traj > self.matched_traj_removal_threshold)

        for indice in _indices:
            ind_row, ind_col = indice[0], indice[1]
            if ind_row != ind_col:
                if (ind_row not in to_del) and (ind_col not in to_del):
                    row_label, col_label = self.traj["label"][ind_row], self.traj["label"][ind_col]
                    ind_row_status, ind_col_status = self.traj["lost"][ind_row], self.traj["lost"][ind_col]
                    ind_row_lifespan, ind_col_lifespan = self.traj["lifespan"][ind_row], self.traj["lifespan"][ind_col]
                    ind_row_conf, ind_col_conf = self.traj["confidence"][ind_row], self.traj["confidence"][ind_col]

                    if row_label != col_label:
                        continue
                    if ind_row_status > ind_col_status:
                        to_del.append(ind_row)
                    elif ind_row_status < ind_col_status:
                        to_del.append(ind_col)
                    elif ind_row_lifespan > ind_col_lifespan:
                        to_del.append(ind_row)
                    elif ind_row_lifespan < ind_col_lifespan:
                        to_del.append(ind_col)
                    elif ind_row_conf > ind_col_conf:
                        to_del.append(ind_row)
                    elif ind_row_conf < ind_col_conf:
                        to_del.append(ind_col)
                    continue

        to_del = np.asarray(to_del)
        if to_del.size > 0:
            inds = np.arange(0, self.traj["lost"].shape[0], dtype=int)
            valid_traj = np.setdiff1d(inds, to_del)
            # Remove from traj
            if valid_traj.size > 0:
                for key in self.traj.keys():
                    self.traj[key] = self.traj[key][valid_traj]
            else:
                self.traj = self._create_traj_dict()



    def _init_new_trajectories(self, boxes: np.ndarray):
        if boxes.size == 0:
            return
        new = boxes.shape[0]
        self.traj["dx"] = np.append(self.traj["dx"], np.zeros(new))
        self.traj["dy"] = np.append(self.traj["dy"], np.zeros(new))
        self.traj["lost"] = np.append(self.traj["lost"], np.zeros(new))
        self.traj["frame"] = np.append(self.traj["frame"], np.ones(new) * len(self.frame_timestamps))
        self.traj["time"] = np.append(self.traj["time"], np.ones(new) * self.frame_timestamps[-1].timestamp())
        self.traj["x1"] = np.append(self.traj["x1"], np.copy(boxes[:, 0]))
        self.traj["y1"] = np.append(self.traj["y1"], np.copy(boxes[:, 1]))
        self.traj["x2"] = np.append(self.traj["x2"], np.copy(boxes[:, 2]))
        self.traj["y2"] = np.append(self.traj["y2"], np.copy(boxes[:, 3]))
        self.traj["confidence"] = np.append(self.traj["confidence"], np.copy(boxes[:, 4]))
        self.traj["label"] = np.append(self.traj["label"], np.copy(boxes[:, 5]))
        self.traj["lifespan"] = np.append(self.traj["lifespan"], np.ones(new))
        self.traj["id"] = np.append(self.traj["id"], np.arange(self.next_id, self.next_id + new))

        # Update Kalman filter
        self.traj["X"] = np.append(self.traj["X"], convert_bboxes_to_z(boxes, initialized=True), axis=0)
        self.traj["P"] = np.append(self.traj["P"], np.tile(self.kalman_P_initialization, (new, 1, 1)), axis=0)
        self.next_id += new
        bb = convert_x_to_bboxes(self.traj["X"])
        self.traj["x1"], self.traj["y1"], self.traj["x2"], self.traj["y2"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]

    def _remove_outdated_trajectories(self):
        # Remove based on lost state
        if self.traj["x1"].size:
            # Check if trajectory is lost for a too long time or longer lost than existing
            is_not_lost = self.traj["lost"] <= np.minimum(self.max_lost_time, self.traj["lifespan"])
            valid_boxes = np.where(is_not_lost)[0]

            if valid_boxes.size > 0:
                for key, item in self.traj.items():
                    self.traj[key] = item[valid_boxes]
            else:
                self.traj = self._create_traj_dict()

    def _store_current_trajectories_to_object_pool(self):
        if self.traj["x1"].size > 0:
            valid = np.where(self.traj["lost"] == 0)[0]
            if valid.size > 0:
                for key in self.objects.keys():
                    self.objects[key] = np.append(self.objects[key], self.traj[key][valid])

    def get_all_objects(self) -> dict:
        """
        Returns the all objects the tracker has seen during the tracking session.
        :return: a dict with single axis arrays containing information at least for keys:
            x1, y1, x2, y2, confidence, id, time, frame, label
        """
        new_dict = dict()
        for key, item in self.objects.items():
            new_dict[key] = np.copy(item)
        return new_dict

    def _create_traj_dict(self):
        """ Creates an empty current trajectories dictionary """
        return dict(
            x1=np.zeros(0), y1=np.zeros(0), x2=np.zeros(0), y2=np.zeros(0), dx=np.zeros(0), dy=np.zeros(0),  # Position
            confidence=np.zeros(0), lost=np.zeros(0), label=np.zeros(0), lifespan=np.zeros(0),  # Tracking data
            id=np.zeros(0), frame=np.zeros(0), time=np.zeros(0),  # Time data
            X=np.zeros((0, self.dim_x)), P=np.zeros((0, self.dim_x, self.dim_x)),  # Kalman filter properties
        )

    @staticmethod
    def _create_object_dict():
        """ Creates an empty object pool dictionary """
        return dict(
            x1=np.zeros(0), y1=np.zeros(0), x2=np.zeros(0), y2=np.zeros(0),  # Position
            confidence=np.zeros(0), id=np.zeros(0), label=np.zeros(0),  # Tracking data
            frame=np.zeros(0), time=np.zeros(0),  # Time data
        )

    def _count_up_lost_state(self):
        """ Counts up the lost state of all currently tracked detections """
        if self.traj["lost"].size > 0:
            self.traj["lost"] += 1

    @staticmethod
    def _suppress_boxes_by_confidence_threshold(boxes: np.ndarray, threshold: float) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        return boxes[boxes[:, 4] >= threshold]

    @staticmethod
    def _suppress_boxes_by_labels(boxes: np.ndarray, label: int) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        return boxes[boxes[:, 5] == label]

    def reset(self):
        """ Resets the tracker to an initial state """
        self.objects = self._create_object_dict()
        self.traj = self._create_traj_dict()
        self.frame_timestamps = list()


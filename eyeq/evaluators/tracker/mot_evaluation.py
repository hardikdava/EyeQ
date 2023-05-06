import motmetrics as mm
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class MOTBenchmark:
    def __init__(self, gt_file: str=None, det_file: str=None):
        # Creates an accumulator that will be updated during each frame
        self.acc = mm.MOTAccumulator(auto_id=True)
        # Runs benchmark
        self.gnd_detections = None
        self.detections = None
        self.gt_file = gt_file
        self.det_file = det_file

    def groundtruth_file(self, gt_file):
        self.gt_file = gt_file

    def tracker_file(self, det_file):
        self.det_file = det_file

    def load_data(self):
        self.gnd_detections = self.read_mot_file(self.gt_file)
        self.detections = self.read_mot_file(self.det_file)

    def run_benchmark(self):

        total_frames = int(np.max(self.gnd_detections[:, 0]))

        for frame_counter in range(total_frames):
            frame_counter += 1 # detection and frame numbers begin at 1

            frame_gt = self.gnd_detections[self.gnd_detections[:, 0] == frame_counter, 1:6]  # select all detections in gt
            frame_dets = self.detections[self.detections[:, 0] == frame_counter, 1:6]  # select all detections in t

            C = mm.distances.iou_matrix(frame_gt[:, 1:], frame_dets[:, 1:], \
                                        max_iou=0.5)  # format: gt, t

            # Call update once for per frame.
            # format: gt object ids, t object ids, distance
            self.acc.update(frame_gt[:, 0].astype('int').tolist(), \
                       frame_dets[:, 0].astype('int').tolist(), C)

        # Computes metrics
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        return summary

    @staticmethod
    def read_mot_file(source):
        data = np.loadtxt(source, delimiter=',')
        return data





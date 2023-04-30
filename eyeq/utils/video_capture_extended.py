import cv2


class VideoCaptureExtended:
    """
    This class implements an extended VideoCapture that is able to step backward
    """
    def __init__(self, video_file):
        """
        Initiates the VideoCapture. Handles the videofile with like cv2.VideoCapture
        :param buffer_length: ONLY FOR COMPATIBILITY!
        :param video_file: The videofile path
        """
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)
        self.current_frame_count = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        """
        Reads the next frame. Like cv2.VideoCapture.read()
        :return: Return value, The numpy frame mat
        """
        ret, frame = self.cap.read()
        if ret is True:
            self.current_frame_count += 1
        return ret, frame

    def frame_backward(self):
        """
        Steps a frame back. Returns False and the same frame, if there is no frame backward. NOTE: If bufferoverflow it
        needs some time
        :return: Return value, The numpy frame mat
        """
        if self.current_frame_count <= 1:
            # Already first frame
            self.current_frame_count = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = self.cap.read()
            return False, img

        else:
            self.current_frame_count -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_count - 1)
            ret, img = self.cap.read()
            return ret, img

    def isOpened(self):
        """
        Like cv2.VideoCapture.isOpened()
        :return:
        """
        return self.cap.isOpened()

    def read_frame_at_position(self, frame):
        """
        Reads the specific frame (First frame has index 1!)
        :param frame:
        :return:
        """
        frame_difference = frame - self.current_frame_count
        if 5 >= frame_difference >= 1:
            for _ in range(frame_difference):
                ret, img = self.read()
            return ret, img

        if frame <= 0:
            return False, None
        elif frame > self.total_frames:
            return False, None
        else:
            self.current_frame_count = frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ret, img = self.cap.read()
            return ret, img

    def get_current_frame_count(self):
        """
        Returns the current video frame count
        :return: Current frame count
        """
        return self.current_frame_count

    def release(self):
        self.cap.release()

    def __len__(self):
        return self.total_frames
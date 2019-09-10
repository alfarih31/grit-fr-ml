"""
high level support for doing this and that.
"""
from time import time
import torch.multiprocessing as mp
from threading import Thread
from pykalman import KalmanFilter as pyKF
import cv2
import numpy as np

from vision.utils import Timer
from utils import Flags

timer = Timer()
class KalmanFilter:
    def __init__(self, eps=2):
        #self.kf = cv2.KalmanFilter(6, 4)
        self.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1],], np.float32)
        self.transitionMatrix = np.array(1*np.eye(4), np.float32)
        self.kf = pyKF(observation_matrices=self.measurementMatrix,
                       transition_matrices=self.transitionMatrix)
        self.last_time = time()
        self.init = False
        self.means = [0, 0, 0, 0]
        self.cov = [2, 2, 2, 2]
        self.eps = eps

    def init_check(self, est, current_box):
        self.init = True
        for i, item in enumerate(est):
            if abs(item-current_box[i]) < self.eps:
                self.init = self.init is True
            else:
                self.init = False

    def estimate(self, box):
        measure = np.array([np.float32(box[0]),
                            np.float32(box[1]),
                            np.float32(box[2]),
                            np.float32(box[3])])
        self.means, self.cov = self.kf.filter_update(self.means, self.cov, measure)
        if not self.init:
            self.init_check(self.means, box)
        return self.means, self.init

    def reset(self):
        self.means = [0, 0, 0, 0]
        self.cov = [2, 2, 2, 2]
        self.init = False

class Matcher:
    """
        Matcher will get matched frame with initial frame and marked by a bbox.
    This object must be initialized by the initial frame, i.e. the template.
        The marked bbox will be used for tracking the same object then by comparing
    the intersect area of all box in detected bboxes.
        The __call__  will return the marked bbox which has been shrinked by 60%. It is needed
    to get shrinked because sometimes there's another nearby detector's bboxes which intersect each other.
    The shrinked marked bbox will be the most intersected with the supposed bbox (the target).
            @return: (xmin, ymin, xmax, ymax)
    """
    def __init__(self, template):
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.pick = lambda image: cv2.minMaxLoc(
            cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                              self.template, cv2.TM_CCOEFF_NORMED))

    def __call__(self, img):
        h, w = self.template.shape
        #h, w = h*0.6, w*0.6
        _, confidence, _, max_loc = self.pick(img)
        return (max_loc[0], max_loc[1], max_loc[0]+w, max_loc[1]+h), confidence

    def update(self, img, box):
        self.template = cv2.cvtColor(img[int(box[1]):int(box[3]),
                                         int(box[0]):int(box[2])],
                                     cv2.COLOR_BGR2GRAY)
        cv2.imshow('test', self.template)

class Tracker:
    def __init__(self, kw: float, kh: float, predictor, flags: Flags):
        self.predictor = predictor
        self.flags = flags
        self.kw = 0.5 * kw
        self.kh = 0.5 * kh
        self.matcher = None
        self.target_box = None
        self.distances = None
        self.boxes = None
        self.data = mp.Manager().dict()
        self.image_queue = mp.Queue()

        # For estimate shift
        self.should_kalman = False
        self.kalman_ready = False
        self.detect_process = None
        self.kf = KalmanFilter()

        # Image Property
        self.image_shape = None

        # Moving Average
        self.box_cache = [[0, 0, 0, 0]]
        self.tick = 0
        self.period = 1
    def __call__(self, img):
        """Return: (Distance by h, Distance by w, image)"""
        if not self.flags.tracker_init:
            try:
                self.init_tracker(img)
            except ValueError:
                return img
        return self.track(img)

    def init_tracker(self, img):
        """Used to initializing the Matcher object with the most nearest object"""
        # Get box with the most largetst area, i.e. the most nearest object
        self.image_shape = np.flip(img.shape[:-1])-1
        self.detect_process = Thread(target=self.detect, args=(img, self.data,), daemon=True)
        self.detect_process.start()
        self.detect_process.join() # Wait for detection process is ended
        if not self.data[0]:
            raise ValueError("%s: There's no detected object"%self.detect.__name__)
        else:
            self.boxes = self.data[1]

        minArea = 0
        for i in range(self.boxes.size(0)):
            box = self.boxes[i, :]
            h, w = box[:-2]
            area = h * w
            if area > minArea:
                box = self.filter_box(box)
                d_w = (box[2]-box[1])*0.08
                box[0] = box[0]+d_w
                box[2] = box[2]-d_w
                minArea = area
                print(box)
        box = self.filter_box(box)
        self.matcher = Matcher(img[int(box[1]):int(box[3]),
                                   int(box[0]):int(box[2])])

        self.target_box = box.numpy()
        self.kf.estimate(self.target_box)

        self.flags.tracker_init = True

    def track(self, img):
        """
        Used to keep an eye to target by picking the bbox with most largest
        intersect area with marked bbox (target's bbox).
            Return: (Distance by h, Distance by w, image)
        """

        # Get target's bbox by matcher object
        track_loc, confidence = self.matcher(img)
        if not self.detect_process.is_alive():
            self.detect_process.join()
            self.detect_process = Thread(target=self.detect, args=(img, self.data,), daemon=True)
            self.detect_process.start()
            if not self.data[0]:
                print("%s: There's no detected object"%self.detect.__name__)
                return img
            else:
                self.boxes = self.data[1]
                # Filter box in detected boxes which has highest intersect area
                minIOU = 0
                for i in range(self.boxes.size(0)):
                    box = self.boxes[i, :]
                    iou = self.compute_iou(track_loc, box)
                    cv2.rectangle(img,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 0, 255), 2)
                    if iou > minIOU:
                        self.target_box = box
                        minIOU = iou
                if self.target_box is None:
                    print("%s: There's no intersect bounding box"%self.track.__name__)
                    return img
                else:
                    self.target_box = self.target_box.numpy()
                    d_w = (self.target_box[2]-self.target_box[1])*0.05
                    self.target_box[0] = self.target_box[0] + d_w
                    self.target_box[2] = self.target_box[2] - d_w
                    self.kf.means = self.target_box
        elif not self.data[0]:
            return img
            print("%s: There's no detected object"%self.detect.__name__)
        estimated_box, self.kalman_ready = self.kf.estimate(self.target_box)
        if self.kalman_ready:
            self.target_box = estimated_box
            self.target_box = self.filter_box(self.target_box) # Avoid negative value in target_box
            self.moving_average()
            self.matcher.update(img, self.target_box) # Update the bounding box

            self.distances = self.get_distance(self.target_box[3]-self.target_box[1],
                                               self.target_box[2]-self.target_box[0])
            cv2.rectangle(img,
                          (int(self.target_box[0]), int(self.target_box[1])),
                          (int(self.target_box[2]), int(self.target_box[3])),
                          (0, 255, 0), 2)
            iou = self.compute_iou(track_loc, self.target_box)
            cv2.putText(img, "IOU: %.2f"%iou, (int(self.target_box[0]+10), int(self.target_box[3]-20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 127, 255), lineType=50)
            cv2.putText(img, "Confidence: %.2f"%confidence, (int(self.target_box[0]+10), int(self.target_box[3]-40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 127, 255), lineType=50)

            for idx, dist in enumerate(self.distances):
                cv2.putText(img, "Distance %d: %.3f cm"%(idx, dist), (int(self.target_box[0]+10), int(30+self.target_box[1]+idx*30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), lineType=50)
        return img

    def detect(self, image, data):
        """Inference the detector to detect all human object in images"""
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, _ = self.predictor.predict(image, 20, 0.6)
        interval = timer.end()
        if labels.size(0) != 0:
            data[0] = True
            data[1] = boxes
        else:
            data[0] = False
            print("%s: There's no detected object"%self.detect.__name__)

        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

    def compute_iou(self, boxA, boxB, eps=1e-5):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        intersect = max(0, xB - xA + eps) * max(0, yB - yA + eps)
        box1_w, box1_h = boxA[2]-boxA[0], boxA[3]-boxA[1]
        box2_w, box2_h = boxB[2]-boxB[0], boxB[3]-boxB[1]
        total_area = (box1_w*box1_h) + (box2_w*box2_h)
        return intersect/(total_area-intersect)

    def get_distance(self, h, w):
        return (self.kh/h, self.kw/w)

    def filter_box(self, box):
        """To avoid box coord with negative value"""
        for i, b in enumerate(box):
            if i < 2:
                if b < 0:
                    box[i] = 1
                else:
                    box[i] = min(self.image_shape[1-i], b)
            else:
                break
        return box

    def moving_average(self):
        self.box_cache = np.append(self.box_cache, [self.target_box], axis=0)
        self.tick += 1
        if self.tick > self.period:
            self.target_box = np.mean(self.box_cache, axis=0)
            self.box_cache = np.delete(self.box_cache, [0], axis=0)

from time import time
from os.path import isdir
from glob import glob
import sys
import cv2
import numpy as np
import torch
from torch import load, set_flush_denormal
from vision.ssd.fpnnet_ssd import create_fpnnet_ssd, create_fpn_ssd_predictor
from vision.utils.misc import Timer
from tracker import Tracker
from utils import Flags

set_flush_denormal(True)
if len(sys.argv) < 3:
    print('Usage: python run_ssd_example.py <net> <model path> <label path> [video file]')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]

is_dir = False
files = []
if len(sys.argv) >= 4:
    if isdir(sys.argv[3]):
        files = [f for f in glob('%s/*'%sys.argv[3])]
        is_dir = True
    else:
        cap = cv2.VideoCapture(sys.argv[3])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 640)
    cap.set(4, 480)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = create_fpnnet_ssd(num_classes, is_test=True)
net.load_state_dict(load(model_path, map_location=lambda storage, loc: storage))
net.eval()
predictor = create_fpn_ssd_predictor(net, candidate_size=200, device=DEVICE)


def predict(image):
    timer.start()
    boxes, labels, probs = predictor.predict(image, 40, 0.7)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    return image, boxes.size(0)

if __name__ == '__main__':
    timer = Timer()
    flags = Flags()
    tracker = Tracker(kw=500, kh=700, predictor=predictor, flags=flags)
    try:
        if is_dir:
            for file in files:
                img = cv2.imread(file)
    #            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, n_box = predict(img)
                file_name = file.strip().split('samples/')[1]
                print(file_name)
                cv2.imwrite('predicted/%s'%file_name, img)
                #cv2.imshow("Pred", img)
                #cv2.waitKey()
        else:
            last_time = time()
            frame_time = time()
            fpss = np.empty((1,))
            t = 0
            while True:
                ret, img = cap.read()
                if img is None:
                    continue
    #           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #img = predict(img)
                img = tracker(img)
                cv2.imshow('annotated', img)
                p_time = (time()-last_time)*1000
                if cv2.waitKey(max(int(32-p_time), 1)) & 0xFF == ord('q'):
                    break
                last_time = time()
                fps = 1/(time()-frame_time)
                fpss = np.append(fpss, fps)
                if t >= 10:
                    print("%.3f FPS"%np.mean(fpss[1:], axis=0))
                    fpss = np.empty((1,))
                    t = 0
                frame_time = time()
                t += 1
    except Exception as e:
        print(e)
        cap.release()
        cv2.destroyAllWindows()

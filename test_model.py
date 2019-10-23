from time import time
from glob import glob
import sys
import cv2
import numpy as np
from torch import load, set_flush_denormal
from vision.ssd.fpnnet_ssd import create_fpnnet_ssd, create_fpn_ssd_predictor
from vision.utils.misc import Timer
from tracker import Tracker
from utils import Flags
from torchsummary import summary
import xml.etree.ElementTree as ET
from tqdm import tqdm

#set_flush_denormal(True)
if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <model path> <label path> [test.txt]')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]
test_set = sys.argv[3]

with open(test_set, 'r') as f:
    files = f.readlines()

print(len(files))
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_fpnnet_ssd(num_classes, is_test=True)
net.load_state_dict(load(model_path, map_location=lambda storage, loc: storage))
net.eval()
predictor = create_fpn_ssd_predictor(net, candidate_size=200)


#summary(net, input_size=(3, 300, 300))
timer = Timer()

def compute_iou(boxA, boxB, eps=1e-5):
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

def predict(image):
    timer.start()
    boxes, labels, probs = predictor.predict(image, 20, 0.5)
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

total_face = 0
predicted_face = 0
for file in tqdm(files[:20]):
    annotation_file = "Datasets/Annotations/%s.xml"%file.strip()
    objects = ET.parse(annotation_file).findall("object")
    img = cv2.imread("Datasets/image_in_image/%s"%file.strip())
    img, n_box = predict(img)
    total_face += len(objects)
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text) - 1
        y1 = int(bbox.find('ymin').text) - 1
        x2 = int(bbox.find('xmax').text) - 1
        y2 = int(bbox.find('ymax').text) - 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    predicted_face += min(n_box, len(objects))
    cv2.imwrite("predicted/%s_%d_of_%d.jpg"%(file.split(".")[0], n_box, len(objects)), img)
with open("result.txt", 'w') as f:
    accuracy = (predicted_face/total_face)*100
    f.writelines("Akurasi model: %.3f percent"%accuracy)

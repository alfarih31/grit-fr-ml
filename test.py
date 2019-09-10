import cv2

with open("ImageSets/train.txt", "r") as f:
    files = f.readlines()

for f in files:
    img = cv2.imread("JPEGImages/%s"%f.strip())
    if img is None:
        files.remove(f)

with open("train.txt", "w") as f:
    for x in files:
        f.writelines("%s"%x.strip())

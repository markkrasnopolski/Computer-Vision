import cv2
import cv2 as cv
import numpy as np
import shutil
import os

from CV_home12 import best_conf

PROJECT_DIR = os.path.dirname(__file__)

IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "out")
PEOPLE_DIR = os.path.join(PROJECT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(PROJECT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

PROTOTXT_DIR = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD.caffemodel")

net = cv.dnn.readNetFromCaffe(PROTOTXT_DIR, MODEL_PATH)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

PERSON_CLASS_ID = CLASSES.index("person")

CONF_THRESHOLD = 0.6

def detect_person(image):
    (h, w) = image.shape[:2]

    lob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), mean=(104, 117, 123), swapRB=True, crop=False)

    net.setInput(lob)
    detections = net.forward()

    best_conf = 0.0
    best_box = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = detections[0, 0, i, 1]

        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            if confidence > best_conf:
                best_conf = confidence
                best_box = (x1, y1, x2, y2)
    found = best_box is not None
    return found, best_conf, best_box

allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp")
files = os.listdir(IMAGE_DIR)

count_people = 0
count_no_people = 0

for file in files:
    if not file.lower().endswith(allowed_extensions):
        continue

    in_path = os.path.join(IMAGE_DIR, file)

    img = cv2.imread(in_path)

    found, best_conf, best_box = detect_person(img)
    if found:
        out_path = os.path.join(OUT_DIR, file)
        shutil.copy(in_path, out_path)
        count_people += 1

        boxed = img.copy()
        x1, y1, x2, y2 = best_box
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(boxed, best_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        boxed_path = os.path.join(PEOPLE_DIR, "boxed" + file)
        cv2.imwrite(boxed_path, boxed)

    else:
        count_no_people += 1
        out_path = os.path.join(NO_PEOPLE_DIR, file)
        shutil.copy(in_path, out_path)
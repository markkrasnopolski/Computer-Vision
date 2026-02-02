import cv2
import numpy as np
import os
import shutil


PROJECT_DIR = os.path.dirname(__file__)

IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "out")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

PROTOTXT_DIR = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD.caffemodel")


net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = CLASSES.index("person")
CONF_THRESHOLD = 0.6


def detect_persons(image):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 117, 123),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            boxes.append((x1, y1, x2, y2))
            confidences.append(confidence)

    return boxes, confidences


allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp")
files = os.listdir(IMAGE_DIR)

count_people = 0
count_no_people = 0

for file in files:
    if not file.lower().endswith(allowed_extensions):
        continue

    image_path = os.path.join(IMAGE_DIR, file)
    image = cv2.imread(image_path)

    boxes, confidences = detect_persons(image)
    people_count = len(boxes)

    output_image = image.copy()

    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            output_image,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        output_image,
        f"People count: {people_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    if people_count == 0:
        save_dir = NO_PEOPLE_DIR
        count_no_people += 1
    else:
        save_dir = PEOPLE_DIR
        count_people += 1

    shutil.copy(image_path, os.path.join(save_dir, file))
    cv2.imwrite(os.path.join(save_dir, "boxed_" + file), output_image)


print("Processing finished")
print(f"Images with people: {count_people}")
print(f"Images without people: {count_no_people}")

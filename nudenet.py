#!/bin/env python

import os
import sys
import math
import time
import cv2
import numpy as np
import onnxruntime
from PIL import Image
from onnxruntime.capi import _pybind_state as C


session = None
detector = None


labels = [
    "female genitalia",
    "female face",
    "buttocks exposed",
    "female breast exposed",
    "female genitalia exposed",
    "male breast exposed",
    "anus exposed",
    "feed exposed",
    "bellt",
    "feet",
    "armpits",
    "armpits exposed",
    "male face",
    "belly exposed",
    "male genitalia exposed",
    "anus",
    "female breast",
    "buttocks",
]


class NudeResult:
    output: None
    censor: list = []
    detections: list = []
    censored: list = []


class NudeDetector:
    def __init__(self, providers=None, model=None):
        global session # pylint: disable=global-statement
        model = model or os.path.join(os.path.dirname(__file__), 'nudenet.onnx')
        if session is None:
            session = onnxruntime.InferenceSession(model, providers=C.get_available_providers() if not providers else providers) # pylint: disable=no-member
        model_inputs = session.get_inputs()
        self.input_width = model_inputs[0].shape[2] # 320
        self.input_height = model_inputs[0].shape[3] # 320
        self.input_name = model_inputs[0].name


    def read_image(self, image, target_size=320):
        if type(image) == str:
            img = cv2.imread(image)
        else:
            img = image
        img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aspect = img_width / img_height
        if img_height > img_width:
            new_height = target_size
            new_width = int(round(target_size * aspect))
        else:
            new_width = target_size
            new_height = int(round(target_size / aspect))
        resize_factor = math.sqrt((img_width**2 + img_height**2) / (new_width**2 + new_height**2))
        img = cv2.resize(img, (new_width, new_height))
        pad_x = target_size - new_width
        pad_y = target_size - new_height
        pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
        pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.resize(img, (target_size, target_size))
        image_data = img.astype("float32") / 255.0  # normalize
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)
        return image_data, resize_factor, pad_left, pad_top

    def postprocess(self, output, resize_factor, pad_left, pad_top, min_score):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= min_score:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int(round((x - w * 0.5 - pad_left) * resize_factor))
                top = int(round((y - h * 0.5 - pad_top) * resize_factor))
                width = int(round(w * resize_factor))
                height = int(round(h * resize_factor))
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        res = []
        for i in indices: # pylint: disable=not-an-iterable
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            res.append({"label": labels[class_id], "id": class_id, "score": round(float(score), 2), "box": box})
        return res

    def pixelate(self, image, blocks=3):
        (h, w) = image.shape[:2] # divide the input image into NxN blocks
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
        return image

    def detect(self, image, min_score):
        preprocessed_image, resize_factor, pad_left, pad_top = self.read_image(image, self.input_width)
        outputs = session.run(None, {self.input_name: preprocessed_image})
        res = self.postprocess(outputs, resize_factor, pad_left, pad_top, min_score)
        return res

    def censor(self, image, min_score=0.2, censor=None, method='pixelate'):
        if type(image) == str:
            image = cv2.imread(image) # input is image path
        else:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # input is pil image
        nude = NudeResult()
        nude.censor = censor or []
        nude.detections = self.detect(image, min_score)
        nude.censored = [d for d in nude.detections if d["label"] in nude.censor]
        for d in nude.censored:
            box = d["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            area = image[y: y+h, x: x+w]
            if method == 'pixelate':
                area = self.pixelate(image[y: y+h, x: x+w], blocks=3)
            elif method == 'blur':
                area = cv2.blur(image[y: y+h, x: x+w], (23, 23))
            elif method == 'block':
                area = (0, 0, 0)
            image[y: y+h, x: x+w] = area
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nude.output = Image.fromarray(image)
        return nude


def cli():
    from rich import print # pylint: disable=redefined-builtin, import-outside-toplevel
    global detector # pylint: disable=global-statement
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        print('nudenet:', 'no files specified')
    for fn in sys.argv:
        t0 = time.time()
        pil = Image.open(fn)
        if detector is None:
            detector = NudeDetector()
        nudes = detector.censor(image=pil, censor=['female breast exposed', 'female genitalia exposed'], min_score=0.2, method='pixelate')
        t1 = time.time()
        print(vars(nudes))
        f = os.path.splitext(fn)[0] + '_censored.jpg'
        nudes.output.save(f)
        print(f'nudenet: input={fn} output={f} time={t1-t0:.2f}s')


if __name__ == "__main__":
    cli()

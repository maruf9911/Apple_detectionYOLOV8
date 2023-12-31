# -*- coding: utf-8 -*-
"""Untitled68.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qQl4ie5xI8IrmnT3kNR-l_rYZlVTqQAR
"""

!pip install ultralytics

from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo mode=checks

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3gTgbuoqfI4V25XRZp8O")
project = rf.workspace("lakshantha-dissanayake").project("apple-detection-5z37o")
dataset = project.version(1).download("yolov8")

!yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=20 imgsz=640

!yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model=/content/runs/detect/train2/weights/best.pt conf=0.5 source=/content/Apple-Detection-1/valid/images save_txt=true save_conf=true

import glob
from IPython.display import Image, display

for image_path in glob.glob(f"/content/runs/detect/predict/apple--363-_jpg.rf.6046be0878512abd0436bce59c2a5605.jpg"):
  display(Image(filename=image_path, height=600))
  print("\n")


from ultralytics import YOLO
import glob
import os

if __name__ == '__main__':
# Load a model
    model = YOLO(r'E:\ultralytics-main-improve\datasets\medicine bottle\runs\segment\train88\weights\best.pt')  # load an official model
    model.val(data=r'E:\ultralytics-main-improve\datasets\medicine bottle\medicine bottle.yaml',
          split='test',
          imgsz=640,
          batch=16,
          save=True,
          )


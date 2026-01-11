# import /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src/dataset/dataset.py
# it is in: ../src/dataset/dataset.py
import os
import pickle
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import ThermalDataset

dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0")

data = dataset.get_ira(0)
print(data)
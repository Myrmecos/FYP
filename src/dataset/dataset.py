
import os
import pickle
import sys
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_collection_module import utils

class ThermalDataset():
    def __init__(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir
        self.camera_dir = os.path.join(dataset_base_dir, "Camera")
        self.ira_dir = os.path.join(dataset_base_dir, "IRA")
        self.tof_dir = os.path.join(dataset_base_dir, "ToF")

        self.img_files = sorted([f for f in os.listdir(self.camera_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.ira_files = sorted([f for f in os.listdir(self.ira_dir) if f.endswith('.pkl')])
        # ira_highres_files = sorted([f for f in os.listdir(ira_highres_dir) if f.endswith('.pkl')])
        self.tof_files = sorted([f for f in os.listdir(self.tof_dir) if f.endswith('.pkl')])
    
    def __len__(self):
        return len(self.img_files)

    def get_camera_image_path(self, index):
        return os.path.join(self.camera_dir, self.img_files[index])
    
    def get_ira_image_path(self, index):
        return os.path.join(self.ira_dir, self.ira_files[index])
    
    def get_tof_image_path(self, index):
        return os.path.join(self.tof_dir, self.tof_files[index])

    def get_image(self, index):
        img_path = self.get_camera_image_path(index)
        return cv2.imread(img_path)
    
    def get_ira(self, index):
        ira_path = self.get_ira_image_path(index)
        with open(ira_path, 'rb') as f:
            pkl_data = pickle.load(f)[-1]
            pkl_data = pkl_data["ira_temp"][0]
        return pkl_data
    
    def get_ira_highres(self, index):
        ira_path = self.get_ira_image_path(index)
        with open(ira_path, 'rb') as f:
            pkl_data = pickle.load(f)[-1]
            pkl_data = pkl_data["ira_temp"][-1]
        return pkl_data

    def get_tof(self, index):
        tof_path = self.get_tof_image_path(index)
        with open(tof_path, 'rb') as f:
            pkl_data = pickle.load(f)[-1]
            pkl_data = pkl_data['tof_depth'][:, :, 0]
        return pkl_data
    
        
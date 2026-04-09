
import os
import pickle
import sys
from pathlib import Path
from torch.utils.data import dataset

import cv2
sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

class ThermalDatasetAggregator():
    """A class to aggregate the dataset and provide easy access to the data.
    This class will load dataset from multiple folders and combine them into a single dataset.
    """
    def __init__(self, dataset_base_dirs):
        self.dataset_base_dirs = dataset_base_dirs
        self.datasets = [ThermalDataset(dataset_base_dir) for dataset_base_dir in dataset_base_dirs]
        self.img_files = []
        self.ira_files = []
        self.tof_files = []
        self.annotations = []
        for dataset in self.datasets:
            self.img_files.extend(dataset.img_files)
            self.ira_files.extend(dataset.ira_files)
            self.tof_files.extend(dataset.tof_files)
            self.annotations.extend(dataset.annotations_expanded)
    def __len__(self):
        "total number of frames"
        lngth = min(len(self.img_files), len(self.ira_files), len(self.tof_files))
        return lngth
    
    def subset_number(self):
        """How many datasets are aggregated"""
        return len(self.datasets)
    
    #  ======== loading data from a specific dataset and frame index ==========
    def subset_len(self, dataset_idx):
        """number of frames in a specific dataset"""
        return len(self.datasets[dataset_idx])
    
    def get_image_ij(self, dataset_idx, frame_idx):
        return self.datasets[dataset_idx].get_image(frame_idx)
    
    def get_ira_ij(self, dataset_idx, frame_idx):
        return self.datasets[dataset_idx].get_ira(frame_idx)
    
    def get_ira_highres_ij(self, dataset_idx, frame_idx):
        return self.datasets[dataset_idx].get_ira_highres(frame_idx)
    
    def get_tof_ij(self, dataset_idx, frame_idx):
        return self.datasets[dataset_idx].get_tof(frame_idx)
    
    # ============ loading data from the aggregated dataset and frame index ============
    def get_image(self, index):
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index].get_image(index)
    def get_ira(self, index):
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index].get_ira(index)
    def get_ira_highres(self, index):
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index].get_ira_highres(index)
    def get_tof(self, index):
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index].get_tof(index)

class ThermalDataset(dataset.Dataset):
    def __init__(self, dataset_base_dir, noCam = False):
        self.dataset_base_dir = dataset_base_dir
        self.camera_dir = os.path.join(dataset_base_dir, "Camera")
        self.ira_dir = os.path.join(dataset_base_dir, "IRA")
        self.tof_dir = os.path.join(dataset_base_dir, "ToF")
        self.annotation_path = os.path.join(dataset_base_dir, "annotation.yaml")
        self.noCam = noCam

        if not noCam:
            self.img_files = sorted([f for f in os.listdir(self.camera_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.ira_files = sorted([f for f in os.listdir(self.ira_dir) if f.endswith('.pkl')])
        # ira_highres_files = sorted([f for f in os.listdir(ira_highres_dir) if f.endswith('.pkl')])
        self.tof_files = sorted([f for f in os.listdir(self.tof_dir) if f.endswith('.pkl')])
        self.annotations_expanded = [-1 for _ in range(self.__len__())]  # type: List[int]
        self.process_annotations()
    
    def __getitem__(self, index):
        if not self.noCam:
            img = self.get_image(index)
        else:
            img = None
        ira = self.get_ira(index)
        ira_highres = self.get_ira_highres(index)
        tof = self.get_tof(index)
        label = self.get_label(index)
        data_dict = {
            "image": img,
            "ira": ira,
            "ira_highres": ira_highres,
            "tof": tof,
        }
        return data_dict, label
    
    def process_annotations(self):
        """annotation format: 
            valid_frame_indices: [[10, 1800]] # ranges of valid frames for this dataset, can be multiple ranges for one dataset
            presence: [[100, 1700]]
            standing: [[150, 200], [1400, 1600]] # ranges of standing events
            sitting_by_bed: [[220, 400], [1000, 1350]] # ranges of sitting by bed events
            sitting_on_bed: [[450, 600], [900, 950]] # ranges of sitting on bed events
            lying_no_cover: [[650, 700], [800, 850]] # ranges of lying without cover events
            lying_with_cover: [[720, 780]] # ranges of lying with cover events
            # -1: unknown or unlabeled; 
            # 0: absence; 
            # 1: presence, unclassified; 
            # 2: standing; 
            # 3: sitting by bed; 
            # 4: sitting on bed; 
            # 5: lying w/o cover; 
            # 6: lying with cover"""
        # self.annotations_expanded = [-1 for _ in range(self.__len__())]
        if not os.path.exists(self.annotation_path):
            print(f"Warning: annotation file {self.annotation_path} does not exist. All frames will be labeled as -1 (unknown).")
            return
        self.annotations = load_yaml(self.annotation_path)
        if len(self.annotations) == 0:
            print(f"Warning: annotation file {self.annotation_path} is empty. All frames will be labeled as -1 (unknown).")
            return
        try:
            for entry in self.annotations["valid_frame_indices"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 0
            for entry in self.annotations["presence"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 1
            for entry in self.annotations["standing"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 2
            for entry in self.annotations["sitting_by_bed"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 3
            for entry in self.annotations["sitting_on_bed"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 4
            for entry in self.annotations["lying_without_cover"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 5
            for entry in self.annotations["lying_with_cover"]:
                for i in range(entry[0], entry[1]+1):
                    self.annotations_expanded[i] = 6
        except:
            print("problems with annotation format. Please check the annotation.yaml file. All frames will be labeled as -1 (unknown).")
            # print error mesasge
            print("problem set name: ", self.dataset_base_dir)
            import traceback
            traceback.print_exc()
    def __len__(self):
        if self.noCam:
            lngth = min(len(self.ira_files), len(self.tof_files))
        else:
            lngth = min(len(self.img_files), len(self.ira_files), len(self.tof_files))
        return lngth
    
    def get_annotation(self, index):
        return self.annotations_expanded[index]

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
    
    def get_label(self, index):
        """get the presence label from the annotation.yaml file"""
        ret = -1 
        # -1: unknown or unlabeled; 
        # 0: absence; 
        # 1: presence, unclassified; 
        # 2: standing; 
        # 3: sitting by bed; 
        # 4: sitting on bed; 
        # 5: lying w/o cover; 
        # 6: lying with cover
        # annotation = load_yaml(self.annotation_path)
        label = self.annotations_expanded[index]
        return label
    

if __name__ == "__main__":
    import cv2
    from data_visualization_module.plot import DataVisualizer
    visualizer = DataVisualizer()
    dataset_base_dirs = [
        "entry_exit_detection/presence_detection_workspace/data/office0_1",
        "entry_exit_detection/presence_detection_workspace/data/office0_2",
    ]
    dataset = ThermalDatasetAggregator(dataset_base_dirs)
    print(f"Total dataset length: {len(dataset)}")
    # check each dataset
    for i in range(dataset.subset_number()):
        print(f"Dataset {i} length",dataset.subset_len(i))
        for j in range(648, 648+20):
            img = dataset.get_image_ij(i, j)
            ira = dataset.get_ira_ij(i, j)
            ira_highres = dataset.get_ira_highres_ij(i, j)
            tof = dataset.get_tof_ij(i, j)
            print(f"Dataset {i} frame {j} shapes: img {img.shape}, ira {ira.shape}, ira_highres {ira_highres.shape}, tof {tof.shape}")
            color_thermal = visualizer.compose_color_and_thermal(img, ira, ira_highres)
            cv2.imshow("Color and Thermal", color_thermal)
            cv2.waitKey(0)


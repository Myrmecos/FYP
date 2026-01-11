# TODO: detect heat source presence in thermal image
# report the indices of heat source presence frames
import os
import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset.dataset import ThermalDataset
from src.heatsource_detection_module.extract import HeatSourceDetector 
import numpy as np

if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0")
    detector = HeatSourceDetector()
    heat_source_indices = []
    thresholds = []
    masked_medians = []
    masked_means = []
    for idx in range(len(dataset)):
        ira_highres = dataset.get_ira_highres(idx)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        thresholds.append(thresh)
        # compute median of masked area
        median_val = detector.get_masked_median(ira_highres, mask)
        masked_medians.append(median_val)
        mean_val = detector.get_masked_mean(ira_highres, mask)
        masked_means.append(mean_val)
    # plot changes of medians and thresholds against index
    plt.plot(masked_medians)
    plt.plot(thresholds)
    plt.plot(masked_means)
    plt.show()
    print("Indices with heat source presence:", heat_source_indices)
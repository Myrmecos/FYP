# plot some useful figures for data visualization

import os
import sys
from pathlib import Path
import logging

# Suppress matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from dataset import ThermalDataset
from heatsource_detection_module.extract import HeatSourceDetector
from data_collection_module import utils

class DataVisualizer:
    def __init__(self):
        pass

    def plot_heatmap(self, ira_img, title="IRA Heatmap"):
        plt.imshow(ira_img, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.show()

    def plot_histogram(self, ira_img, title="IRA Histogram", bins=50):
        plt.hist(ira_img.flatten(), bins=bins, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()
    
    def _get_heatsource_means_highres(self, dataset):
        means = []
        for idx in range(len(dataset)):
            ira_highres = dataset.get_ira_highres(idx)
            thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
            means.append(mean_val)
        return means
    
    def _get_feature_highres(self, dataset, feature='median'):
        features = []
        if feature == 'median':
            function = detector.get_masked_median
        elif feature == 'mean':
            function = detector.get_masked_mean
        elif feature == 'threshold':
            thresholds = []
            for idx in range(len(dataset)):
                ira_highres = dataset.get_ira_highres(idx)
                thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
                thresholds.append(thresh)
            return thresholds
        for idx in range(len(dataset)):
            ira_highres = dataset.get_ira_highres(idx)
            thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
            feature_val = function(ira_highres, mask)
            features.append(feature_val)
        return features
    
    def plot_features_highres_over_time(self, dataset):
        means = self._get_feature_highres(dataset, "mean")
        medians = self._get_feature_highres(dataset, "median")
        thresholds = self._get_feature_highres(dataset, "threshold")
        plt.plot(means)
        plt.plot(medians)
        plt.plot(thresholds)
        plt.title("Mean, median, threshold of IRA High-Res over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean IRA High-Res Value")
        # add legend
        plt.legend(["Mean", "Median", "Threshold"])
    
    def plot_presence_segments(self, yaml_path):
        presence_segments = self.construct_presence_segemnts(yaml_path)
        plt.figure()
        for start, end in presence_segments:
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.title("Presence Segments")
        plt.xlabel("Frame Index")
        plt.ylabel("Presence")
        plt.yticks([0, 1], ['Absent', 'Present'])
    
    def construct_presence_segemnts(self, yaml_path):
        info = utils.load_yaml_as_dict(yaml_path)
        occlusion_starts = info['occlusion_start']
        occlusion_ends = info['occlusion_end']
        presence_segments = []
        for start, end in zip(occlusion_starts, occlusion_ends):
            presence_segments.append((start, end))
        return presence_segments


if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0")
    detector = HeatSourceDetector()
    visualizer = DataVisualizer()
    visualizer.plot_presence_segments("presence_detection_workspace/data/hall0/annotation.yaml")
    visualizer.plot_features_highres_over_time(dataset)
    plt.show()
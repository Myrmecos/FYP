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
            means.append(thresh)
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
        plt.plot(means, label="Mean")
        plt.plot(medians, label="Median")
        plt.plot(thresholds, label="Threshold")
        plt.title("Mean, median, threshold of IRA High-Res over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean IRA High-Res Value")
        # add legend
        plt.legend()
    
    def plot_occlusion_segments(self, yaml_path, new_figure=True):
        occlusion_segments = self.construct_occlusion_segments(yaml_path)
        if new_figure:
            plt.figure()
        for i, (start, end) in enumerate(occlusion_segments):
            # Only label the first segment to avoid duplicate legend entries
            label = "Occlusion Segment" if i == 0 else None
            plt.axvspan(start, end, color='red', alpha=0.1, label=label)
        if new_figure:
            plt.title("Occluded Segments")
            plt.xlabel("Frame Index")
            plt.ylabel("Presence")
            plt.yticks([0, 1], ['Clear', 'Occluded'])
    
    def plot_presence_segments(self, yaml_path, new_figure=True):
        presence_segments = self.construct_presence_segments(yaml_path)
        if new_figure:
            plt.figure()
        for i, (start, end) in enumerate(presence_segments):
            # Only label the first segment to avoid duplicate legend entries
            label = "Presence Segment" if i == 0 else None
            plt.axvspan(start, end, color='red', alpha=0.1, label=label)
        if new_figure:
            plt.title("Presence Segments")
            plt.xlabel("Frame Index")
            plt.ylabel("Presence")
            plt.yticks([0, 1], ['Absent', 'Present'])
    
    def construct_occlusion_segments(self, yaml_path):
        info = utils.load_yaml_as_dict(yaml_path)
        occlusion_starts = info['occlusion_start']
        occlusion_ends = info['occlusion_end']
        occlusion_segments = []
        for start, end in zip(occlusion_starts, occlusion_ends):
            occlusion_segments.append((start, end))
        return occlusion_segments

    def construct_presence_segments(self, yaml_path):
        info = utils.load_yaml_as_dict(yaml_path)
        entries = info['entries']
        exits = info['exits']
        presence_segments = []
        for entry, exit in zip(entries, exits):
            presence_segments.append((entry, exit))
        return presence_segments

if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0")
    detector = HeatSourceDetector()
    visualizer = DataVisualizer()
    visualizer.plot_features_highres_over_time(dataset)
    visualizer.plot_presence_segments("presence_detection_workspace/data/hall0/annotation.yaml", new_figure=False)
    visualizer.plot_occlusion_segments("presence_detection_workspace/data/hall0/annotation.yaml", new_figure=False)
    plt.show()
# plot some useful figures for data visualization

import os
import sys
from pathlib import Path
import logging
import cv2

# Suppress matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from dataset import ThermalDataset
from heatsource_detection_module.extract import HeatSourceDetector
from data_collection_module import utils

def stitch_images(images):
    resized_images = []
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * (max_height / h))
        resized_img = cv2.resize(img, (new_w, max_height))
        resized_images.append(resized_img)
    stitched_image = cv2.hconcat(resized_images)
    return stitched_image

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
        """returns threshold of IRA high-res over time as a list"""
        means = []
        for idx in range(len(dataset)):
            ira_highres = dataset.get_ira_highres(idx)
            thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
            means.append(thresh)
        return means
    
    def _get_feature_highres(self, dataset, feature='median'):
        """returns the specified feature of IRA high-res over time as a list"""
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
    
    # plot the mean, median, threshold of IRA high-res over time in the same figure
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

    def _prepare_thermal_for_colormap(self, thermal_img):
        """Convert temperature maps into a valid uint8 image for cv2.applyColorMap."""
        thermal_img = np.asarray(thermal_img)
        thermal_img = np.nan_to_num(thermal_img, nan=0.0, posinf=0.0, neginf=0.0)

        if thermal_img.ndim == 3 and thermal_img.shape[2] == 1:
            thermal_img = thermal_img[:, :, 0]

        if thermal_img.dtype != np.uint8:
            thermal_img = thermal_img.astype(np.float32)
            if np.ptp(thermal_img) == 0:
                thermal_img = np.zeros_like(thermal_img, dtype=np.uint8)
            else:
                thermal_img = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX)
                thermal_img = thermal_img.astype(np.uint8)

        return thermal_img

    # helper that aids visualization of data
    # stitches RGB image, IRA low-res, IRA high-res together in a color image
    def compose_color_and_thermal(self, rgb_img, m08_img, mlx_img):
        # color m08 and mlx
        m08_vis = self._prepare_thermal_for_colormap(m08_img)
        mlx_vis = self._prepare_thermal_for_colormap(mlx_img)
        m08_color = cv2.applyColorMap(m08_vis, cv2.COLORMAP_JET)
        mlx_color = cv2.applyColorMap(mlx_vis, cv2.COLORMAP_JET)

        # stitch the three images together
        stitched = stitch_images([rgb_img, m08_color, mlx_color])
        return stitched


if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    visualizer = DataVisualizer()
    visualizer.plot_features_highres_over_time(dataset)
    visualizer.plot_presence_segments("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1/annotation.yaml", new_figure=False)
    visualizer.plot_occlusion_segments("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1/annotation.yaml", new_figure=False)
    plt.show()
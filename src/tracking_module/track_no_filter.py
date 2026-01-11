import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from organizer_module.blob import Blob
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.blobs = []  # list of Blob objects
    
    # move Hungarian algorithm out to here
    def _associate_blobs(self, detected_heat_sources):
        # we use simple IoU based tracking
        # the intersection over union (IoU) between existing blobs and detected heat sources
        # detected_heat_sources: a list of (mask) of detected heat sources [mask0, mask1, ...]
        #       the masks are binary numpy arrays of the same shape as original_ira_img
        # existing blobs: self.blobs, containing Blobs objects [blob0, blob1, ...]
        #       blob.get_mask() returns the mask of the blob, 
        #       which will be used to compare against detected_heat_sources to compute IoU

        # TODO: use Hungarian algorithm to associate existing blobs with detected heat sources

        # 1. compute IoU matrix
        num_existing = len(self.blobs)
        num_detected = len(detected_heat_sources)
        iou_matrix = np.zeros((num_existing, num_detected))
        for i, blob in enumerate(self.blobs):
            old_mask = blob.get_mask()
            for j, new_mask in enumerate(detected_heat_sources):
                iou = self._compute_iou(old_mask, new_mask)
                iou_matrix[i, j] = iou
        # 2. Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize IoU

        # 3. create matching result
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > 0.3:  # threshold for matching
                matched_pairs.append((r, c))
        return matched_pairs

    
    def update_blobs(self, detected_heat_sources, original_ira_img, background_avg):
        # we use simple IoU based tracking
        # the intersection over union (IoU) between existing blobs and detected heat sources
        # detected_heat_sources: a list of (mask) of detected heat sources [mask0, mask1, ...]
        #       the masks are binary numpy arrays of the same shape as original_ira_img
        # existing blobs: self.blobs, containing Blobs objects [blob0, blob1, ...]
        #       blob.get_mask() returns the mask of the blob,
        matched_pairs = self._associate_blobs(detected_heat_sources)
        num_existing = len(self.blobs)
        num_detected = len(detected_heat_sources)

        for r, c in matched_pairs:
            blob = self.blobs[r]
            mask = detected_heat_sources[c]
            masked_temps = original_ira_img[mask.astype(bool)]
            blob.update(mask, masked_temps)
        unmatched_existing_blobs_indices = set(range(num_existing)) - set([r for r, c in matched_pairs])
        unmatched_new_heat_sources_indices = set(range(num_detected)) - set([c for r, c in matched_pairs])

        # check for new heat sources that do not match existing blobs
        for source_idx in unmatched_new_heat_sources_indices:
            source = detected_heat_sources[source_idx]
            mask = source
            masked_temps = original_ira_img[mask.astype(bool)]
            avg_temp = masked_temps.mean()
            if avg_temp < background_avg + 3:  # threshold to filter out noise
                continue
            new_blob = Blob()
            new_blob.update(mask, masked_temps)
            self.blobs.append(new_blob)
    def _compute_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        else:
            return intersection / union
        
# use data from hall1, frame 18055-18115 for testing
if __name__ == "__main__":
    from dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    tracker = Tracker()

    for idx in range(18000, 18315): #18115
        ira_highres = dataset.get_ira_highres(idx)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        cleaned_mask = detector.get_connected_components(mask, min_size=100)
        tracker.update_blobs(cleaned_mask, ira_highres, detector.get_unmasked_mean(ira_highres, mask))
    
    # plot the blobs' centroids movements
    for i, blob in enumerate(tracker.blobs):
        centroids = blob.centroid_history
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        plt.plot(xs, ys, marker='o', label=f'Blob {i}')
    plt.gca().invert_yaxis()  # invert y axis to match image coordinates
    plt.title("Blob Centroid Movements")
    plt.xlim(0, ira_highres.shape[1])
    plt.ylim(0, ira_highres.shape[0])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

        


import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from organizer_module.blob import Blob


class Tracker:
    def __init__(self):
        self.blobs = []  # list of Blob objects

    def _find_matching_heatsource(self, blob, detected_heat_sources):
        # find the detected heat source that matches the blob based on IoU
        # use Hungarian algorithm for association
        best_iou = 0
        best_source = None
        blob_mask = blob.mask.astype(bool)
        for i, source_mask in enumerate(detected_heat_sources):
            source_mask_bool = source_mask.astype(bool)
            intersection = np.logical_and(blob_mask, source_mask_bool).sum()
            union = np.logical_or(blob_mask, source_mask_bool).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou: 
                best_iou = iou
                best_source = source_mask
                best_index = i
        if best_source is not None:
            # remove the matched source from the list
            detected_heat_sources.pop(best_index)
        return best_source
    
    def update_blobs(self, detected_heat_sources, original_ira_img, background_avg):
        # we use simple IoU based tracking
        # the intersection over union (IoU) between existing blobs and detected heat sources
        # detected heat sources: a list of (mask) of detected heat sources [mask0, mask1, ...]
        for blob in self.blobs:
            matching_source = self._find_matching_heatsource(blob, detected_heat_sources) # the function removes matched source from detected_heat_sources
            if matching_source is not None:
                mask = matching_source
                masked_temps = original_ira_img[mask.astype(bool)]
                blob.update(mask, masked_temps)
            else:
                # no matching heat source found, for now we will not update
                # TODO: predict lost blobs via Kalman filter
                pass

        # check for new heat sources that do not match existing blobs
        for source in detected_heat_sources:
            mask = source
            masked_temps = original_ira_img[mask.astype(bool)]
            avg_temp = masked_temps.mean()
            if avg_temp < background_avg + 3:  # threshold to filter out noise
                continue
            new_blob = Blob()
            new_blob.update(mask, masked_temps)
            self.blobs.append(new_blob)

# use data from hall1, frame 18055-18115 for testing
if __name__ == "__main__":
    from dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    tracker = Tracker()
    for idx in range(18055, 18115):
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

        


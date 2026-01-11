import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from organizer_module.kalman_blob import KalmanBlob, mask_to_bbox
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.blobs = []  # list of KalmanBlob objects
    
    # move Hungarian algorithm out to here
    def _associate_blobs(self, detected_heat_sources, frame_shape):
        """
        Associate detections to tracks using IoU plus a centroid-distance fallback.
        This helps keep association when the predicted box drifts outside the frame
        (common near top edge) and IoU collapses to zero.
        """
        num_existing = len(self.blobs)
        num_detected = len(detected_heat_sources)
        score_matrix = np.zeros((num_existing, num_detected))
        img_h, img_w = frame_shape
        diag = np.sqrt(img_w ** 2 + img_h ** 2)

        for i, blob in enumerate(self.blobs):
            predicted_bbox = blob.predict()
            pred_cx = (predicted_bbox[0] + predicted_bbox[2]) / 2.0
            pred_cy = (predicted_bbox[1] + predicted_bbox[3]) / 2.0
            for j, new_mask in enumerate(detected_heat_sources):
                new_bbox = mask_to_bbox(new_mask)
                det_cx = (new_bbox[0] + new_bbox[2]) / 2.0
                det_cy = (new_bbox[1] + new_bbox[3]) / 2.0
                iou = self._compute_iou(predicted_bbox, new_bbox)
                # normalized center distance in [0, 1]
                dist = np.sqrt((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2) / diag
                # score balances IoU with proximity (higher is better)
                score_matrix[i, j] = iou + 0.3 * (1.0 - min(1.0, dist))

        # maximize score -> minimize negative score
        row_ind, col_ind = linear_sum_assignment(-score_matrix)

        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            iou = score_matrix[r, c]  # note: score, not pure IoU
            # accept if IoU good, or centers close even when IoU collapsed
            centers_close = score_matrix[r, c] >= 0.3  # blended metric threshold
            if centers_close:
                matched_pairs.append((r, c))
        return matched_pairs

    
    def update_blobs(self, detected_heat_sources, original_ira_img, background_avg):
        # we use simple IoU based tracking
        # the intersection over union (IoU) between existing blobs and detected heat sources
        # detected_heat_sources: a list of (mask) of detected heat sources [mask0, mask1, ...]
        #       the masks are binary numpy arrays of the same shape as original_ira_img
        # existing blobs: self.blobs, containing Blobs objects [blob0, blob1, ...]
        #       blob.get_mask() returns the mask of the blob,
        matched_pairs = self._associate_blobs(detected_heat_sources, original_ira_img.shape)
        num_existing = len(self.blobs)
        num_detected = len(detected_heat_sources)

        for r, c in matched_pairs:
            blob = self.blobs[r]
            mask = detected_heat_sources[c]
            masked_temps = original_ira_img[mask.astype(bool)]
            blob.update(mask, masked_temps, True)

        unmatched_existing_blobs_indices = set(range(num_existing)) - set([r for r, c in matched_pairs])
        unmatched_new_heat_sources_indices = set(range(num_detected)) - set([c for r, c in matched_pairs])

        # check for new heat sources that do not match existing blobs
        for source_idx in unmatched_new_heat_sources_indices:
            source = detected_heat_sources[source_idx]
            mask = source
            masked_temps = original_ira_img[mask.astype(bool)]
            avg_temp = masked_temps.mean()
            heatsource_size = np.sum(mask.astype(bool))
            if avg_temp < background_avg + 3 or heatsource_size < 400:  # threshold to filter out noise
                continue
            new_blob = KalmanBlob(mask=mask, masked_temps=masked_temps)
            self.blobs.append(new_blob)
        
        for blob_idx in unmatched_existing_blobs_indices:
            blob = self.blobs[blob_idx]
            blob.update(blob.get_mask(), blob.masked_temps, False)

    def _compute_iou(self, bbox1, bbox2):
        # bbox: [x_min, y_min, x_max, y_max]
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
        
# use data from hall1, frame 18055-18115 for testing
if __name__ == "__main__":
    from dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    tracker = Tracker()

    for idx in range(18230, 18260): #18115
        ira_highres = dataset.get_ira_highres(idx)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        cleaned_mask = detector.get_connected_components(mask, min_size=200)
        tracker.update_blobs(cleaned_mask, ira_highres, detector.get_unmasked_mean(ira_highres, mask))
    
    # plot the blobs' centroids movements

    for i, blob in enumerate(tracker.blobs):
        centroids = blob.kalman_centroid_history
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
    
    
        


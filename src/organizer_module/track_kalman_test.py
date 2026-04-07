import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from heat_patch_tracker_module.kalman_blob import KalmanBlob, mask_to_bbox
from residual_heat_detection_module.residual_detect import ResidualHeatDetector
from scipy.optimize import linear_sum_assignment, curve_fit
HUNG_THRESH = 0.2  # minimum score for match acceptance
SIZE_THRESH = 100  # minimum size of heat source to be considered a blob

class Tracker:
    def __init__(self):
        self.blobs = []  # list of KalmanBlob objects
        self.residual_detector = ResidualHeatDetector()
        self.relations = dict() # adjacency list to track the relations between blobs
    
    # move Hungarian algorithm out to here
    def _associate_blobs(self, detected_heat_sources, frame_shape):
        """
        Associate detections to tracks using IoU plus a centroid-distance fallback.
        This helps keep association when the predicted box drifts outside the frame
        (common near top edge) and IoU collapses to zero.
        # detected_heat_sources: list of masks of detected heat sources in the current frame
        # frame_shape: the shape of the original IRA image, used for normalizing distance
        """
        num_existing = len(self.blobs)
        num_detected = len(detected_heat_sources)
        score_matrix = np.zeros((num_existing, num_detected))
        img_h, img_w = frame_shape
        diag = np.sqrt(img_w ** 2 + img_h ** 2)
        neglect_residual = False

        if num_existing > num_detected:
            neglect_residual = True

        for i, blob in enumerate(self.blobs):
            predicted_bbox = blob.predict()
            pred_cx = (predicted_bbox[0] + predicted_bbox[2]) / 2.0
            pred_cy = (predicted_bbox[1] + predicted_bbox[3]) / 2.0
            for j, new_mask in enumerate(detected_heat_sources):
                new_bbox = mask_to_bbox(new_mask)
                det_cx = (new_bbox[0] + new_bbox[2]) / 2.0
                det_cy = (new_bbox[1] + new_bbox[3]) / 2.0
                iou = self._compute_iou(predicted_bbox, new_bbox)
                if neglect_residual and blob.is_residual==True: #neglect residual and is residual
                    iou *= 0.8  # soft ignore residual blobs
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
            centers_close = score_matrix[r, c] >= HUNG_THRESH  # blended metric threshold
            if centers_close:
                matched_pairs.append((r, c))
        return matched_pairs

    def update_relations(self, blob1, blob2):
        """ Update the adjacency list to track the relations between blobs (blob1 and blob2 are from the same split event)"""
        if blob1.id_fixed not in self.relations:
            self.relations[blob1.id_fixed] = set()
        if blob2.id_fixed not in self.relations:
            self.relations[blob2.id_fixed] = set()
        self.relations[blob1.id_fixed].add(blob2.id_fixed)
        self.relations[blob2.id_fixed].add(blob1.id_fixed)
    
    def update_blobs(self, detected_heat_sources, original_ira_img, background_avg, idx = -1):
        return_dict = {"human_in_scene": False, "bed_exit": False, "Frame index": idx}
        # we use simple IoU based tracking
        # the intersection over union (IoU) between existing blobs and detected heat sources
        # detected_heat_sources: a list of (mask) of detected heat sources [mask0, mask1, ...]
        #       the masks are binary numpy arrays of the same shape as original_ira_img
        # existing blobs: self.blobs, containing Blobs objects [blob0, blob1, ...]
        #       blob.get_mask() returns the mask of the blob,
        matched_pairs = self._associate_blobs(detected_heat_sources, original_ira_img.shape)
        num_existing = len(self.blobs)
        num_detected = len(detected_heat_sources)

        # =========== A. Match and update existing blobs ===========
        for r, c in matched_pairs:
            blob = self.blobs[r]
            mask = detected_heat_sources[c]
            masked_temps = original_ira_img[mask.astype(bool)]
            blob.update(mask, masked_temps, True)

        # ========== B. Handle unmatched blobs and new detections ===========
        unmatched_existing_blobs_indices = set(range(num_existing)) - set([r for r, c in matched_pairs])
        unmatched_new_heat_sources_indices = set(range(num_detected)) - set([c for r, c in matched_pairs])
        # print("DEBUG: unmatched:", unmatched_existing_blobs_indices, unmatched_new_heat_sources_indices)
        
        # B1. check for new heat sources that do not match existing blobs
        for source_idx in unmatched_new_heat_sources_indices:
            print("DEBUG: New heat source detected at index ", idx, " source index: ", source_idx)
            source = detected_heat_sources[source_idx]
            mask = source
            masked_temps = original_ira_img[mask.astype(bool)]
            avg_temp = masked_temps.mean()
            heatsource_size = np.sum(mask.astype(bool))
            # =========== Case 1: Noise or insignificant heat source ===========
            # print("DEBUG:avg temp too low ", avg_temp < background_avg + 3, "; heatsource size too small ", heatsource_size < SIZE_THRESH)
            if avg_temp < background_avg + 3 or heatsource_size < SIZE_THRESH:  # threshold to filter out noise
                continue

            # ============ Case 2: New blob detected ============
            new_blob = KalmanBlob(mask=mask, masked_temps=masked_temps)

            # ======== check if residual is generated =========
            print("mean temp of new blob: ", new_blob.mean_temp)
            residual_index, ori_index = self.residual_detector.get_residual_ori_index(self.blobs, new_blob) # residual index and original thermal blob index
            # update the relation if one patch split event is detected
            print("residual index: ", residual_index, " ori index: ", ori_index)
            if residual_index is not None and ori_index is not None:
                self.update_relations(new_blob, self.blobs[ori_index])
            print("residual index: ", residual_index)
            if residual_index is not None:
                print("Human left the bed! Residual heat detected. Frame index: ", idx)
                return_dict["bed_exit"] = True
                if residual_index == -1:
                    new_blob.is_residual = True
                    new_blob.id = -1  # mark as residual blob
                else:
                    self.blobs[residual_index].is_residual = True
                    new_blob.id = self.blobs[residual_index].id
                    self.blobs[residual_index].id = -1  # mark as residual blob

            self.blobs.append(new_blob)

        # B2. Update unmatched existing blobs as unobserved
        for blob_idx in unmatched_existing_blobs_indices:
            blob = self.blobs[blob_idx]
            blob.update(blob.get_mask(), blob.masked_temps, False)
        
        # B3. remove blobs that are lost for too long or moved outside frame
        for idx, blob in enumerate(self.blobs):
            if blob.is_residual != True:
                return_dict["human_in_scene"] = True
            if blob.outside_frame(original_ira_img.shape) or blob.time_since_observed > blob.age / 2 or blob.time_since_observed > 50:
                self.blobs.pop(idx)
                self.relations.pop(blob.id_fixed, None)  # remove from relations as well
                for key in self.relations:
                    self.relations[key].discard(blob.id_fixed)
        
        self.check_blob_type(background_avg)  # check if any blob is residual based on temp trend
        return return_dict
    
    def find_sibling(self, blob_id):
        for blob in self.blobs:
            if blob.id_fixed == blob_id:
                return blob
        return None
    def check_blob_type(self, background_temp):
        """
        Verify the blobs' types (residual/human) based on their temperature history and movement patterns.
        """
        print("=========================")
        TEMP_DECREASE_THRESH = -0.8  # threshold for temperature decrease trend
        for id, blob in enumerate(self.blobs):
            print("+++++++++++++++++++Blob ID: ", blob.id_fixed)
            blob.update_temp_trend() #update blob.temp_trend variable
            print("Blob ID: ", blob.id_fixed, " Temp: ", blob.temp_history, " Is residual: ", blob.is_residual)
            # if blob.temp_trend < TEMP_DECREASE_THRESH:
            #     blob.is_residual = True
            #     blob.id = -1  # mark as residual blob
            # model the temperature decreasing trend with a more robust method, considering the environmental temperature
            # temperature attenuation function: T(t) = T_env + (T_initial - T_env) * exp(-k*t), where k is the attenuation coefficient
            # we can estimate k from the temp history and determine if the blob is residual based on whether the estimated k is above a certain threshold
            if len(blob.temp_history) >= 5:
                sibling_ids = self.relations.get(blob.id_fixed, set())
                min_temp_history_len = len(blob.temp_history)
                max_temp_history_len = len(blob.temp_history)
                min_blob = blob
                max_blob = blob
                for sibling_id in sibling_ids:
                    sibling = self.find_sibling(sibling_id)
                    if sibling is None:
                        continue
                    if min_temp_history_len > len(sibling.temp_history):
                        min_temp_history_len = len(sibling.temp_history)
                        min_blob = sibling
                    if max_temp_history_len < len(sibling.temp_history):
                        max_temp_history_len = len(sibling.temp_history)
                        max_blob = sibling
                    
                start = max(0, max_temp_history_len - min_temp_history_len - 20)  # get the last 20 frames of temp history for fitting
                end = max_temp_history_len - min_temp_history_len
                seg1 = max_blob.temp_history[start:end]
                seg2 = blob.temp_history[:]
                print("seg1: ", seg1, " seg2: ", seg2)
                history = seg1
                history.extend(seg2)


                # k = -np.log(blob.temp_history[-1] / blob.temp_history[-5]) / 5
                # fit k according to previous temp history, using curve fitting
                # def temp_func(t, k):
                #     return background_temp + (blob.temp_history[0] - background_temp) * np.exp(-k * t)
                # try:
                #     t = np.arange(len(blob.temp_history))
                #     popt, pcov = curve_fit(temp_func, t, history, bounds=(0, 1))
                #     k = popt[0]
                #     blob.k = k
                # except:
                #     k = 0

                # use the history to directly compute the correlation between time and temperature, if the correlation is strongly negative, it indicates a decreasing trend
                # we want to detect the case where temperature is steadily and gradually decreasing, indicating heat residual
                # but we don't want to classify a blob with a sharp decline as residual, because it could be human being occluded by blanket
                if len(history) <= 5:
                    continue
                corr = np.corrcoef(np.arange(len(history)), history)[0, 1]
                print("Correlation between time and temp history: ", corr)
    
                # quantify if there exists a sharp decline in the temp history, which indicates a non-residual blob that is occluded by blanket
                # sharp decline is defined as a drop of more than 5 degrees within 10 frames
                for i in range(len(history)-10):
                    if history[i] - history[i+10] > 5:
                        corr = 0  # if there exists a sharp decline, we set k to 0 to avoid misclassifying occluded human as residual
                        print("Sharp decline detected in temp history, likely occluded human. Blob ID: ", blob.id_fixed)
                        break
                
                if corr < TEMP_DECREASE_THRESH:
                    blob.is_residual = True
                    blob.id = -1  # mark as residual blob

                
                velocity = blob.get_velocity()
                print("Blob ID: ", blob.id_fixed, " Temp history len: ", len(blob.temp_history), " Min temp history len: ", min_temp_history_len, " Sibling IDs: ", sibling_ids)
                # print("k value:", k, "k > 0.1:", k>0.1)
                print("Velocity: ", velocity)
                print("is residual: ", blob.is_residual)
                

                # if k > 0.1:
                #     blob.is_residual = True
                #     blob.id = -1  # mark as residual blob
        
        
                    

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
        
    def update(self, ira, detector):
        thresh, mask = detector.get_thresh_mask_otsu(ira)
        mask_processed = detector.process_frame_mask(ira, min_size=200)
        mask_individual = detector.process_frame_connected_components(ira, min_size=200)
        self.update_blobs(mask_individual, ira, detector.get_unmasked_mean(ira, mask))
        self.track_blob_relations()


class DataVisualizer:
    """Improved visualizer for thermal blob tracking"""
    
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.colors = {}  # persistent colors per blob.id_fixed
        self.cmap = cv2.COLORMAP_JET
        
    def get_blob_color(self, blob_id, is_residual=False):
        """Return consistent BGR color for each blob"""
        if blob_id not in self.colors:
            if is_residual:
                self.colors[blob_id] = (0, 100, 255)   # Orange-red for residual
            else:
                # Generate nice distinct colors for human blobs
                hue = (blob_id * 137) % 180  # golden angle for good distribution
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                self.colors[blob_id] = tuple(map(int, color_bgr))
        return self.colors[blob_id]

    def prepare_thermal_image(self, ira_img):
        """Convert raw thermal to nice JET colormap"""
        img = self._prepare_thermal_for_colormap(ira_img)
        return cv2.applyColorMap(img, self.cmap)

    def _prepare_thermal_for_colormap(self, img):
        # Normalize to 0-255
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img_norm.astype(np.uint8)

    def draw_blob_visualization(self, ira_highres, tracker, frame_idx=None):
        """
        Main visualization function - returns a nice annotated image
        """
        # 1. Prepare base images
        thermal_vis = self.prepare_thermal_image(ira_highres)
        height, width = thermal_vis.shape[:2]
        
        # Create a blank mask visualization (or use processed mask if available)
        mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 2. Draw trajectories and current blobs
        for blob in tracker.blobs:
            color = self.get_blob_color(blob.id_fixed, blob.is_residual)
            label_color = (255, 255, 255) if not blob.is_residual else (0, 255, 255)
            
            # --- Draw trajectory history ---
            if len(blob.kalman_centroid_history) > 1:
                points = np.array(blob.kalman_centroid_history[-self.max_history:], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(thermal_vis, [points * 1], False, color, thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(mask_vis, [points * 1], False, color, thickness=2, lineType=cv2.LINE_AA)
            
            # --- Current position and bounding box ---
            if blob.kalman_centroid_history:
                cx, cy = map(int, blob.kalman_centroid_history[-1])
                
                # Bounding box
                bbox = blob.get_bbox()  # assume this method exists
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    thickness = 2 if not blob.is_residual else 1
                    cv2.rectangle(thermal_vis, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), color, thickness)
                
                # Centroid
                radius = 6 if not blob.is_residual else 4
                cv2.circle(thermal_vis, (cx, cy), radius, color, -1)
                cv2.circle(mask_vis, (cx, cy), radius, color, -1)
                cv2.circle(thermal_vis, (cx, cy), radius+2, (255,255,255), 1)
                
                # Label with rich information
                status = "RESIDUAL" if blob.is_residual else "HUMAN"
                temp_str = f"{blob.mean_temp:.1f}°C"
                trend_str = f"trend: {blob.temp_trend:.2f}" if hasattr(blob, 'temp_trend') else ""
                
                label = f"ID{blob.id_fixed} {status} {temp_str}"
                if trend_str:
                    label += f" {trend_str}"
                
                font_scale = 0.55
                thickness = 2
                
                # Background for readability
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                cv2.rectangle(thermal_vis, 
                            (cx + 8, cy - 20), 
                            (cx + 8 + text_size[0], cy - 20 + text_size[1] + 6),
                            (0, 0, 0), -1)
                
                cv2.putText(thermal_vis, label, (cx + 10, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness, cv2.LINE_AA)
                
                # Also draw on mask view (smaller text)
                cv2.putText(mask_vis, f"ID{blob.id_fixed}", (cx + 5, cy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Add frame info
        if frame_idx is not None:
            info_text = f"Frame: {frame_idx} | Blobs: {len(tracker.blobs)} | Human: {any(not b.is_residual for b in tracker.blobs)}"
            cv2.putText(thermal_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Combine side by side
        combined = np.hstack([thermal_vis, mask_vis])
        
        # Optional: Add a small separator line
        cv2.line(combined, (width, 0), (width, height), (255, 255, 255), 2)
        
        return combined


# ==================== Updated Main Visualization Loop ====================

def plot_trajectory_on_img_improved():
    from dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    from data_visualization_module.plot import stitch_images  # if you still need it

    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    tracker = Tracker()
    visualizer = DataVisualizer(max_history=40)

    indices = range(18200, len(dataset))   # or your specific range

    for idx in indices:
        ira_highres = dataset.get_ira_highres(idx)
        
        # Detection
        mask_individual = detector.process_frame_connected_components(ira_highres, min_size=200)
        
        # Tracking update
        result = tracker.update_blobs(mask_individual, ira_highres, 
                                    detector.get_unmasked_mean(ira_highres, None), idx=idx)

        # === Visualization ===
        vis_image = visualizer.draw_blob_visualization(ira_highres, tracker, frame_idx=idx)

        cv2.imshow("Thermal Blob Tracking", vis_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):   # space = pause
            cv2.waitKey(0)
        elif key == ord('s'):   # save frame
            cv2.imwrite(f"frame_{idx:06d}.png", vis_image)

    cv2.destroyAllWindows()


# Optional: Enhanced trajectory plot (after tracking)
def plot_all_trajectories(tracker, ira_shape):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    for blob in tracker.blobs:
        if len(blob.kalman_centroid_history) < 3:
            continue
            
        centroids = np.array(blob.kalman_centroid_history)
        color = '#ff4444' if blob.is_residual else f'C{blob.id_fixed % 10}'
        
        ax.plot(centroids[:, 0], centroids[:, 1], 
                marker='o', markersize=3, linewidth=2,
                label=f'ID{blob.id_fixed} {"(Residual)" if blob.is_residual else ""}',
                color=color, alpha=0.9)
        
        # Mark start and end
        ax.plot(centroids[0, 0], centroids[0, 1], 'go', markersize=8, label='Start' if blob.id_fixed == 0 else "")
        ax.plot(centroids[-1, 0], centroids[-1, 1], 'rx', markersize=8)
    
    ax.invert_yaxis()
    ax.set_xlim(0, ira_shape[1])
    ax.set_ylim(0, ira_shape[0])
    ax.set_title("Blob Trajectories Over Time")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_trajectory_on_img_improved()
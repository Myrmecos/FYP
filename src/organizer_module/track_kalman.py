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
SIZE_THRESH = 80  # minimum size of heat source to be considered a blob
HUMAN_ENV_THRESH = 4 # temperature differences between human and env
TEMP_DECREASE_THRESH = -0.9 # temperature decrease correlation threshold
K_THRESH = 0.004 # THRESHOLD for attenuation coefficient. Exceeding the threshold indicates decreasing trend
HISTORY_THRESH = 100

class Tracker:
    def __init__(self, hung_thresh = HUNG_THRESH, size_thresh = SIZE_THRESH, human_env_thresh = HUMAN_ENV_THRESH, temp_decrease_thresh = TEMP_DECREASE_THRESH, k_thresh = K_THRESH, history_thresh = HISTORY_THRESH):
        self.blobs = []  # list of KalmanBlob objects
        self.residual_detector = ResidualHeatDetector()
        self.relations = dict() # adjacency list to track the relations between blobs

        self.HUNG_THRESH = hung_thresh
        self.SIZE_THRESH = size_thresh
        self.HUMAN_ENV_THRESH = human_env_thresh
        self.TEMP_DECREASE_THRESH = temp_decrease_thresh
        self.K_THRESH = k_thresh
        self.HISTORY_THRESHOLD = history_thresh
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
            centers_close = score_matrix[r, c] >= self.HUNG_THRESH  # blended metric threshold
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
            # print("DEBUG: New heat source detected at index ", idx, " source index: ", source_idx)
            source = detected_heat_sources[source_idx]
            mask = source
            masked_temps = original_ira_img[mask.astype(bool)]
            avg_temp = masked_temps.mean()
            heatsource_size = np.sum(mask.astype(bool))
            # =========== Case 1: Noise or insignificant heat source ===========
            # print("DEBUG:avg temp too low ", avg_temp < background_avg + 3, "; heatsource size too small ", heatsource_size < SIZE_THRESH)
            if avg_temp < background_avg + self.HUMAN_ENV_THRESH  or heatsource_size < self.SIZE_THRESH:  # threshold to filter out noise
                continue

            # ============ Case 2: New blob detected ============
            new_blob = KalmanBlob(mask=mask, masked_temps=masked_temps)

            # ======== check if residual is generated =========
            # print("mean temp of new blob: ", new_blob.mean_temp)
            residual_index, ori_index, residual_blob, human_blob = self.residual_detector.get_residual_ori_index(self.blobs, new_blob) # residual index and original thermal blob index
            
            
            # update the relation if one patch split event is detected
            # print("residual index: ", residual_index, " ori index: ", ori_index)
            if residual_index is not None and ori_index is not None:
                self.update_relations(new_blob, self.blobs[ori_index])

            if residual_index is not None:
                print("Human left the bed! Residual heat detected. Frame index: ", idx)
                return_dict["bed_exit"] = True
                if residual_index == -1:
                    new_blob.is_residual = True
                    new_blob.temp_history.extend(residual_blob.temp_history[-5:]) # inherit the temp history from the original blob
                    new_blob.id = -1  # mark as residual blob
                else:
                    self.blobs[residual_index].is_residual = True
                    new_blob.id = self.blobs[residual_index].id
                    new_blob.temp_history.extend(human_blob.temp_history[:]) # inherit the temp history from the original blob
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
        # print("=========================")
        # threshold for temperature decrease trend


        # ============================ MAIN LOGIC for determining if a blob is residual ============================
        for id, blob in enumerate(self.blobs):
            # print("+++++++++++++++++++Blob ID: ", blob.id_fixed)
            blob.update_temp_trend() #update blob.temp_trend variable
            # print("Blob ID: ", blob.id_fixed, " Temp: ", blob.temp_history, " Is residual: ", blob.is_residual)
            # if blob.temp_trend < TEMP_DECREASE_THRESH:
            #     blob.is_residual = True
            #     blob.id = -1  # mark as residual blob
            # model the temperature decreasing trend with a more robust method, considering the environmental temperature
            # temperature attenuation function: T(t) = T_env + (T_initial - T_env) * exp(-k*t), where k is the attenuation coefficient
            # we can estimate k from the temp history and determine if the blob is residual based on whether the estimated k is above a certain threshold
            if len(blob.temp_history) >= self.HISTORY_THRESHOLD//2: # 3 seconds
                # print("seg1: ", seg1, " seg2: ", seg2)
                history = blob.temp_history
                history = history[-self.HISTORY_THRESHOLD:]  # only keep the last HISTORY_THRESHOLD frames of history to capture the recent trend, which is more relevant for determining if it's residual
                # otherwise we risk classifying residual as a human (longer history, more coverage, k is not captured correctly)
                
                # calculate correlation between time and temperature history
                len_history = min(self.HISTORY_THRESHOLD, len(history))  
                corr = np.corrcoef(range(len_history), history[-len_history:])[0, 1]
                blob.corr = corr

                # quantify if there exists a sharp decline in the temp history, which indicates a non-residual blob that is occluded by blanket
                # sharp decline is defined as a drop of more than 5 degrees within 10 frames
                for i in range(len(history)-20):
                    if history[i] - history[i+20] > 3:
                        corr = 0  # if there exists a sharp decline, we set k to 0 to avoid misclassifying occluded human as residual
                        blob.k = 0  # if there exists a sharp decline, we set k to 0 to avoid misclassifying occluded human as residual
                        # print("Sharp decline detected in temp history, likely occluded human. Blob ID: ", blob.id_fixed)
                        break
                
                # check blob velocity and shape changes
                velocity = blob.get_velocity()
                if velocity > 1:# or blob.get_bbox_change_velocity() > 5:  # if the blob is moving fast or changing shape rapidly, it is likely a human even if the temp trend is decreasing
                    blob.is_residual = False
                    blob.id = blob.id_fixed  # restore original id if it was marked as residual before
                    # print("Blob is moving fast or changing shape rapidly, likely human. Blob ID: ", blob.id_fixed)
                    continue
                    
                # corr and temp difference between current temp and background temp can jointly determine if the blob is residual
                # we know that corr threshold should be large if temp diff is large, 
                # if corr < TEMP_DECREASE_THRESH:
                # the relations between temp and environmental temp is: T(t) = T_env + (T_initial - T_env) * exp(-k*t)
                t_initial = history[0]
                t_final = history[-1]
                t_len = len(history)
                t_env = background_temp
                k = -np.log((t_final - t_env) / (t_initial - t_env)) / t_len
                blob.k = k
                
                if corr < self.TEMP_DECREASE_THRESH and k > self.K_THRESH:
                    # print("classified as residual due to decreasing temp trend. Blob ID: ", blob.id_fixed)
                    blob.is_residual = True
                    blob.id = -1  # mark as residual blob
                # else:
                #     print("Blob ID: ", blob.id_fixed, " Temp trend corr: ", corr, " Attenuation coeff k: ", k, " Not classified as residual.")

                
                velocity = blob.get_velocity()

        
        
                    

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
        mask_processed = detector.process_frame_mask(ira, min_size=self.SIZE_THRESH)
        mask_individual = detector.process_frame_connected_components(ira, min_size=self.SIZE_THRESH)
        self.update_blobs(mask_individual, ira, detector.get_unmasked_mean(ira, mask))
        self.track_blob_relations()

# use data from hall1, frame 18055-18115 for testing
if __name__ == "__main__":
    def plot_trajectory_on_img():
        from dataset import ThermalDataset
        from heatsource_detection_module.extract import HeatSourceDetector
        from data_visualization_module.plot import DataVisualizer, stitch_images
        datavisualizer = DataVisualizer()
        dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/home31")
        detector = HeatSourceDetector()
        tracker = Tracker()

        indices = range(18230, 18265)
        indices = range(0, len(dataset), 10) # test on all frames in the dataset

        for idx in indices: #18115
            ira_highres = dataset.get_ira_highres(idx)
            thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
            mask_processed = detector.process_frame_mask(ira_highres, min_size=tracker.SIZE_THRESH)
            mask_individual = detector.process_frame_connected_components(ira_highres, min_size=SIZE_THRESH)
            # cleaned_mask = detector.get_connected_components(mask, min_size=SIZE_THRESH)
            print(len(mask_individual), "heat sources detected in frame", idx)
            tracker.update_blobs(mask_individual, ira_highres, detector.get_unmasked_mean(ira_highres, mask))

            # visualize the blobs and their centroids on the original IRA high-res image and the mask
            ira_highres_color = datavisualizer._prepare_thermal_for_colormap(ira_highres)
            ira_highres_color = cv2.applyColorMap(ira_highres_color, cv2.COLORMAP_JET)
            map_color = datavisualizer._prepare_thermal_for_colormap(mask_processed.astype('uint8') * 255)
            map_color = cv2.applyColorMap(map_color, cv2.COLORMAP_JET)
            scaling_factor = 10
            ira_highres_color = cv2.resize(ira_highres_color, (0, 0), fx=scaling_factor, fy=scaling_factor)
            map_color = cv2.resize(map_color, (0, 0), fx=scaling_factor, fy=scaling_factor)
            # plot the centroid of each blob on the IRA high-res image
            for i, blob in enumerate(tracker.blobs):
                if not blob.kalman_centroid_history:  # check if history is not empty
                    continue
                idx = blob.id_fixed
                centroid = blob.kalman_centroid_history[-1]  # get the latest centroid
                cv2.circle(ira_highres_color, (int(centroid[0] * scaling_factor), int(centroid[1] * scaling_factor)), 5, (0, 0, 0), -1)  # draw centroid on image with radiu 5
                cv2.putText(ira_highres_color, f'Blob {idx}: is residual: {blob.is_residual}', (int(centroid[0] * scaling_factor) -5, int(centroid[1] * scaling_factor) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(map_color, (int(centroid[0] * scaling_factor), int(centroid[1] * scaling_factor)), 5, (0, 0, 0), -1)  # also plot on mask color map for better visibility
                cv2.putText(map_color, f'Blob {idx}: {round(blob.temp_trend, 2)}', (int(centroid[0] * scaling_factor) + 5, int(centroid[1] * scaling_factor) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # cv2.imshow("IRA High-Res with Blob Centroids", ira_highres_color)
            # cv2.waitKey(0) 
            # stich the original IRA high-res and the color map of detected blobs side by side for better visualization
            print("number of blobs found: ", len(tracker.blobs))
            combined_vis = stitch_images([ira_highres_color, map_color])
            cv2.imshow("Combined Visualization", combined_vis)
            cv2.waitKey(0)
            
    plot_trajectory_on_img()

    def plot_trajectory():
        from dataset import ThermalDataset
        from heatsource_detection_module.extract import HeatSourceDetector
        dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
        detector = HeatSourceDetector()
        tracker = Tracker()

        rng = range(18240, 18260) #18115

        for idx in rng: #18115
            ira_highres = dataset.get_ira_highres(idx)
            thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
            cleaned_mask = detector.get_connected_components(mask, min_size=self.SIZE_THRESH)
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
    
        


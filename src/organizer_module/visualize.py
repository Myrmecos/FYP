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
from organizer_module.track_kalman import Tracker
from dataset import ThermalDataset
from heatsource_detection_module.extract import HeatSourceDetector
from postprocessor import PostProcessor


if __name__ == "__main__":
    def plot_trajectory_on_img():
        from dataset import ThermalDataset
        from heatsource_detection_module.extract import HeatSourceDetector
        from data_visualization_module.plot import DataVisualizer, stitch_images

        data_path = "/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall2"
        dataset = ThermalDataset(data_path)
        detector = HeatSourceDetector()
        tracker = Tracker()
        visualizer = DataVisualizer()                    # your original DataVisualizer

        # indices = range(18230, 18265)                  # small test range
        indices = range(18200, len(dataset))             # full range
        indices = range(14392, len(dataset))
        indices = range(14392, 14452)

        # # save results to mp4 file
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # shape_ira = dataset.get_ira_highres(0).shape
        # shape_rescaled = (shape_ira[1] * 10, shape_ira[0] * 10)  # (width, height) after upscaling
        # shape_concatenated = (shape_rescaled[0] * 2, shape_rescaled[1])  # side-by-side concatenation
        # out = cv2.VideoWriter(f'{data_path}/tracking_result.mp4', fourcc, 30.0, shape_concatenated)
        # initialize video writer with correct frame size (after upscaling)
        
        postprocessor = PostProcessor()
        for idx in indices:
            ira_highres = dataset.get_ira_highres(idx)

            # === Detection ===
            thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
            mask_processed = detector.process_frame_mask(ira_highres, min_size=200)
            mask_individual = detector.process_frame_connected_components(ira_highres, min_size=200)

            print(f"Frame {idx:5d} | Heat sources detected: {len(mask_individual)}")

            # === Tracking update ===
            background_avg = detector.get_unmasked_mean(ira_highres, mask)
            tracker.update_blobs(mask_individual, ira_highres, background_avg)

            # postprocessor update ===
            postprocessor.get_blobs(tracker.blobs, idx)

            # === Visualization Preparation ===
            # Original thermal image with JET colormap
            ira_highres_color = visualizer._prepare_thermal_for_colormap(ira_highres)
            ira_highres_color = cv2.applyColorMap(ira_highres_color, cv2.COLORMAP_JET)

            # Processed mask visualization
            mask_color = visualizer._prepare_thermal_for_colormap(mask_processed.astype('uint8') * 255)
            mask_color = cv2.applyColorMap(mask_color, cv2.COLORMAP_JET)

            # Upscale for better visibility (10x)
            scaling_factor = 10
            ira_highres_color = cv2.resize(ira_highres_color, (0, 0), fx=scaling_factor, fy=scaling_factor)
            mask_color = cv2.resize(mask_color, (0, 0), fx=scaling_factor, fy=scaling_factor)

            # === Draw blobs, trajectories, and info ===
            for blob in tracker.blobs:
                if not blob.kalman_centroid_history:
                    continue

                centroid = blob.kalman_centroid_history[-1]
                cx = int(centroid[0] * scaling_factor)
                cy = int(centroid[1] * scaling_factor)

                color = (0, 0, 255) if blob.is_residual else (0, 255, 0)   # Red = residual, Green = human

                # Draw centroid on both images
                cv2.circle(ira_highres_color, (cx, cy), 7, color, -1)
                cv2.circle(ira_highres_color, (cx, cy), 9, (255, 255, 255), 2)
                cv2.circle(mask_color, (cx, cy), 7, color, -1)

                # draw rectangles on both images
                x_min, y_min, x_max, y_max = blob.get_state()
                x_min = int(x_min * scaling_factor)
                y_min = int(y_min * scaling_factor)
                x_max = int(x_max * scaling_factor)
                y_max = int(y_max * scaling_factor)
                cv2.rectangle(ira_highres_color, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.rectangle(mask_color, (x_min, y_min), (x_max, y_max), color, 2)

                # Labels
                label_thermal = f"ID{blob.id_fixed} | {'RESIDUAL' if blob.is_residual else 'HUMAN'}"
                label_mask = f"ID{blob.id_fixed} | {blob.corr:.2f}" if hasattr(blob, 'temp_trend') else f"ID{blob.id_fixed}"

                cv2.putText(ira_highres_color, label_thermal,
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(mask_color, label_mask,
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Frame information overlay
            human_in_scene = any(not b.is_residual for b in tracker.blobs)
            info_text = f"Frame: {idx} | Blobs: {len(tracker.blobs)} | Human: {'Yes' if human_in_scene else 'No'}"
            cv2.putText(ira_highres_color, info_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            print(f"Number of blobs found: {len(tracker.blobs)}")

            # Combine both views side-by-side
            combined_vis = stitch_images([ira_highres_color, mask_color])

            cv2.imshow("Thermal Blob Tracking Visualization", combined_vis)
            key = cv2.waitKey(1) & 0xFF   # Press any key to go to next frame
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"vis_frame_{idx:06d}.png", combined_vis)
                print(f"Saved: vis_frame_{idx:06d}.png")
            
            # out.write(combined_vis)
        # out.release()
        print("Visualization video saved as 'tracking_result.mp4'")

        # Output blob records
        postprocessor.output_results(f"{data_path}/blob_records.json")

    plot_trajectory_on_img()


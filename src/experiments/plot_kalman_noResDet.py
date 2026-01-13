import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from kalman_blob_noResDet import KalmanBlob, mask_to_bbox
from scipy.optimize import linear_sum_assignment
from data_collection_module import utils

from track_kalman_noResDet import Tracker

# use data from hall1, frame 18055-18115 for testing
if __name__ == "__main__":
    from dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    tracker = Tracker()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    IRA_height, IRA_width = dataset.get_ira_highres(0).shape
    out = cv2.VideoWriter('kalman_noResDet.mp4', fourcc, 30.0, (IRA_width*5, IRA_height*5))

    detection_result = []

    for idx in range(300, 20712): #18115
        ira_highres = dataset.get_ira_highres(idx)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        cleaned_mask = detector.get_connected_components(mask, min_size=10)

        detection_result.append(tracker.update_blobs(cleaned_mask, ira_highres, detector.get_unmasked_mean(ira_highres, mask), idx))

        ira_color = utils.colorize_thermal_map(ira_highres)
        ira_color = cv2.resize(ira_color, (IRA_width*5, IRA_height*5), interpolation=cv2.INTER_NEAREST)
        for i, blob in enumerate(tracker.blobs):
            if len(blob.kalman_centroid_history) == 0:
                continue
            cX, cY = blob.kalman_centroid_history[-1]
            cv2.putText(ira_color, str(blob.id), (int(cX)*5, int(cY)*5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        out.write(ira_color)
    out.release()

    # write detection_result to a pkl file
    import pickle as pkl
    with open("kalman_detection_result_noResDet.pkl", "wb") as f:
        pkl.dump(detection_result, f)

    # # plot the blobs' centroids movements

    # # for i, blob in enumerate(tracker.blobs):
    # #     centroids = blob.kalman_centroid_history
    # #     xs = [c[0] for c in centroids]
    # #     ys = [c[1] for c in centroids]
    # #     plt.plot(xs, ys, marker='o', label=f'Blob {i}')
        
    # # plot the blobs' kalman centroid in original IRA images 
    # # the centroids are signified as the index of the blob
    # # save them in a video kalman.mp4
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # IRA_height, IRA_width = ira_highres.shape
    # out = cv2.VideoWriter('kalman.mp4', fourcc, 10.0, (IRA_width, IRA_height))
    # for idx in range(18000, 18260):
    #     ira_highres = dataset.get_ira_highres(18260)
    #     # convert ira to color image for better visualization
    #     ira_color = utils.colorize_thermal_map(ira_highres)

    #     for i, blob in enumerate(tracker.blobs):
    #         cX, cY = blob.kalman_centroid_history[idx-18000]
    #         cv2.putText(ira_color, str(i), (int(cX), int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     plt.imshow(ira_color)
    #     plt.show()
    #     out.write(ira_color)
    # out.release()

    
    
        


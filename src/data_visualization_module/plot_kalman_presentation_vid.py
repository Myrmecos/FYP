import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from organizer_module.kalman_blob import KalmanBlob, mask_to_bbox
from scipy.optimize import linear_sum_assignment
from data_collection_module import utils



from tracking_module.track_kalman import Tracker

SRC_PATH = "/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1"
DEST_VID = "kalman_demo_clean.mp4"
DEST_PKL = "kalman_detection_result_demo.pkl"

# use data from hall1, frame 18055-18115 for testing
if __name__ == "__main__":
    from dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    dataset = ThermalDataset(SRC_PATH)
    detector = HeatSourceDetector()
    tracker = Tracker()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    IRA_height, IRA_width = dataset.get_ira_highres(0).shape
    SCALE = 10
    out = cv2.VideoWriter(DEST_VID, fourcc, 30.0, (IRA_width*SCALE, IRA_height*SCALE))

    detection_result = []

    # import time
    # start_time = time.time()

    countdown = 0
    presence = False

    for idx in range(450, 1450): #18115
        ira_highres = dataset.get_ira_highres(idx)
        # flip ira 180 deg
        ira_highres = np.rot90(ira_highres, 2)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        cleaned_mask = detector.get_connected_components(mask, min_size=10)

        dres = tracker.update_blobs(cleaned_mask, ira_highres, detector.get_unmasked_mean(ira_highres, mask), idx)
        if dres['bed_exit']:
            countdown = 30  # show for 30 frames
        if dres['human_in_scene']:
            presence = True
        else:
            presence = False
        ira_color = utils.colorize_thermal_map_thres(ira_highres, 10, 25)
        ira_color = cv2.resize(ira_color, (IRA_width*SCALE, IRA_height*SCALE), interpolation=cv2.INTER_NEAREST)
        
        for i, blob in enumerate(tracker.blobs):
            if len(blob.kalman_centroid_history) == 0:
                continue
            cX, cY = blob.kalman_centroid_history[-1]
            mark_str = "subject" if blob.id != -1 else "heat residual"
            # put the mark_str to the left of the centroid
            # draw a white bbox around the target blob
            bbox = blob.get_state()
            cv2.rectangle(ira_color, (int(bbox[0]*SCALE), int(bbox[1]*SCALE)), (int(bbox[2]*SCALE), int(bbox[3]*SCALE)), (255, 255, 255), 2)
            cv2.putText(ira_color, mark_str, (int(cX)*SCALE - 50, int(cY)*SCALE - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        if presence:
            cv2.putText(ira_color, "Subject Present", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            cv2.putText(ira_color, "No Subject", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        if countdown > 0:
            # put "Bed Exit Detected!" on the top-left corner
            cv2.putText(ira_color, "Bed Exit Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            countdown -= 1
        out.write(ira_color)
    out.release()

    # end_time = time.time()
    # print(f"Inference speed: {(20712 - 300)/(end_time - start_time)} frames per second")

    # write detection_result to a pkl file
    import pickle as pkl
    with open(DEST_PKL, "wb") as f:
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

    
    
        


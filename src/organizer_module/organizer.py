# responsible for organizing results from detection and tracking modules
# It generates new Blob if new, detached heat blobs are detected
# It loops through existing Blobs to check for temperatures and presence status

IRA_width = 80
IRA_height = 62
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from presence_detection_workspace.src.tracking_module.track_no_filter import Tracker
from heatsource_detection_module.extract import HeatSourceDetector
from dataset import ThermalDataset

import matplotlib.pyplot as plt
import numpy as np

class Organizer: 
    def __init__(self, dataset, detector, tracker):
        self.dataset = dataset
        self.detector = detector
        self.tracker = tracker
    
    # process one frame at index idx
    def process_frame(self, idx):
        ira_highres = self.dataset.get_ira_highres(idx)
        thresh, mask = self.detector.get_thresh_mask_otsu(ira_highres)
        cleaned_mask = self.detector.get_connected_components(mask, min_size=100)
        self.tracker.update_blobs(cleaned_mask, ira_highres, self.detector.get_unmasked_mean(ira_highres, mask))
    
    # plot blob's centroid movements
    def _plot_blob_movements(self):

        for i, blob in enumerate(self.tracker.blobs):
            centroids = blob.centroid_history
            centroid_len = len(centroids)
            xs = [c[0] for c in centroids]
            ys = [c[1] for c in centroids]
            # let the color of points change from light to dark
            colors = plt.cm.viridis(np.linspace(0, 1, centroid_len))
            
            # Plot line segments with gradient colors
            for j in range(len(xs) - 1):
                plt.plot(xs[j:j+2], ys[j:j+2], color=colors[j], linewidth=2)
            
            # Plot points with gradient colors
            plt.scatter(xs, ys, c=colors, s=50, zorder=5, label=f'Blob {i}')

        plt.gca().invert_yaxis()  # invert y axis to match image coordinates
        plt.title("Blob Centroid Movements")
        plt.xlim(0, IRA_width)
        plt.ylim(0, IRA_height)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

        
    
if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1")
    detector = HeatSourceDetector()
    tracker = Tracker()
    organizer = Organizer(dataset, detector, tracker)

    for idx in range(18055, 18343):
        organizer.process_frame(idx)
    
    organizer._plot_blob_movements()
        

        


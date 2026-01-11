import os
import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import ThermalDataset

# TODO: implementa a class for detecting heat source
# The scene can contain human
# we first use adaptive thresholding to find human (TODO: explore the histogram of the entire image)

import numpy as np

class HeatSourceDetector:
    def __init__(self):
        pass

    # returns threshold and a binary mask
    def get_thresh_mask_otsu(self, ira_img):
        # dynamic threshold obtained via Gaussian
        ira_uint8 = ira_img.astype('uint8')
        thresh, mask = cv2.threshold(ira_uint8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh, mask
    
    def get_thresh_mask_otsu_gaussian(self, ira_img):
        # apply Gaussian blur to smooth the image
        ira_uint8 = ira_img.astype('uint8')
        ira_blurred = cv2.GaussianBlur(ira_uint8, (5, 5), 0)
        
        thresh, mask = cv2.threshold(ira_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh, mask
    def get_masked_values(self, ira_img, mask):
        # apply mask to the image
        ira_flat = ira_img.flatten()
        # print("number of total pixels: ", len(ira_flat))
        mask_flat = mask.flatten().astype(bool)
        masked_values = ira_flat[mask_flat]
        # print("number of masked values: ", len(masked_values))
        return masked_values
    
    def get_masked_median(self, ira_img, mask):
        masked_values = self.get_masked_values(ira_img, mask)
        median_val = np.median(masked_values)
        return median_val
    
    def get_masked_mean(self, ira_img, mask):
        masked_values = self.get_masked_values(ira_img, mask)
        mean_val = np.mean(masked_values)
        return mean_val


if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0")
    idx = 510
    ira_highres = dataset.get_ira_highres(idx)
    detector = HeatSourceDetector()
    thresh, mask = detector.get_thresh_mask_otsu_gaussian(ira_highres)
    plt.imshow(mask)
    plt.show()
    print(thresh)
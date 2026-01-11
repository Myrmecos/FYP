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
    
    def get_unmasked_mean(self, ira_img, mask):
        unmasked_values = ira_img[~mask.astype(bool)]
        mean_val = np.mean(unmasked_values)
        return mean_val
    
    def remove_small_regions(self, mask, min_size):
        # remove small connected components in the binary mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 1
        return cleaned_mask
    
    def get_connected_components(self, mask, clean=True, min_size=400):
        if clean:
            mask = self.remove_small_regions(mask, min_size)
        # return list of connected component masks
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
        component_masks = []
        for i in range(1, num_labels):  # skip background
            component_mask = np.zeros_like(mask)
            component_mask[labels == i] = 1
            component_masks.append(component_mask)
        return component_masks

    def get_centroid(self, mask):
        # compute centroid of the binary mask
        moments = cv2.moments(mask.astype('uint8'))
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
        else:
            cX, cY = -1, -1  # no centroid found
        return (cX, cY)
    
    def get_centroid_per_blob(self, mask):
        # compute centroids of all connected components in the binary mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
        centroid_list = []
        for i in range(1, num_labels):  # skip background
            centroid_list.append((int(centroids[i][0]), int(centroids[i][1])))
        return centroid_list


if __name__ == "__main__":
    dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0")
    idx = 510
    ira_highres = dataset.get_ira_highres(idx)
    detector = HeatSourceDetector()
    thresh, mask = detector.get_thresh_mask_otsu_gaussian(ira_highres)
    plt.imshow(mask)
    plt.show()
    print(thresh)
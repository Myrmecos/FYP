import os
import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.dataset import ThermalDataset, ThermalDatasetAggregator


# TODO: implementa a class for detecting heat source
# The scene can contain human
# we first use adaptive thresholding to find human (TODO: explore the histogram of the entire image)

import numpy as np

class HeatSourceDetector:
    def __init__(self):
        pass

    # given an ira image, return a binary mask of the heat source
    # utilize all methods in this class to find the best mask
    def process_frame_mask(self, ira_img, min_size=10):
        thresh, mask = self.get_thresh_mask_otsu_gaussian(ira_img)
        eroded_mask = self.erode_mask(mask)
        cleaned_mask = self.remove_small_regions(eroded_mask, min_size)
        return cleaned_mask
    
    # given an ira image, return a list of binary masks of the heat source blobs
    # utilize all methods in this class to find the best mask
    def process_frame_connected_components(self, ira_img, min_size=400):
        cleaned_mask = self.process_frame_mask(ira_img, min_size)
        component_masks = self.get_connected_components(cleaned_mask, min_size=min_size)
        return component_masks

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
        self.erode_mask(mask)

        return thresh, mask
    
    def erode_mask(self, mask, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask.astype('uint8'), kernel, iterations=iterations)
        return eroded_mask
    
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
    from tqdm import tqdm

    def visualize():
        from data_visualization_module.plot import DataVisualizer
        visualizer = DataVisualizer()
        dataset_base_dirs = [
            "entry_exit_detection/presence_detection_workspace/data/office0_1",
            "entry_exit_detection/presence_detection_workspace/data/office0_2"
        ]
        dataset = ThermalDatasetAggregator(dataset_base_dirs)
        detector = HeatSourceDetector()


        for i in range(1, dataset.subset_number()):
            print(f"Dataset {i} length: {dataset.subset_len(i)}")
            for j in range(648, 648+2600, 20):
                img = dataset.get_image_ij(i, j)
                ira = dataset.get_ira_ij(i, j)
                ira_highres = dataset.get_ira_highres_ij(i, j)
                tof = dataset.get_tof_ij(i, j)

                # show the mask
                # thresh, mask = detector.get_thresh_mask_otsu_gaussian(ira_highres)
                mask = detector.process_frame_mask(ira_highres, min_size=100)
                mask = mask.astype('uint8') * 255

                color_thermal_mask = visualizer.compose_color_and_thermal(img, ira_highres, mask)
                cv2.imshow("Color and Thermal", color_thermal_mask)
                cv2.waitKey(0)

    visualize()
    
    def visualize_to_vid():
        # visualize data to an mp4 video
        from data_visualization_module.plot import DataVisualizer
        visualizer = DataVisualizer()
        dataset_base_dirs = [
            "entry_exit_detection/presence_detection_workspace/data/office0_1",
            "entry_exit_detection/presence_detection_workspace/data/office0_2"
        ]
        dataset = ThermalDatasetAggregator(dataset_base_dirs)
        detector = HeatSourceDetector()
        
        # get image shape
        img = dataset.get_image_ij(0, 0)
        ira = dataset.get_ira_ij(0, 0)
        ira_highres = dataset.get_ira_highres_ij(0, 0)
        tof = dataset.get_tof_ij(0, 0)
        thresh, mask = detector.get_thresh_mask_otsu_gaussian(ira_highres)
        color_thermal_mask = visualizer.compose_color_and_thermal(img, ira_highres, mask)
        shape = color_thermal_mask.shape[1], color_thermal_mask.shape[0]  # width, height

        # initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter('heat_source_detection.mp4', fourcc, 30, shape)  # adjust frame size as needed


        for i in range(1, dataset.subset_number()):
            print(f"Dataset {i} length: {dataset.subset_len(i)}")
            idx = 0
            for j in tqdm(range(dataset.subset_len(i))):
                img = dataset.get_image_ij(i, j)
                ira = dataset.get_ira_ij(i, j)
                ira_highres = dataset.get_ira_highres_ij(i, j)
                tof = dataset.get_tof_ij(i, j)

                # show the mask
                # thresh, mask = detector.get_thresh_mask_otsu_gaussian(ira_highres)
                mask = detector.process_frame_mask(ira_highres, min_size=100)
                mask = mask.astype('uint8') * 255 # after processing, mask is binary 0/1, convert to 0/255 for visualization

                color_thermal_mask = visualizer.compose_color_and_thermal(img, ira_highres, mask)

                # write idx on the frame
                cv2.putText(color_thermal_mask, str(idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(color_thermal_mask)
                idx += 1

        out.release()

    # visualize_to_vid()
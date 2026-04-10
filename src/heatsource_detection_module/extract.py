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

# ==============================================================================
# Thermal image preprocessing utilities for cross-environment generalization
# ==============================================================================

def clip_temperature(ira_img, min_val=10.0, max_val=40.0):
    """Clip raw thermal values to the valid temperature range [min_val, max_val]."""
    return np.clip(ira_img, min_val, max_val)


def normalize_temperature(ira_img, min_val=10.0, max_val=40.0):
    """Min-max normalize thermal values to [0, 1] using the known temperature range.

    Args:
        ira_img: raw IRA thermal array (Celsius)
        min_val: lower bound of valid temperature range (default 10.0°C)
        max_val: upper bound of valid temperature range (default 40.0°C)

    Returns:
        normalized array in [0, 1]
    """
    clipped = clip_temperature(ira_img, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)


def compute_thermal_features(ira_img, min_blob_size=40):
    """Extract temperature-invariant features from a thermal image.

    Returns a dict of ~21 features designed for cross-environment generalization.
    All features are relative/ratio-based and invariant to absolute temperature shifts.
    """
    flat = ira_img.flatten().astype(np.float64)
    h, w = ira_img.shape

    # ---- 1. Statistical (temperature-invariant) --------------------------------
    sorted_flat = np.sort(flat)
    p10, p25, p50, p75, p90 = np.percentile(sorted_flat, [10, 25, 50, 75, 90])
    iqr = p75 - p25
    range_val = sorted_flat[-1] - sorted_flat[0] + 1e-8

    normalized_spread = iqr / range_val
    cv = np.std(flat) / (np.mean(flat) + 1e-8)

    from scipy.stats import skew, kurtosis
    skewness = float(skew(flat))
    kurt = float(kurtosis(flat))

    # ---- 2. Histogram (16 bins, normalized) ------------------------------------
    norm_img = (ira_img - sorted_flat[0]) / (range_val + 1e-8)
    hist, _ = np.histogram(norm_img.flatten(), bins=16, range=(0, 1), density=True)
    hist = hist.astype(np.float32)
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    modal_ratio = float(np.max(hist))

    # Bimodality coefficient
    n = len(flat)
    bc_numerator = skewness ** 2 + 1
    bc_denom = (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-8)) + 1e-8
    bimodality_coef = bc_numerator / bc_denom

    # ---- 3. Spatial / heat distribution ----------------------------------------
    # Normalized centroid of the full thermal image (weighted by intensity)
    total = np.sum(flat) + 1e-8
    cx_float = np.sum(flat * np.tile(np.arange(w), h) / total)
    cy_float = np.sum(flat * np.repeat(np.arange(h), w) / total)
    centroid_norm_x = cx_float / w
    centroid_norm_y = cy_float / h

    # Hottest point normalized location
    hot_idx = np.unravel_index(np.argmax(ira_img), ira_img.shape)
    hotspot_norm_x = hot_idx[1] / w
    hotspot_norm_y = hot_idx[0] / h

    # ---- 4. Blob features (using HeatSourceDetector) ----------------------------
    detector = HeatSourceDetector()
    try:
        blobs = detector.process_frame_connected_components(ira_img, min_size=min_blob_size)
    except Exception:
        blobs = []

    blob_count = len(blobs)
    total_area = h * w

    if blobs:
        # Sort by area descending
        areas = [b.sum() for b in blobs]
        order = np.argsort(areas)[::-1]
        largest_mask = blobs[order[0]]

        # Bounding box of largest blob
        y_coords, x_coords = np.where(largest_mask > 0)
        min_x, max_x = int(x_coords.min()), int(x_coords.max())
        min_y, max_y = int(y_coords.min()), int(y_coords.max())
        bbox_h = max_y - min_y + 1
        bbox_w = max_x - min_x + 1
        bbox_area = largest_mask.sum()

        largest_bbox_aspect_ratio = bbox_w / (bbox_h + 1e-8)
        # Contour of largest blob
        _, contours, _ = cv2.findContours(largest_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(cv2.arcLength(contours[0], True))
        largest_compactness = 4 * np.pi * bbox_area / ((perimeter ** 2) + 1e-8)
        blob_area_ratio = bbox_area / total_area

        # Spatial spread (covariance of blob pixels)
        if len(x_coords) > 5:
            cov = np.cov(x_coords, y_coords)
            eigenvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            spread_aspect_ratio = eigenvals[0] / (eigenvals[1] + 1e-8)
            heat_spread_norm = np.sqrt(eigenvals[0] + eigenvals[1]) / np.sqrt(h * w)
        else:
            spread_aspect_ratio = 0.0
            heat_spread_norm = 0.0
    else:
        largest_bbox_aspect_ratio = 0.0
        largest_compactness = 0.0
        blob_area_ratio = 0.0
        spread_aspect_ratio = 0.0
        heat_spread_norm = 0.0

    # ---- 5. Texture features (LBP histogram) ------------------------------------
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(norm_img, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
    except Exception:
        lbp_entropy = 0.0

    # ---- 6. Thermal gradient features -------------------------------------------
    sobel_x = cv2.Sobel(ira_img.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(ira_img.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_mag_normalized = grad_mag / (range_val + 1e-8)
    grad_mean = float(np.mean(grad_mag_normalized))
    grad_max = float(np.max(grad_mag_normalized))

    return {
        # Statistical
        'normalized_spread': normalized_spread,
        'cv': cv,
        'skewness': skewness,
        'kurtosis': kurt,
        'entropy': entropy,
        # Histogram
        'modal_ratio': modal_ratio,
        'bimodality_coef': bimodality_coef,
        # Spatial
        'centroid_norm_x': centroid_norm_x,
        'centroid_norm_y': centroid_norm_y,
        'hotspot_norm_x': hotspot_norm_x,
        'hotspot_norm_y': hotspot_norm_y,
        # Blob
        'blob_count': float(blob_count),
        'blob_area_ratio': blob_area_ratio,
        'largest_bbox_aspect_ratio': largest_bbox_aspect_ratio,
        'largest_compactness': largest_compactness,
        'spread_aspect_ratio': spread_aspect_ratio,
        'heat_spread_norm': heat_spread_norm,
        # Texture
        'lbp_entropy': lbp_entropy,
        # Gradient
        'grad_mean': grad_mean,
        'grad_max': grad_max,
    }


class HeatSourceDetector:
    def __init__(self):
        pass

    # given an ira image, return a binary mask of the heat source
    # utilize all methods in this class to find the best mask
    def process_frame_mask(self, ira_img, min_size=100):
        thresh, mask = self.get_thresh_mask_otsu(ira_img)
        eroded_mask = self.erode_mask(mask)
        cleaned_mask = self.remove_small_regions(eroded_mask, min_size)
        return cleaned_mask
    
    # given an ira image, return a list of binary masks of the heat source blobs
    # utilize all methods in this class to find the best mask
    def process_frame_connected_components(self, ira_img, min_size=100):
        cleaned_mask = self.process_frame_mask(ira_img, min_size)
        component_masks = self.get_connected_components(cleaned_mask, min_size=min_size)
        return component_masks

    # returns threshold and a binary mask
    def get_thresh_mask_otsu(self, ira_img):
        # dynamic threshold obtained via Gaussian
        ira_uint8 = ira_img.astype('uint8')
        thresh, mask = cv2.threshold(ira_uint8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # get all pixel values in ira_img that are above the threshold
        pixels_above_thresh = ira_img[ira_img >= thresh]
        if len(pixels_above_thresh) == 0:
            return thresh, mask
        # remove lower 25 percentile of the pixels above the threshold to reduce noise
        lower_percentile = np.percentile(pixels_above_thresh, 10)
        mask[ira_img < lower_percentile] = 0
        # erode mask
        mask = self.erode_mask(mask)
        thresh = lower_percentile

        return thresh, mask
    
    def get_thresh_mask_otsu_gaussian(self, ira_img):
        # apply Gaussian blur to smooth the image
        ira_uint8 = ira_img.astype('uint8')
        ira_blurred = cv2.GaussianBlur(ira_uint8, (5, 5), 0)
        
        thresh, mask = cv2.threshold(ira_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        pixels_above_thresh = ira_blurred[ira_blurred >= thresh]
        if len(pixels_above_thresh) == 0:
            return thresh, mask
        # remove lower 25 percentile of the pixels above the threshold to reduce noise
        lower_percentile = np.percentile(pixels_above_thresh, 25)
        mask[ira_img < lower_percentile] = 0
        # erode mask
        mask = self.erode_mask(mask)
        thresh = lower_percentile
        
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
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
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
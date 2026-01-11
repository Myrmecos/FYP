# TODO: given a directory of images, convert them to a video file
# the top left of the image should have the index
# usage: python imgs_to_vid.py <img_dir> <output_vid_path> <fps>
# example: python presence_detection_workspace/src/data_visualization_module/imgs_to_vid.py /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1 /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall1/video/video1.mp4 30 all
# python presence_detection_workspace/src/data_visualization_module/imgs_to_vid.py /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0 /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall0/video/thermal.mp4 30 thermal
import cv2
import os
import sys
from pathlib import Path

# Add src directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_collection_module import utils
from dataset import ThermalDataset
from heatsource_detection_module.extract import HeatSourceDetector
from plot import DataVisualizer


# resize and stick the images side by side
# the height of all images should be the same
# the widths may be different
def stitch_images(images):
    resized_images = []
    heights = [img.shape[0] for img in images]
    min_height = min(heights)
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * (min_height / h))
        resized_img = cv2.resize(img, (new_w, min_height))
        resized_images.append(resized_img)
    stitched_image = cv2.hconcat(resized_images)
    return stitched_image

# convert images to video
def imgs_to_vid(data_dir, output_vid_path, fps=10):

    img_dir = os.path.join(data_dir, "Camera")
    ira_dir = os.path.join(data_dir, "IRA")
    ira_highres_dir = os.path.join(data_dir, "IRA_highres")
    tof_dir = os.path.join(data_dir, "ToF")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
    ira_files = sorted([f for f in os.listdir(ira_dir) if f.endswith('.pkl')])
    # ira_highres_files = sorted([f for f in os.listdir(ira_highres_dir) if f.endswith('.pkl')])
    tof_files = sorted([f for f in os.listdir(tof_dir) if f.endswith('.pkl')])

    first_img_path = os.path.join(img_dir, img_files[0])
    first_ira_path = os.path.join(ira_dir, ira_files[0])
    # first_ira_highres_path = os.path.join(ira_highres_dir, ira_highres_files[0])
    first_tof_path = os.path.join(tof_dir, tof_files[0])

    # the frame size should contain the four images side by side
    first_img = cv2.imread(first_img_path)
    first_ira = utils.load_pkl_as_img(first_ira_path, 0)
    first_ira_highres = utils.load_pkl_as_img(first_ira_path, -1)
    first_tof = utils.load_pkl_as_img(first_tof_path)
    stitched_frame = stitch_images([first_img, first_ira, first_ira_highres, first_tof])
    height, width, layers = stitched_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))
    cnt = 0

    for index in range(len(img_files)):
        img_path = os.path.join(img_dir, img_files[index])
        ira_path = os.path.join(ira_dir, ira_files[index])
        # ira_highres_path = os.path.join(ira_highres_dir, ira_highres_files[index])
        tof_path = os.path.join(tof_dir, tof_files[index])

        image = cv2.imread(img_path)
        ira_img = utils.load_pkl_as_img(ira_path, 0)
        ira_highres_img = utils.load_pkl_as_img(ira_path, -1)
        tof_img = utils.load_pkl_as_img(tof_path)
        frame = stitch_images([image, ira_img, ira_highres_img, tof_img])
        # add index to top left corner
        cv2.putText(frame, str(cnt), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cnt += 1
        video.write(frame)
    video.release()
    print(f"Video saved to {output_vid_path}")

def thermal_to_vid(path = None, output_vid_path = None, fps = 30):
    dataset = ThermalDataset(path)
    detector = HeatSourceDetector()
    visualizer = DataVisualizer()
    if output_vid_path is None:
        output_vid_path = os.path.join(path, "video", "thermal_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    first_ira_highres = dataset.get_ira_highres(0)
    height, width = first_ira_highres.shape
    video = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))

    for idx in range(len(dataset)):
        ira_highres = dataset.get_ira_highres(idx)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        mask = detector.remove_small_regions(mask, min_size=50)
        centroid = detector.get_centroid(mask)
       
        # color the thermal
        thermal_colored = utils.colorize_thermal_map(ira_highres)

        # plot the plot the centroid as a white cross on the thermal map:
        # please help write it:
        if centroid != (-1, -1):
            cv2.drawMarker(thermal_colored, centroid, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        # save to video
        video.write(thermal_colored)
    video.release()
    print(f"Thermal video saved to {output_vid_path}")  

def thermal_to_vid_with_blobs(path = None, output_vid_path = None, fps = 30):
    print("Generating thermal video with detected blobs...")
    dataset = ThermalDataset(path)
    detector = HeatSourceDetector()
    visualizer = DataVisualizer()
    if output_vid_path is None:
        output_vid_path = os.path.join(path, "video", "thermal_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    first_ira_highres = dataset.get_ira_highres(0)
    height, width = first_ira_highres.shape
    video = cv2.VideoWriter(output_vid_path, fourcc, fps, (width*10, height*10))

    for idx in range(len(dataset)):
        ira_highres = dataset.get_ira_highres(idx)
        thresh, mask = detector.get_thresh_mask_otsu(ira_highres)
        mask = detector.remove_small_regions(mask, min_size=50)
        centroids = detector.get_centroid_per_blob(mask)
       
        # color the thermal
        thermal_colored = utils.colorize_thermal_map(ira_highres)

        # plot the plot the centroid as a white cross on the thermal map:
        # please help write it:
        for centroid in centroids:
            if centroid != (-1, -1):
                cv2.drawMarker(thermal_colored, centroid, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        # resize the image to (640, 480)
        thermal_colored = cv2.resize(thermal_colored, (width*10, height*10))
        # save to video
        video.write(thermal_colored)
    video.release()
    print(f"Thermal video saved to {output_vid_path}")  
        

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python imgs_to_vid.py <img_dir> <output_vid_path> <fps> <all/thermal>")
    else:
        if sys.argv[4] == "thermal":
            path = sys.argv[1]
            output_vid_path = sys.argv[2]
            fps = int(sys.argv[3])
            thermal_to_vid(path, output_vid_path, fps)
        if sys.argv[4] == "blobs":
            path = sys.argv[1]
            output_vid_path = sys.argv[2]
            fps = int(sys.argv[3])
            thermal_to_vid_with_blobs(path, output_vid_path, fps)
        else:
            img_dir = sys.argv[1]
            output_vid_path = sys.argv[2]
            fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            imgs_to_vid(img_dir, output_vid_path, fps)


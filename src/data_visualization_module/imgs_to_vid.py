# TODO: given a directory of images, convert them to a video file
# the top left of the image should have the index
# usage: python imgs_to_vid.py <img_dir> <output_vid_path> <fps>
import cv2
import os
import sys
from pathlib import Path

# Add src directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_collection_module import utils


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
    first_ira = utils.load_pkl_as_img(first_ira_path)
    # first_ira_highres = utils.load_pkl_as_img(first_ira_highres_path)
    first_tof = utils.load_pkl_as_img(first_tof_path)
    stitched_frame = stitch_images([first_img, first_ira, first_tof])
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
        ira_img = utils.load_pkl_as_img(ira_path)
        # ira_highres_img = utils.load_pkl_as_img(ira_highres_path)
        tof_img = utils.load_pkl_as_img(tof_path)
        frame = stitch_images([image, ira_img, tof_img])
        # add index to top left corner
        cv2.putText(frame, str(cnt), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cnt += 1
        video.write(frame)
    video.release()
    print(f"Video saved to {output_vid_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python imgs_to_vid.py <img_dir> <output_vid_path> <fps>")
    else:
        img_dir = sys.argv[1]
        output_vid_path = sys.argv[2]
        fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        imgs_to_vid(img_dir, output_vid_path, fps)


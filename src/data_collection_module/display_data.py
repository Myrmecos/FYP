import os
import cv2
import pickle as pkl
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DataDisplayer():
    # ===========================read in data============================
    # read one image file given the filename
    # returns the image
    def read_image(self, imgname):
        return cv2.imread(imgname)
    # read one IRA data given the filename
    # returns the ira
    def read_ira(self, iraname):
        thermal = None
        with open(iraname, 'rb') as file:
            thermal = pkl.load(file)
        return thermal

    # read the dict of TOF
    # return last frame's composite data
    def read_tof(self, tofname):
        tof = None
        with open(tofname, 'rb') as file:
            tof = pkl.load(file)
        return tof

    # given a directory
    # returns all filenames in a list
    # since corresponding data from different modalities have same name except for suffix
    # we only need one list
    def read_data_names(self, base_dir_name):
        names = os.listdir(os.path.join(base_dir_name,"IRA/"))
        return_lst = [name.split(".")[0] for name in names]
        return_lst.sort()
        return return_lst
    
    # ============================visualize data=============================
    # thermal: color thermal image
    def colorThermal(self, senxor_temperature_map_m16):
        # print(senxor_temperature_map_m16)
        m16_min = -1024
        m16_max = -1024
        m16_min = np.min(senxor_temperature_map_m16)
        m16_max = np.max(senxor_temperature_map_m16)
        senxor_temperature_map_m16 = senxor_temperature_map_m16.astype(np.uint8)
        senxor_temperature_map_m16 = cv2.normalize(senxor_temperature_map_m16, None, 0, 255, cv2.NORM_MINMAX)
        senxor_temperature_map_m16 = cv2.resize(senxor_temperature_map_m16, (320, 240), interpolation=cv2.INTER_NEAREST)
        senxor_temperature_map_m16 = cv2.applyColorMap(senxor_temperature_map_m16, cv2.COLORMAP_JET)
        return senxor_temperature_map_m16
    
    # tof: color depth map
    def colorDepth(self, depth):
        # print("DEPTH: ", depth[:,:,0])
        vis_data = cv2.resize(depth[:,:,0], (240, 240), interpolation=cv2.INTER_NEAREST)
        vis_data = (vis_data) / (3500) * 255
        vis_data[vis_data > 255] = 255
        vis_data = cv2.applyColorMap((vis_data).astype(np.uint8), cv2.COLORMAP_JET)
        vis_data = np.flip(vis_data, 0)
        return vis_data

    def _colorDepth(self, depth):
        vis_data = (depth) / (3500) * 255
        vis_data[vis_data > 255] = 255
        vis_data = cv2.applyColorMap((vis_data).astype(np.uint8), cv2.COLORMAP_JET)
        vis_data = np.flip(vis_data, 0)
        return vis_data
    
    # display depth on a 3d-axis
    def _barDepth(self, depth, ax, elev = 52, azim = 109, roll = 4): #39, -35, 4
        # Create grid coordinates for 8x8 pixels
        # print(depth)
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        x = x.flatten()  # Flatten to 1D for bar3d
        y = y.flatten()
        z = np.zeros_like(x, dtype=float)  # Base of bars at z=0
        dx = dy = 0.5  # Width of bars (adjust for spacing)
        dz = (3000-depth).flatten()  # Heights of bars (depth values)

        # generate colors to correspond to 3d bar
        colors = self._colorDepth(depth)
        colors = colors[::-1, :, :]
        colors = colors[:, :, [2, 1, 0]].copy()
        colors = colors.reshape((64, 3)).astype(float) / 255.0
        
        # print("colors: ", colors)
        ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.8)
        ax.invert_xaxis()
        ax.view_init(elev, azim, roll)
    
    # displays depth image.
    def barDepth(self, depth, figsize = (10, 8), title = ""):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
        ax1.imshow(depth)
        self._barDepth(depth, ax2)
        plt.title(title)
        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close()
        # Convert to RGB (drop alpha channel if not needed)
        rgb = rgba[:, :, :3]  # Shape: (height, width, 3)
        return rgb

    # tof: color reflection
    def colorReflection(self, reflections):
        reflections_vis = cv2.resize(reflections[:,:,0], (240, 240), interpolation=cv2.INTER_NEAREST)
        reflections_vis = cv2.applyColorMap((reflections_vis).astype(np.uint8), cv2.COLORMAP_JET)
        reflections_vis = np.flip(reflections_vis, 0)
        return reflections_vis
    
    def combine_images_in_row(self, image_arrays):
        # print(image_arrays)
        # Convert arrays to PIL Images and find max height
        images = [Image.fromarray(img) for img in image_arrays]
        max_height = max(img.size[1] for img in images)
        total_width = sum(img.size[0] for img in images)
        
        # Create new image with max height and total width
        combined = Image.new('RGB', (total_width, max_height))
        
        # Paste images in a row
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.size[0]
            
        return np.array(combined)
    
    def combine_images_in_col(self, image_arrays):
        # Convert arrays to PIL Images and find max height
        # print(image_arrays[0].shape)
        images = [Image.fromarray(img) for img in image_arrays]
        max_width = max(img.size[0] for img in images)
        total_height = sum(img.size[1] for img in images)
        
        # Create new image with max height and total width
        combined = Image.new('RGB', (max_width, total_height))
        
        # Paste images in a row
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.size[1]
            
        return np.array(combined)
    
def test1():
    base_dir = "data/Data1"
    dd = DataDisplayer()
    img_dir = base_dir + "/Camera"
    ira_dir = base_dir + "/IRA"
    tof_dir = base_dir + "/tof"
    name_lst = dd.read_data_names(base_dir)
    rgb = dd.read_image(img_dir + "/" + name_lst[0] + ".jpg")
    # # debug: thermal read and visualize
    thermal = dd.read_ira(ira_dir + "/" + name_lst[0] + ".pkl")
    thermal = dd.colorThermal(thermal)
    # cv2.imshow("image", thermal)
    # cv2.waitKey(0)
    # # debug: tof read and visualize
    tof = dd.read_tof(tof_dir + "/" + name_lst[0] + ".pkl")
    #print(tof[0].keys())
    depth = dd.colorDepth(tof[0]['tof_depth'])
    reflections = dd.colorReflection(tof[0]['reflections'])

    combined = dd.combine_images_in_row([rgb, reflections, thermal])
    cv2.imshow('reflection', combined)
    cv2.waitKey(0)
    
if __name__=="__main__":
    base_dir = "data/Std06"
    dd = DataDisplayer()
    img_dir = base_dir + "/Camera"
    ira_dir = base_dir + "/IRA"
    tof_dir = base_dir + "/tof"
    name_lst = dd.read_data_names(base_dir)
    rgb = dd.read_image(img_dir + "/" + name_lst[0] + ".jpg")
    # # debug: thermal read and visualize
    thermal = dd.read_ira(ira_dir + "/" + name_lst[0] + ".pkl")
    # thermal = dd.colorThermal(thermal)
    # cv2.imshow("image", thermal)
    # cv2.waitKey(0)
    # # debug: tof read and visualize
    tof = dd.read_tof(tof_dir + "/" + name_lst[0] + ".pkl")
    
    depth = tof[0]['tof_depth'][:, :, 0]
    # print(depth)

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.imshow(depth)
    dd._barDepth(depth, ax2)
    plt.tight_layout()
    plt.show()
    # #test1()
    
    # cv2.imshow("depth", dd.barDepth(depth))
    # cv2.waitKey(0)

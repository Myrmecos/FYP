import cv2
import numpy as np
import time
import os
import pickle
from utils import *

# class def
class DataCollector():
    def __init__(self, save_path="entry_exit_detection/presence_detection_workspace/data", target_file="sleep_casestudy1"):
        self.save_path = save_path
        self.target_file = target_file
        print("*DEBUG: save path is ", self.save_path)
        self.makePath()
        self.thermalHighresCollector = Collector("/dev/cu.usbmodem11201")
        self.cnt = 0

    def run(self):
        """Main loop to collect and save IRA_highres data"""
        while True:
            self.collectAndSave()
            time.sleep(0.1)  # ~10 FPS

    def collectAndSave(self):
        """Read from thermal sensor and save"""
        # Get IRA highres from USB sensor
        ira_highres = self.thermalHighresCollector.getImage()
        if ira_highres is None:
            return

        # Create timestamp
        time_struct = time.localtime()
        ms = int((time.time() - int(time.time())) * 1000)
        formatted_time = time.strftime('%Y-%m-%d_%H-%M-%S', time_struct)
        time_string = f"{formatted_time}_{ms:03d}"

        # Save in same format: [{'time': ..., 'ira_temp': [None, ira_highres]}]
        # Wrapped in list so pickle.load(f)[-1] works
        ira_path = os.path.join(self.ira_highres_path, f"{time_string}.pkl")
        save_data = [{'time': time_string, 'ira_temp': [None, np.flip(ira_highres, 1)]}]
        with open(ira_path, 'wb') as f:
            pickle.dump(save_data, f)

        # Visualize
        self.visualize(ira_highres)

    def visualize(self, ira_highres):
        """Visualize IRA highres"""
        if ira_highres is None:
            return

        ira_highres = ira_highres.copy()
        ira_highres[ira_highres > 50] = 50
        ira_highres[ira_highres < 0] = 0
        ira_highres_vis = cv2.resize(ira_highres, (320, 240), interpolation=cv2.INTER_NEAREST)
        ira_highres_vis = (ira_highres_vis - ira_highres_vis.min()) / (ira_highres_vis.max() - ira_highres_vis.min()) * 255
        ira_highres_vis = cv2.applyColorMap((ira_highres_vis).astype(np.uint8), cv2.COLORMAP_JET)
        ira_highres_vis = np.flip(ira_highres_vis, 0)
        ira_highres_vis = np.flip(ira_highres_vis, 1)
        cv2.imshow('IRA Temp highres', ira_highres_vis)
        cv2.waitKey(1)

    def makePath(self):
        """Create save directories"""
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.save_path + '/' + self.target_file):
            os.mkdir(self.save_path + '/' + self.target_file)

        self.ira_highres_path = self.save_path + '/' + self.target_file + "/IRA/"
        if not os.path.exists(self.ira_highres_path):
            os.mkdir(self.ira_highres_path)


if __name__ == "__main__":
    dc = DataCollector()
    dc.run()

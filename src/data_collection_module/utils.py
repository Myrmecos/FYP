import serial
import time
import ast
import numpy as np
import cv2
import sys
import os
import signal
import logging
import cv2 as cv
from pprint import pprint
import argparse
import copy
import threading
from pathlib import Path

from collections import deque
import threading
import pickle
from time import sleep
from threading import Condition

# Add current module to path for senxor imports
sys.path.insert(0, str(Path(__file__).parent))

from senxor.utils import connect_senxor, data_to_frame, remap
from senxor.utils import cv_filter, cv_render, RollingAverageFilter

def put_temp(image, temp1, temp2, sensor_name):
    cv2.putText(image, f"{sensor_name}: {temp1:.1f}~{temp2:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f"{sensor_name}: {temp1:.1f}~{temp2:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

class MLXSensor:
    def __init__(self, sensor_port):
        self.sensor_port = sensor_port
        self.ser = serial.Serial(self.sensor_port, 921600, timeout=1)

    def read_data(self):
        data = self.ser.readline().strip()
        if len(data) > 0:
            try:
                msg_str = str(data.decode('utf-8'))
                msg = ast.literal_eval(msg_str)
                return msg
            except:
                return None
        return None
    
    def get_temperature_map(self):
        data = self.read_data()
        if data is not None:
            temp = np.array(data["temperature"]) # 768
            if len(temp) == 768:
                temp = temp.reshape(24, 32)
                return temp
        return None   
    
    def get_ambient_temperature(self):
        data = self.read_data()
        if data:
            return data["at"]
        return None
    
    def close(self):
        self.ser.close()
    
    def SubpageInterpolating(self,subpage):
        shape = subpage.shape
        mat = subpage.copy()
        for i in range(shape[0]):
            for j in range(shape[1]):
                if mat[i,j] > 0.0:
                    continue
                num = 0
                try:
                    top = mat[i-1,j]
                    num = num+1
                except:
                    top = 0.0
                
                try:
                    down = mat[i+1,j]
                    num = num+1
                except:
                    down = 0.0
                
                try:
                    left = mat[i,j-1]
                    num = num+1
                except:
                    left = 0.0
                
                try:
                    right = mat[i,j+1]
                    num = num+1
                except:
                    right = 0.0
                mat[i,j] = (top + down + left + right)/num
        return mat


class senxor:
    def __init__(self, sensor_port = "/dev/cu.usbmodem101"):
        self.sensor_port = sensor_port
        self.mi48 = connect_senxor(comport=self.sensor_port, reset_module=True)
        self.setup_thermal_camera(fps_divisor=3) 
        
        self.mi48.set_data_type('temperature')
        self.mi48.set_temperature_units('Celsius')
        
        self.ncols, self.nrows = self.mi48.fpa_shape
        self.mi48.start(stream=True, with_header=True)

    def get_temperature_map(self):
        return self.mi48.read() # data, header 
    
    def get_temperature_map_shape(self):
        return self.ncols, self.nrows
    
    def setup_thermal_camera(self, fps_divisor = 3):
        self.mi48.regwrite(0xB4, fps_divisor)  #
        # Disable firmware filters and min/max stabilisation
        if self.mi48.ncols == 160:
            # no FW filtering for Panther in the mi48 for the moment
            # self.mi48.regwrite(0xD0, 0x00)  # temporal
            # self.mi48.regwrite(0x20, 0x00)  # stark
            # self.mi48.regwrite(0x25, 0x00)  # MMS
            # self.mi48.regwrite(0x30, 0x00)  # median

            self.mi48.regwrite(0xD0, 0x00)  # temporal
            self.mi48.regwrite(0x30, 0x00)  # median
            self.mi48.regwrite(0x20, 0x03)  # stark
            self.mi48.regwrite(0x25, 0x01)  # MMS
        else:
            # MMS and STARK are sufficient for Cougar
            # self.mi48.disable_filter(f1=True, f2=True, f3=True)
            self.mi48.regwrite(0xD0, 0x00)  # temporal
            self.mi48.regwrite(0x30, 0x00)  # median
            self.mi48.regwrite(0x20, 0x03)  # stark
            self.mi48.regwrite(0x25, 0x01)  # MMS
        self.mi48.set_fps(30)
        self.mi48.set_emissivity(0.95)  # emissivity to 0.95, as used in calibration,
                                       # so there is no sensitivity change
        self.mi48.set_sens_factor(1.0)  # sensitivity factor 1.0
        self.mi48.set_offset_corr(0.0)  # offset 0.0
        self.mi48.set_otf(0.0)          # otf = 0
        self.mi48.regwrite(0x02, 0x00)  # disable readout error compensation
    
    def close(self):
        self.mi48.stop()
        
        
class realsense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)
        #below for testing only ====
        # device = profile.get_device()
        # device.hardware_reset()
        #above for testing only ====
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

def argb2bgr(frame):
    """Converts an RGBA8888 frame to a BGR frame."""
    if frame.shape[2] != 4:
        raise ValueError("Input frame must be RGBA8888")
    bgr_image = frame[:, :, 1:][:, :, ::-1]
    return bgr_image

    
class image_buffer():
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.read = 0
        self.write = 0
        self.buffer = []
        for i in range (buffer_size):
            self.buffer.append(None)

    
    def add(self, image):
        #if self.buffer[self.write] is not None:
        self.buffer[self.write] = image
        self.write += 1
        self.write = self.write%self.buffer_size

    
    def get(self):
        self.read += 1
        self.read %= self.buffer_size
        return self.buffer[self.read]


class Collector():
    def __init__(self, port = "/dev/cu"):
        self.senxor_sensor_m16 = senxor(sensor_port=port)
        self.img_buffer = [None]*3 # a ring buffer
        self.latest_available_ind = -1 # the index of the most recent image in the buffer
        self.buffer_lock = threading.Lock()
        self.reading_thread = threading.Thread(target = self._start_listening, args=())
        self.reading_thread.start()
    # listen to images from senxor m16
    # put them into buffer
    def _start_listening(self):
        num_rows_m16, num_cols_m16 = self.senxor_sensor_m16.get_temperature_map_shape()
        while True:
            # print("receiving image")
            senxor_temperature_map_m16, header2 = self.senxor_sensor_m16.get_temperature_map()
            if senxor_temperature_map_m16 is None:
                continue
            senxor_temperature_map_m16 = senxor_temperature_map_m16.reshape(num_cols_m16, num_rows_m16)
            senxor_temperature_map_m16 = np.flip(senxor_temperature_map_m16, 0)
            with self.buffer_lock:
                self.latest_available_ind = (self.latest_available_ind + 1)%3
                self.img_buffer[self.latest_available_ind] = senxor_temperature_map_m16
    def getImage(self):
        returnImg = None
        with self.buffer_lock:
            returnImg = self.img_buffer[self.latest_available_ind]
        return returnImg
    
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
        put_temp(senxor_temperature_map_m16, m16_min, m16_max, "m16")
        return senxor_temperature_map_m16


def load_npy_as_img(npy_path):
    npy_data = np.load(npy_path)
    npy_data = npy_data.astype(np.uint8)
    npy_data = cv2.normalize(npy_data, None, 0, 255, cv2.NORM_MINMAX)
    npy_data = cv2.resize(npy_data, (320, 240), interpolation=cv2.INTER_NEAREST)
    npy_data = cv2.applyColorMap(npy_data, cv2.COLORMAP_JET)
    return npy_data

# idx = 0: mlx (low res)
# idx = -1: m08 (high res)
def load_pkl_as_img(pkl_path, idx = -1):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)[-1]
    # print(pkl_data)
    if "ira_temp" in pkl_data:
        pkl_data = pkl_data["ira_temp"][idx]
    elif "tof_depth" in pkl_data:
        pkl_data = pkl_data['tof_depth'][:, :, 0]
    pkl_data = pkl_data.astype(np.uint8)
    pkl_data = cv2.normalize(pkl_data, None, 0, 255, cv2.NORM_MINMAX)
    pkl_data = cv2.resize(pkl_data, (320, 240), interpolation=cv2.INTER_NEAREST)
    pkl_data = cv2.applyColorMap(pkl_data, cv2.COLORMAP_JET)
    return pkl_data

def colorize_thermal_map(thermal_map):
    thermal_map = thermal_map.astype(np.uint8)
    thermal_map = cv2.normalize(thermal_map, None, 0, 255, cv2.NORM_MINMAX)
    thermal_map = cv2.applyColorMap(thermal_map, cv2.COLORMAP_JET)
    return thermal_map

def load_yaml_as_dict(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

if __name__ == "__main__":
    '''
    Visualization only:
        python data_collection.py --save_data 0
    Collecting data:
        python data_collection.py --collection_duration 600 --save_data 1 --save_path path_to_save_the_data --sleep_time 0 --vis_flag 0
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_duration", type=int, default=10, help="Duration to collect data, seconds")
    parser.add_argument("--save_data", type=int, default=0, help="0: not save, just visualize, 1: save to a pickle file without visualization")
    parser.add_argument("--save_path", type=str, default="data", help="path to save data")
    parser.add_argument("--sleep_time", type=float, default=0, help="sleep time between each frame")
    parser.add_argument("--vis_flag", type=int, default=0, help="enable visualization or not")
    args = parser.parse_args()
    args.save_path = "/media/zx/zx-data/" + args.save_path
    
    senxor_sensor_m16 = senxor(sensor_port="/dev/cu.usbmodem101")
    
    buffer_len = 3

    seek_camera_buffer = image_buffer(buffer_len)
    realsense_color_buffer = image_buffer(buffer_len)
    realsense_depth_buffer = image_buffer(buffer_len)
    mlx_buffer = image_buffer(buffer_len)

    num_rows_m16, num_cols_m16 = senxor_sensor_m16.get_temperature_map_shape()

    print("before collecting data=================================================")
    framecnt = 0   # the number of the received frames
    saved_frame_cnt = 0  # the number of the saved frames
    start_time = time.time()
    collection_duration = args.collection_duration
    sleep_time = args.sleep_time   # sleep time between each frame, control the collecting speed
    last_collect_time = time.time()

    while True:
        print("into cycle====================================")
        #print("===========debug: start collecting data, frame:", framecnt, "================") 
        framecnt+=1
        
        senxor_temperature_map_m16_ori, header2 = senxor_sensor_m16.get_temperature_map()
        

        
        if senxor_temperature_map_m16_ori  is None:
            print("senxor_m16 is none")
            continue
        else:
            # adjust the orientation of the images/frames
            senxor_temperature_map_m16 = senxor_temperature_map_m16_ori.reshape(num_cols_m16, num_rows_m16)
            senxor_temperature_map_m16 = np.flip(senxor_temperature_map_m16, 0)
            
            timestamp = time.time()
            if args.save_data==1 and (timestamp-last_collect_time) > sleep_time:
                np.save(f"{args.save_path}/senxor_m16/{timestamp}.npy", senxor_temperature_map_m16)
                saved_frame_cnt += 1
                last_collect_time = timestamp
            
            # for visualization only
            if args.vis_flag:

                m16_min = -1024
                m16_max = -1024
                m16_min = np.min(senxor_temperature_map_m16)
                m16_max = np.max(senxor_temperature_map_m16)
                senxor_temperature_map_m16 = senxor_temperature_map_m16.astype(np.uint8)
                senxor_temperature_map_m16 = cv2.normalize(senxor_temperature_map_m16, None, 0, 255, cv2.NORM_MINMAX)
                senxor_temperature_map_m16 = cv2.resize(senxor_temperature_map_m16, (320, 240), interpolation=cv2.INTER_NEAREST)
                senxor_temperature_map_m16 = cv2.applyColorMap(senxor_temperature_map_m16, cv2.COLORMAP_JET)
            
                put_temp(senxor_temperature_map_m16, m16_min, m16_max, "m16")

                #print(realsense_depth_image.shape, realsense_color_image.shape, seek_camera_frame.shape,  senxor_temperature_map_m08.shape, MLX_temperature_map.shape,)
                cv2.imshow("Final Image", senxor_temperature_map_m16)

            time_lasting = time.time() - start_time
            if time_lasting > collection_duration:
                timestamp = time.time()
                print(f"Senxor temperature map m16 collected at {timestamp}", senxor_temperature_map_m16.shape)
                print(f"Frame rate: {framecnt / time_lasting} Hz")

                # save the above as metadata in meta.log in the save_path
                try:
                    with open(f"{args.save_path}/meta.log", "w") as f:
                        f.write(f"Collecting time: {time_lasting} seconds\n")
                        f.write(f"Total frames received: {framecnt}\n")
                        f.write(f'Total frames saved: {saved_frame_cnt}\n')
                        f.write(f"Frame rate: {framecnt / time_lasting} Hz\n")
                except:
                    pass
                break
            #cv2.imshow("Realsense Color Image", realsense_color_image)
            if args.save_data==0:
                key = cv.waitKey(1)
                if key in [ord("q"), ord('Q'), 27]:
                    break
        
    senxor_sensor_m16.close()
 
         
import socket
import cv2
import numpy as np
import multiprocessing
import struct
import time
import serial
import argparse
import os
import pickle
import subprocess
import re
import traceback
import threading
from display_data import DataDisplayer
from utils import *

# ==========utility functions=============
# interpolation for ira temperature data
def SubpageInterpolating(subpage):
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

# decode TOF result
def tof_decode_results(packet_data):
    offset = 0
    # tof_depth = np.frombuffer(packet_data[offset:offset + 128*3], dtype=np.uint16).reshape((8, 8,3))
    tof_depth = np.frombuffer(packet_data[offset:offset + 128*3], dtype='>u2').reshape(( 8, 8,3))  # '>u2' 表示大端的 uint16

    offset += 128*3
    reflections = np.frombuffer(packet_data[offset:offset + 64*3], dtype=np.uint8).reshape((8, 8,3))
    offset += 64
    target_status = np.frombuffer(packet_data[offset:offset + 64*3], dtype=np.uint8).reshape((8, 8,3))
    return tof_depth, reflections, target_status

def ira_decode_results(packet_data):
    if len(packet_data) < 768 * 2:
        raise ValueError("Received packet size is too small!")
    # 解析数据（大端序）
    ira_temp = np.frombuffer(packet_data[:768*2], dtype='>u2')  # 解析为 uint16
    ira_temp = ira_temp.astype(np.float32) / 100.0  # 还原为 float
    ira_temp = ira_temp.reshape((24, 32))  # 形状匹配
    # print("Decoded IRA Temp: Max =", ira_temp.max(), "Min =", ira_temp.min())
    return ira_temp

# class def
class DataCollector():
    def __init__(self, save_path="presence_detection_workspace/data", target_file="hall1"):
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 6900
        self.BUFFER_SIZE = 1460 // 2 + 13  # 数据包的长度：数据部分加上5个字节的头部信息
        self.save_path = save_path
        self.target_file = target_file #save path #head chest waist leg feet
        #self.target_file = "calib"
        self.dd = DataDisplayer()
        print("*DEBUG: save path is ", self.save_path)
        self.makePath()
        self.frame_queue = multiprocessing.Queue()
        self.thermalHighresCollector = Collector("/dev/cu.usbmodem1301")
        self.cnt = 0

    # the "main" function for displaying and saving data
    def displayAndSave(self):
        self.framecnt = 0
        self.tof_ready = False; self.ira_ready = False; self.rgb_ready = False
        self.tof_save = []; self.ira_save = []; self.rgb_save = []
        while True:
            self.framecnt += 1
            # print("================Processing data===============")
            frame_data = self.frame_queue.get()
            self.makeTimeString(frame_data)
            self.processFrame(frame_data)

            print(f"tof, ira, rgb ready: {self.tof_ready}, {self.ira_ready}, {self.rgb_ready}")
            if self.tof_ready and self.ira_ready and self.rgb_ready:
                self.tof_ready = False; self.ira_ready = False; self.rgb_ready = False
                self.save()
                self.visualize()
                self.rgb_save = []
                self.tof_save = []
                self.ira_save = []
                

    
    def save(self):
        camera_image = self.rgb_save[0]
        if camera_image is None or self.tof_save is None or self.ira_save is None:
            return
        
        # self.cnt += 1
        # if self.cnt % 4 == 0:
        #     self.cnt = 0
        # else:
        #     return
        # ============ rgb ================
        
        camera_filename = os.path.join(self.camera_path, f"{self.time_string}.jpg")
        if self.framecnt%20==0:
            pass
        cv2.imwrite(camera_filename, camera_image)

        # =========== TOF depth =============
        tof_path = os.path.join(self.tof_path, f"{self.time_string}.pkl")
        with open(tof_path, 'wb') as f:
            pickle.dump(self.tof_save, f)

        # ============= IRA =========
        ira_filename = os.path.join(self.ira_path, f"{self.time_string}.pkl")
        with open(ira_filename, 'wb') as f:
            pickle.dump(self.ira_save, f)



    def visualize(self):
        rgb = self.rgb_save[-1]
        tof_depth = self.tof_save[-1]['tof_depth'][:, :, 0]
        tof_reflection = self.tof_save[-1]['reflections'][:, :, 0]
        ira_temp = self.ira_save[-1]['ira_temp'][0]
        ira_highres = self.ira_save[-1]['ira_temp'][1]
        # ============ rgb ================
        # print(f"rgb is none?: {rgb is None}")
        # print(f"tof is none?: {tof_depth is None}")
        # print(f"ira is none?: {ira_temp is None}")
        # print(f"ira_highres is none?: {ira_highres is None}")

        if rgb is None or tof_depth is None or ira_temp is None or ira_highres is None:
            return
        # print(rgb)
        rgb = np.flip(rgb, 0)
        rgb = np.flip(rgb, 1)
        cv2.imshow("RGB image", rgb)
        cv2.waitKey(1)

        # =========== TOF depth =============
        # print("tof_data looks like: ", self.tof_data)
        tof_vis = cv2.resize(tof_depth, (240, 240), interpolation=cv2.INTER_NEAREST)
        tof_vis = (tof_vis) / (3500) * 255
        tof_vis[tof_vis > 255] = 255
        tof_vis = cv2.applyColorMap((tof_vis).astype(np.uint8), cv2.COLORMAP_JET)
        tof_vis = np.flip(tof_vis, 0)
        tof_vis = np.flip(tof_vis, 1)
        #vis_data = self.dd.barDepth(tof_depth[:,:,0], (5, 4))
        cv2.imshow('TOF Depth', tof_vis)
        cv2.waitKey(1)

        # ============ TOF reflection ===========
        reflections_vis = cv2.resize(tof_reflection, (240, 240), interpolation=cv2.INTER_NEAREST)
        reflections_vis = cv2.applyColorMap((reflections_vis).astype(np.uint8), cv2.COLORMAP_JET)
        reflections_vis = np.flip(reflections_vis, 0)
        reflections_vis = np.flip(reflections_vis, 1)
        cv2.imshow('TOF reflection', reflections_vis)
        cv2.waitKey(1)

        # ===============IRA =================
        # Visualize IRA temperature data
        ira_temp = SubpageInterpolating(ira_temp)
        # if there is value lager than 100, set it to 100
        ira_temp[ira_temp > 50] = 50
        ira_temp[ira_temp < 0] = 0
        ira_temp_vis = cv2.resize(ira_temp, (320, 240), interpolation=cv2.INTER_NEAREST)
        # normalize the temperature data
        ira_temp_vis = (ira_temp_vis - ira_temp_vis.min()) / (ira_temp_vis.max() - ira_temp_vis.min()) * 255
        ira_temp_vis = cv2.applyColorMap((ira_temp_vis).astype(np.uint8), cv2.COLORMAP_JET)
        ira_temp_vis = np.flip(ira_temp_vis, 0)
        ira_temp_vis = np.flip(ira_temp_vis, 1)
        cv2.imshow('IRA Temp', ira_temp_vis)
        cv2.waitKey(1)

        # ================= IRA highres ===============
        # Visualize IRA temperature data
        # if there is value lager than 50, set it to 50
        # similarly for 0
        ira_highres[ira_highres > 50] = 50
        ira_highres[ira_highres < 0] = 0
        ira_temp_vis = cv2.resize(ira_highres, (320, 240), interpolation=cv2.INTER_NEAREST)
        # normalize the temperature data
        ira_temp_vis = (ira_temp_vis - ira_temp_vis.min()) / (ira_temp_vis.max() - ira_temp_vis.min()) * 255
        ira_temp_vis = cv2.applyColorMap((ira_temp_vis).astype(np.uint8), cv2.COLORMAP_JET)
        ira_temp_vis = np.flip(ira_temp_vis, 0)
        ira_temp_vis = np.flip(ira_temp_vis, 1)
        cv2.imshow('IRA Temp highres', ira_temp_vis)
        cv2.waitKey(1)     

    # take one frame, process it and store info
    def processFrame(self, frame_data):
        # print("frame data looks like this: ", frame_data)
        print("keys of frame data", frame_data.keys())
        if isinstance(frame_data, dict):
            if 'tof_depth' in frame_data.keys():
                tof_depth = frame_data.get('tof_depth')
                reflections = frame_data.get('reflections')
                target_status = frame_data.get('target_status')
                self.tof_ready = True
                self.tof_save.append({'time':self.time_string, 'tof_depth':tof_depth, 'reflections':reflections, 'target_status':target_status})
            if 'ira_temp' in frame_data.keys():
                ira_temp = frame_data.get('ira_temp')
                self.ira_save.append({'time': self.time_string, 'ira_temp': ira_temp})
                self.ira_ready = True
            if 'camera_data' in frame_data.keys():
                #cam_count+=1
                camera_data = frame_data.get('camera_data')
                np_data = np.frombuffer(camera_data, dtype=np.uint8)
                # decoding the image data, and flip the image
                image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                image = cv2.flip(image, 1)
                
                self.rgb_save.append(image)
                self.rgb_ready = True
            
    # make a string that represents the frame time
    def makeTimeString(self, frame_data):
        frame_time = frame_data.get('time')
        ms = int((frame_time-int(frame_time))*1000)
        frame_time = frame_time / 1000  # 转为秒级时间戳（浮点数）
        time_struct = time.localtime()
        formatted_time = time.strftime('%Y-%m-%d_%H-%M-%S', time_struct)
        self.time_string = f"{formatted_time}_{ms:03d}"

    # make directories to save results
    def makePath(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.save_path + '/' + self.target_file):
            os.mkdir(self.save_path + '/' + self.target_file)
        print(f"*DEBUG: target path {self.save_path + '/' + self.target_file} exists? {os.path.exists(self.save_path + '/' + self.target_file)}")
        
        self.camera_path = self.save_path + '/' + self.target_file + "/Camera/"   # this is the path to save the camera frames
        if not os.path.exists(self.camera_path):
            os.mkdir(self.camera_path)

        
        self.tof_path = self.save_path + '/' + self.target_file + "/ToF/"    # this is the path to save the tof data
        if not os.path.exists(self.tof_path):
            os.mkdir(self.tof_path)

        self.ira_path = self.save_path + '/' + self.target_file + "/IRA/"  # this is the path to save the ira data
        if not os.path.exists(self.ira_path):
            os.mkdir(self.ira_path)

        self.ira_highres_path = self.save_path + '/' + self.target_file + "/IRA_highres/"
        if not os.path.exists(self.ira_highres_path):
            os.mkdir(self.ira_highres_path)
    
    # handle decoded frames in another process
    def reset_frame(self):
        self.current_frame_data = {}
        self.tof_data = {}
        self.irate_data = {}

        
        self.camera_received_packet_count = 0
        self.camera_expected_packet_count = 0

        self.tof_received_packet_count = 0

        self.irate_received_packet_count = 0

        self.camera_frame_complete = False
        self.tof_frame_complete = False
        self.irate_frame_complete = False
    

    # listen to udp, collect, put into queue
    def listenAndCollect(self):
        self.reset_frame()
        self.current_frame_id = 0

        self.tof_expected_packet_count = 0
        self.tof_received_packet_count = 0

        self.irate_expected_packet_count = 0
        self.irate_received_packet_count = 0

        self.camera_expected_packet_count = 0
        self.camera_received_packet_count = 0


        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许端口重用
        sock.bind((self.UDP_IP, self.UDP_PORT))
        while True:
            data, addr = sock.recvfrom(self.BUFFER_SIZE)
            if len(data) < 5:
                print("DEBUG: packet too small, we skip it.")
            
            self.parse_data(data)
    
    def parse_data(self, data):
        # 解析数据类型和帧序号、包序号
        data_type = chr(data[0])
        frame_id = (data[2] << 8) | data[1]
        packet_id = (data[4] << 8) | data[3]
        timestamp_bytes = data[5:13]  # 提取时间戳的 4 字节
        timestamp = struct.unpack('<Q', timestamp_bytes)[0]  # 小端字节序解析 uint64_t
        # print(timestamp)
        # 数据部分
        packet_data = data[13:]
        frame_time = time.time()

        #print(f"Packet id: {packet_id}")
        if frame_id != self.current_frame_id:
            if self.current_frame_id is not None and self.camera_frame_complete:
                sorted_camera_data = b''.join(self.current_frame_data[i] for i in sorted(self.current_frame_data))
                # 将完整的帧数据放入队列中
                if sorted_camera_data is not None:
                    self.frame_queue.put({
                        'camera_data': sorted_camera_data,
                        'time': frame_time
                    })
            
            if self.current_frame_id is not None and self.tof_frame_complete:
                # 组装 TOF 数据并发送至显示队列
                #print("saving TOF data")
                full_tof_data = b''.join(self.tof_data[i] for i in sorted(self.tof_data))
                tof_depth, reflections, target_status = tof_decode_results(full_tof_data)
                # print("DEBUG: tof depth: ", tof_depth)
                self.frame_queue.put({
                    'tof_depth': np.flip(tof_depth, 1),
                    'reflections': np.flip(reflections, 1),
                    'target_status': target_status,
                    'time': frame_time
                })
            
            if self.current_frame_id is not None and self.irate_frame_complete:
                # 组装ira data
                #print("saving ira data")
                print(f"ira data size: {len(self.irate_data)}")
                print(f"ira expected packet num: {self.irate_expected_packet_count}")
                print(f"irate frame complete: {self.irate_frame_complete}")
                full_irate_data = b''.join(self.irate_data[i] for i in sorted(self.irate_data))
                ira_temp = ira_decode_results(full_irate_data)
                ira_temp_highres = self.thermalHighresCollector.getImage()
                self.frame_queue.put({
                    'ira_temp': [ira_temp, np.flip(ira_temp_highres, 1)], 
                    'time': frame_time
                })
                

            self.current_frame_id = frame_id
            # frame_time = timestamp
            #frame_time = time.time()
            self.reset_frame()
        
        #print(f"data type is: {data_type}")
        if data_type == 'R':
            self.tof_data[packet_id] = packet_data
            self.tof_received_packet_count += 1
            self.tof_expected_packet_count = max(self.tof_expected_packet_count, packet_id)
        elif data_type == 'C':
            self.current_frame_data[packet_id] = packet_data
            self.camera_received_packet_count += 1
            self.camera_expected_packet_count = max(self.camera_expected_packet_count, packet_id)
        elif data_type == 'T':
            self.irate_data[packet_id] = packet_data
            self.irate_received_packet_count += 1
            self.irate_expected_packet_count = max(self.irate_expected_packet_count, packet_id)

        if (self.tof_received_packet_count == self.tof_expected_packet_count) and len(self.tof_data) > 0:
            self.tof_frame_complete = True

        print(f"DEBUG: camera received {self.camera_received_packet_count} frames, need {self.camera_expected_packet_count} frames.")
        if (self.camera_received_packet_count == self.camera_expected_packet_count) and len(self.current_frame_data) > 0:
            
            self.camera_frame_complete = True
        
        if (self.irate_received_packet_count == self.irate_expected_packet_count) and len(self.irate_data) > 0:
            self.irate_frame_complete = True


if __name__ == "__main__":
    # parse arguments from presence_detection_workspace/config/data_collect.yaml

    dc = DataCollector()
    display_process = threading.Thread(target=dc.listenAndCollect)
    display_process.start()
    #dc.listenAndCollect()
    dc.displayAndSave()
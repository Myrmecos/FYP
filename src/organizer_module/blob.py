# A Blob represent a detected heat source in the scene
import cv2

class Blob:
    def __init__(self, id=None, mask=None, masked_temps=None, mean_temp=None, centroid=None):
        self.id = id  # unique identifier for the blob
        self.mask = mask
        self.masked_temps = masked_temps # the thermal image with mask applied
        self.mean_temp = mean_temp
        self.centroid = centroid
        self.temp_history = [] # if temp is decreasing, is heat residual, not human
        self.centroid_history = [] # if move is directional (in some segments), likely human
        self.queue_len = 400 # length of history to keep, longer history for better analysis

    def update(self, mask, masked_temps):
        self.mask = mask
        self.masked_temps = masked_temps
        self._compute_centroid()
        self._compute_mean_temp()
        self.temp_history.append(self.mean_temp)
        self.centroid_history.append(self.centroid)

        if len(self.temp_history) > self.queue_len:
            self.temp_history.pop(0)
        if len(self.centroid_history) > self.queue_len:
            self.centroid_history.pop(0)
    
    def get_position(self):
        return self.centroid

    def get_mask(self):
        return self.mask
    
    def _compute_centroid(self):
        moments = cv2.moments(self.mask.astype('uint8'))
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            self.centroid = (cX, cY)
        else:
            self.centroid = (-1, -1)

    def _compute_mean_temp(self):
        if self.masked_temps.size == 0:
            return -1
        self.mean_temp = self.masked_temps.mean()
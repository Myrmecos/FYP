# A Blob represent a detected heat source in the scene
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from organizer_module import convert

# remove residual detection functionality
def mask_to_bbox(mask):
    # compute bounding box from binary mask
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    return [x_min, y_min, x_max, y_max]

def bbox_to_centroid(bbox):
    x_center = (bbox[0] + bbox[2]) // 2
    y_center = (bbox[1] + bbox[3]) // 2
    return (x_center, y_center)

class KalmanBlob(object):
    id = 0
    '''
    This class represents the internal state of individual tracked objects observed as blobs.
    '''
    def __init__(self, id=None, mask=None, masked_temps=None, mean_temp=None, centroid=None):
        self.id = KalmanBlob.id  # unique identifier for the blob
        KalmanBlob.id += 1
        self.mask = mask
        self.prev_mask = None
        self.masked_temps = masked_temps # the thermal image with mask applied
        self.mean_temp = self._compute_mean_temp()
        self.centroid = centroid
        self.temp_history = [] # if temp is decreasing, is heat residual, not human
        self.centroid_history = [] # if move is directional (in some segments), likely human
        self.kalman_centroid_history = []  # history of centroids from Kalman filter
        self.queue_len = 400 # length of history to keep, longer history for better analysis

        self.bbox = mask_to_bbox(mask) if mask is not None else [0,0,0,0]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        
        self.kf.R[2:, 2:] *= 1.   # 10.
        self.kf.P[4:, 4:] *= 10.  # 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert.bbox_to_z(self.bbox)

        self.time_since_update = 0
        self.age = 0
        self.time_since_observed = 0
        self.confidence = 0.5
        

    def update(self, mask, masked_temps, isobserved):
        self.prev_mask = self.mask
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

        self.time_since_update = 0
        if not isobserved:
            self.kf.x[6] /= 2
            self.kf.update(None)
            self.time_since_observed += 1
        else:
            self.kf.update(convert.bbox_to_z(mask_to_bbox(mask)))
            self.time_since_observed = 0
        self.kalman_centroid_history.append(bbox_to_centroid(self.get_state()))
        if len(self.kalman_centroid_history) > self.queue_len:
            self.kalman_centroid_history.pop(0)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # to prevent area become negative after prediction, make zero the rate of area change
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return convert.x_to_bbox(self.kf.x)
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert.x_to_bbox(self.kf.x)
    
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
        self.mean_temp = self.masked_temps[self.masked_temps > 0].mean()
        return self.mean_temp

    def outside_frame(self, frame_shape, margin = 3):
        img_h, img_w = frame_shape
        x_min, y_min, x_max, y_max = self.get_state()
        if x_max < -margin or y_max < -margin or x_min > img_w + margin or y_min > img_h + margin:
            return True
        return False
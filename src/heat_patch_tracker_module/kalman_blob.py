# A Blob represent a detected heat source in the scene
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

# Helpers
def bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))[0]
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))[0]

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
        self.max_k = 0 #DEBUG
        self.id = KalmanBlob.id  # unique identifier for the blob
        self.id_fixed = KalmanBlob.id  # fixed id for the blob, will not change even if marked as residual
        KalmanBlob.id += 1
        self.is_residual = False
        self.mask = mask
        self.prev_mask = None
        self.masked_temps = masked_temps # the thermal image with mask applied
        self.mean_temp = self._compute_mean_temp()
        self.max_temp = self._compute_max_temp()
        self.median_temp = self._compute_median_temp()
        self.centroid = centroid
        self.temp_history = [] # if temp is decreasing, is heat residual, not human
        self.temp_trend = 0
        self.corr = 0 # temperature attenuation coefficient estimated
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
        self.kf.x[:4] = bbox_to_z(self.bbox)

        self.time_since_update = 0
        self.age = 0
        self.time_since_observed = 0
        self.confidence = 0.5
        

    def update(self, mask, masked_temps, isobserved = True):
        """Update the state of the blob with new observations.
        mask: binary mask of the detected blob
        masked_temps: the thermal image with mask applied, used to compute temperature features"""
        self.prev_mask = self.mask
        self.mask = mask
        self.masked_temps = masked_temps
        self._compute_centroid()
        self._compute_mean_temp()
        self._compute_median_temp()
        self._compute_max_temp()
        self.temp_history.append(self.max_temp)
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
            self.kf.update(bbox_to_z(mask_to_bbox(mask)))
            self.time_since_observed = 0
        self.kalman_centroid_history.append(bbox_to_centroid(self.get_state()))
        if len(self.kalman_centroid_history) > self.queue_len:
            self.kalman_centroid_history.pop(0)

    def update_temp_trend(self):
        if len(self.temp_history) < 5:  # need enough history to make a decision
            return 0
        self.temp_trend = np.polyfit(range(len(self.temp_history)), np.array(self.temp_history, dtype=np.float32), 1)[0]  # linear trend
        return self.temp_trend
    
    def get_velocity(self):
        # return the velocity based on Kalman filter state
        # vel is the absolute values
        return np.linalg.norm([self.kf.x[4], self.kf.x[5]])
    
    def predict(self):
        """
        update when no observation
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # to prevent area become negative after prediction, make zero the rate of area change
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return x_to_bbox(self.kf.x)
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return x_to_bbox(self.kf.x)
    
    def get_position(self):
        return self.centroid

    def get_mask(self):
        return self.mask
    
    def _compute_centroid(self):
        """Compute current frames' centroid of the blob, excluding zero values which are outside mask"""
        moments = cv2.moments(self.mask.astype('uint8'))
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            self.centroid = (cX, cY)
        else:
            self.centroid = (-1, -1)

    def _compute_mean_temp(self):
        """Compute current frames' mean temerature of the blob, excluding zero values which are outside mask"""
        if self.masked_temps.size == 0:
            return -1
        # take the 25 to 75 percentile mean to reduce the influence of noise and outliers
        lower_percentile = np.percentile(self.masked_temps[self.masked_temps > 0], 25)
        upper_percentile = np.percentile(self.masked_temps[self.masked_temps > 0], 75)
        self.mean_temp = self.masked_temps[(self.masked_temps > lower_percentile) & (self.masked_temps < upper_percentile)].mean()
        return self.mean_temp
    
    def _compute_median_temp(self):
        """Compute current frames' median temerature of the blob, excluding zero values which are outside mask"""
        if self.masked_temps.size == 0:
            return -1
        # take the 25 to 75 percentile mean to reduce the influence of noise and outliers
        lower_percentile = np.percentile(self.masked_temps[self.masked_temps > 0], 25)
        upper_percentile = np.percentile(self.masked_temps[self.masked_temps > 0], 75)
        self.median_temp = np.median(self.masked_temps[(self.masked_temps > lower_percentile) & (self.masked_temps < upper_percentile)])
        return self.median_temp

    def _compute_max_temp(self):
        """Compute current frames' maximum temerature of the blob, excluding zero values which are outside mask"""
        if self.masked_temps.size == 0:
            return -1
        # take the 25 to 75 percentile mean to reduce the influence of noise and outliers
        lower_percentile = np.percentile(self.masked_temps[self.masked_temps > 0], 1)
        upper_percentile = np.percentile(self.masked_temps[self.masked_temps > 0], 99)
        self.max_temp = np.max(self.masked_temps[(self.masked_temps > lower_percentile) & (self.masked_temps < upper_percentile)])
        # self.max_temp = 30
        return self.max_temp

    def outside_frame(self, frame_shape, margin = 3):
        # check if the blob is outside the frame
        img_h, img_w = frame_shape
        x_min, y_min, x_max, y_max = self.get_state()
        if x_max < -margin or y_max < -margin or x_min > img_w + margin or y_min > img_h + margin:
            return True
        return False
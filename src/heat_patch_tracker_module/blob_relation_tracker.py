# This module is responsible for tracking the relationships between blobs detected in the heat patches.
# It helps:
# 1. identify the splitting events that occurs when human heat patch and residual heat patch are merged or split.
# 2. identify if there are multiple objects (e.g. human + residual heat) in the same blob.
# 3. differentiate between human heat patch and residual heat patch based on the temporal changes of the blob's properties (e.g. mean temperature, area, etc.)
from .kalman_blob import KalmanBlob, mask_to_bbox, bbox_to_centroid
class BlobRelationTracker:

    def __init__(self):
        self.humanTempRange = [30.0, 40.0] # the temperature threshold to differentiate between human heat patch and residual heat patch, can be tuned based on the dataset

    def identify_blob_type(self, blob):
        """Based on 1. the temperature profile (min, max, median) and 
        2. temperature history (a given number of frames) and
        3. the motion history of the blob, 
        identify if it's a human heat patch or a residual heat patch.
        and update the criteria for human heat patch and residual heat patch.
        """
        pass

    def track_blob_relations(self, blobs):
        pass

    def wereSameBlob(self, blob_id1, blob_id2):
        """Check if two blobs were results of one blob split event"""
        return False


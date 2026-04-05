# This module serves the purpose of:
# 1. Classifies the posture
# 2. warning of bed exit events
# It aims to do so using learning based methods
# e.g. 1. V-JEPA
from torch import nn

class PostureDetector:
    def __init__(self, imgHistorySize=10):
        self.isExiting = False
        self.imgHistory = []
        self.imgHistorySize = imgHistorySize

    def classify_posture(self, img):
        # posture categories:
        #   # -1: unknown or unlabeled; 
            # 0: absence; 
            # 1: presence, unclassified; 
            # 2: standing; 
            # 3: sitting by bed; 
            # 4: sitting on bed; 
            # 5: lying w/o cover; 
            # 6: lying with cover
        if len(self.imgHistory) != self.imgHistorySize:
            self.imgHistory = [img] * self.imgHistorySize
        else:
            self.imgHistory.pop(0)
            self.imgHistory.append(img)
        
        return -1

    def exitWarn(self, img):
        return self.isExiting

# A simple classifier for posture classification, which can be trained on the dataset. The input is a sequence of images, and the output is the posture category.
class PostureClassifier(nn.Module):
    def __init__(self):
        super(PostureClassifier, self).__init__()
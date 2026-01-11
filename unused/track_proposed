# This is a module for sleep monitoring system.
# It tracks human presence, detecting entry/exit events correctly even when occlusion by blanket occassionally happens.
# Occlusion should not trigger false exit events.
# TODO: implement 
# 1. Kalman Filter
# 2. Hidden Markov Model
# 3. Finite State Machine
# for tracking a human centroid that sometimes get occluded by blanket

# Kalman Filter for tracking human centroid given measurements (either detected position or occlusion state).
# The Kalman Filter should predict the next position and update based on measurements.
# We can assume a 2D position (x, y) for the human centroid.

import numpy as np

class KalmanOcclusionTracker:
    def __init__(self):
        # Initialize Kalman Filter parameters
        self.state = np.zeros((4, 1))  # [x, y, vx, vy]
        

    def predict(self):
        # Predict the next position of the human centroid
        pass

    def update(self, measurement):
        # Update the Kalman Filter with the new measurement
        pass

    def is_occluded(self):
        # Determine if the human centroid is currently occluded
        pass
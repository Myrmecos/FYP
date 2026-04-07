# keep a record of the heat blob dynamics 
# and output the final classification of each blob as human or residual heat for each frame at the end
# also, keep a record of the postures
# output the final classification of posture in each image seq
    # -1: unknown or unlabeled; 
    # 0: absence; 
    # 1: presence, unclassified; 
    # 2: standing; 
    # 3: sitting by bed; 
    # 4: sitting on bed; 
    # 5: lying w/o cover; 
    # 6: lying with cover

import yaml
import json


class PostProcessor:
    def __init__(self):
        self.blob_records = {}  # key: blob id_fixed, value: dict.
        # value of blob record dict:
        # {'start_frame': int, 'end_frame': int, 'temp_history': list of float, 'centroid_history': list of (x,y), 'is_residual': bool}
        self.posture_records = []  # key: frame_idx, value: posture
        self.posture_frame_idx = []

    def get_blobs(self, blobs, frame_idx):
        for idx, blob in enumerate(blobs):
            if blob.mean_temp is None or blob.centroid is None:
                continue
            if blob.id_fixed not in self.blob_records:
                self.blob_records[blob.id_fixed] = {
                    'start_frame': frame_idx,  # to be updated
                    'end_frame': -1,    # to be updated
                    'temp_history': [],
                    'centroid_history': [],
                    'is_residual': []
                }
            self.blob_records[blob.id_fixed]['temp_history'].append(blob.mean_temp)
            self.blob_records[blob.id_fixed]['centroid_history'].append(blob.centroid)
            self.blob_records[blob.id_fixed]['is_residual'].append(blob.is_residual)
            self.blob_records[blob.id_fixed]['end_frame'] = frame_idx
    
    def get_posture(self, posture, frame_idx):
        self.posture_records.append(posture)
        self.posture_frame_idx.append(frame_idx)
        return
    
    def postprocess_posture(self):
        # 1. make posture_records into a continuous list, pad the missing frame with either "absence" or "presence, unclassified" based on the presence of blobs in that frame
        
        # 2. use either state machine or hidden markov model to smooth the posture classification over time, and output the final posture classification for each frame
        pass

    def _markov_smoothing(self):
        # use the Viterbi algorithm to find the most likely sequence of postures given the observed posture_records and the transition probabilities between postures
        pass

    def _state_machine_smoothing(self):
        # define a state machine with states corresponding to the postures, and transitions based on the observed posture_records and the presence of blobs in each frame
        pass

    def output_results(self, output_path = "blob_records.json"):
        out = dict()
        # need to solve problem of TypeError: Object of type float16 is not JSON serializable
        print(self.blob_records)
        for key in self.blob_records:
            print(len(self.blob_records[key]['temp_history']), len(self.blob_records[key]['centroid_history']), len(self.blob_records[key]['is_residual']))
            record = self.blob_records[key]
            out[key] = {
                'start_frame': record['start_frame'],
                'end_frame': record['end_frame'],
                'temp_history': [float(t) for t in record['temp_history']],  # convert float16 to float
                'centroid_history': [tuple(map(float, c)) for c in record['centroid_history']],  # convert float16 to float
                'is_residual': record['is_residual']
            }
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=4)
            

if __name__ == "__main__":
    from dataset.dataset import ThermalDataset
    def test_postprocessor():
        # use a data entry as test: /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall5
        # 1. load the dataset

        # 2. initialize our system's components
        #   2.1. heatsource detection module: load the module
        #   2.1. posture detector module: load the model
        #   2.2. kalman tracker module: load the module
        #   2.4. postprocessor module: load the module

        postprocessor = PostProcessor()
        # 3. loop through each fraome
        #   3.1. detect heat source
        #   3.2. kalman tracker: record it in postprocessor
        #   3.3. posture detection if kalman shows presence; record it in postprocessor
        
        # 4. postprocess
        #   4.1. postprocess presence, make a list of presence data
        #   4.2. postprocess postures, make a list of posture data.

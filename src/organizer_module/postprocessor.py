# keep a record of the heat blob dynamics 
# and output the final classification of each blob as human or residual heat for each frame at the end


import yaml
import json


class PostProcessor:
    def __init__(self):
        self.blob_records = {}  # key: blob id_fixed, value: dict.
        # value of blob record dict:
        # {'start_frame': int, 'end_frame': int, 'temp_history': list of float, 'centroid_history': list of (x,y), 'is_residual': bool}

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
            
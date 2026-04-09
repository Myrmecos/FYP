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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset.dataset import ThermalDataset
    from heatsource_detection_module.extract import HeatSourceDetector
    from posture_detection_module.CNN_model import SimpleIRA_CNN
    from posture_detection_module.utils import remap_labels_simple, inverse_remap_labels_simple, label_to_text_simple, ThermalInvariantPreprocessor
    from organizer_module.track_kalman import Tracker
    from data_visualization_module.plot import DataVisualizer
    import torch
    import cv2

    dv = DataVisualizer()

    def visualize_frame(ira, blobs, posture_label, index):
        # visualize the blobs and posture label on the ira_highres image, and save the image to disk
        thermal_prepared = dv._prepare_thermal_for_colormap(ira)
        ira_color = cv2.applyColorMap(thermal_prepared, cv2.COLORMAP_JET)
        # rescale the image to 5 times
        scale_factor = 10
        ira_color = cv2.resize(ira_color, (ira_color.shape[1] * scale_factor, ira_color.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

        # plot the blobs on the image, use green circle for human and yellow circle for residual heat
        for blob in blobs:
            if blob.mean_temp is None or blob.centroid is None:
                continue
            color = (0, 255, 0) if blob.is_residual == False else (0, 255, 255)
            # cv2.circle(ira_color, (int(blob.centroid[1]), int(blob.centroid[0])), 10, color, 2)
            # draw bbox
            x_min, y_min, x_max, y_max = blob.get_state()
            x_min = int(x_min * scale_factor)
            y_min = int(y_min * scale_factor)
            x_max = int(x_max * scale_factor)
            y_max = int(y_max * scale_factor)
            cv2.rectangle(ira_color, (x_min, y_min), (x_max, y_max), color, 2)

        cv2.putText(ira_color, f'Frame: {index}| Posture: {posture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('ira', ira_color)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        # cv2.destroyAllWindows()

    data_name = "office1_0"

    def test_postprocessor():
        # use a data entry as test: /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall5
        # 1. load the dataset
        # dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/office1_0")
        dataset = ThermalDataset(f"/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/{data_name}")
        # 2. initialize our system's components ==================================
        #   2.1. heatsource detection module: load the module
        heat_detector = HeatSourceDetector()
        thermalinvariantpreprocessor = ThermalInvariantPreprocessor()
        #   2.1. posture detector module: load the model
        posture_detector_model = SimpleIRA_CNN()
        # load the pretrained weights for posture detection model
        posture_detector_model.load_state_dict(torch.load('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/posture_cnn_cross_env_env5.pth'))
        #   2.2. kalman tracker module: load the module
        tracker = Tracker()
        #   2.4. postprocessor module: load the module
        postprocessor = PostProcessor()
        
        gt_result_lst = []


        # 3. loop through each fraome
        for idx in range(0, len(dataset)):
            label = dataset.annotations_expanded[idx]
            gt_result_lst.append(label)
            ira_highres = dataset.get_ira_highres(idx)
            #   3.1. detect heat source
            thresh, mask = heat_detector.get_thresh_mask_otsu(ira_highres)
            # mask_processed = heat_detector.process_frame_mask(ira_highres, min_size=100)
            mask_individual = heat_detector.process_frame_connected_components(ira_highres, min_size=100)
            #   3.2. detect presence with kalman tracker
            tracker.update_blobs(mask_individual, ira_highres, heat_detector.get_unmasked_mean(ira_highres, mask), idx)
            postprocessor.get_blobs(tracker.blobs, idx)
            #   3.3. posture detection if kalman shows presence; record it in postprocessor
            hasHuman = False
            for blob in tracker.blobs:
                if blob.is_residual == False: # if it is classified as human
                    hasHuman = True
            if hasHuman:
                # clip and normalize the ira_highres image, and convert to tensor before feeding into the posture detection model
                ira_highres = thermalinvariantpreprocessor(ira_highres)
                posture = posture_detector_model(torch.tensor(ira_highres, dtype=torch.float32).unsqueeze(0)) # add batch and channel dimension
                posture_label = torch.argmax(posture, dim=1).item()
                # print("DEBUG: posture label: ", posture_label)
                posture_label = inverse_remap_labels_simple(posture_label)  # remap the posture label
                # print("DEBUG: inverse remap posture label: ", posture_label)
                postprocessor.get_posture(posture_label, idx)  # inverse remap the posture label
                posture_str = label_to_text_simple(posture_label)
            else:
                posture_label = 0
                posture_str = label_to_text_simple(posture_label)
                postprocessor.get_posture(0, idx)
            
            # visualize the result for this frame
            visualize_frame(ira_highres, tracker.blobs, posture_str, idx)
        
        print("DEBUG: posture records: ", len(postprocessor.posture_records), len(gt_result_lst))

        # draw the confusion matrix for posture classificatino result


        def confusion_matrix_draw(results, gt_result_lst):
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            y_true = gt_result_lst
            y_pred = [1 if r[1] == 'HUMAN' else 0 for r in results]

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Human', 'Human'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix for Posture Classification")
            plt.show()

        results = postprocessor.posture_records
        # write results and gt_result_lst to a json file for later analysis
        with open(f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/{data_name}.json', 'w') as f:
            json.dump({'results': results, 'gt_result_lst': gt_result_lst}, f, indent=4)






        # 4. postprocess
        #   4.1. postprocess presence, make a list of presence data
        # postprocessor.postprocess_presence()

        #   4.2. postprocess postures, make a list of posture data.
        # postprocessor.postprocess_posture()

        # 5. compare the smoothed posture classification result with the ground truth label, and visualize the comparison
        # 5. visualize: plot the posture classification result for each frame, and compare with the ground truth label
        #   5.1. visualize the presence detection result for each frame, and compare with the ground truth label
        #   5.2. visualize the blob classification result for each frame, and compare with the ground truth label

    test_postprocessor()
    
    def test_results():
        import matplotlib.pyplot as plt
        # load from json
        with open(f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/{data_name}.json', 'r') as f:
            data = json.load(f)
            results = data['results']
            gt_result_lst = data['gt_result_lst']

        print("DEBUG: results: ", len(results))
        print("DEBUG: gt_result_lst: ", len(gt_result_lst))

        plt.plot(results, label='Predicted Posture')
        plt.plot(gt_result_lst, label='Ground Truth Posture')
        plt.legend()
        plt.show()

        # when gt_results_lst is -1, we regard it as 0
        # when gt_results_lst is 1, it can match all the presence labels (2, 3, 4, 5, 6)
        # compute the accuracy
        correct = 0
        total = 0
        for i in range(len(gt_result_lst)):
            gt = gt_result_lst[i]
            pred = results[i]
            if gt == -1:
                gt = 0
            if gt == 1:
                if pred in [2, 3, 4, 5, 6]:
                    correct += 1
            else:
                if pred == gt:
                    correct += 1
            total += 1
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")
        #print acc and recall and F1 score for pose classification
        from sklearn.metrics import classification_report
        # resutls and gt_rersult_lst are both a list of pose labels
        # target_names=['Absence', 'Presence', 'Standing', 'Sitting by Bed', 'Sitting on Bed', 'Lying w/o Cover', 'Lying with Cover']
        print(classification_report(gt_result_lst, results))

    test_results()



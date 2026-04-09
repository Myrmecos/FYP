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
        """
        Use the Viterbi algorithm to find the most likely sequence of postures given the observed posture_records.
        Hidden states: 0 (absence), 1 (presence unclassified), 2 (standing), 3 (sitting by bed),
                       4 (sitting on bed), 5 (lying without cover), 6 (lying with cover)
        Observations: 0, 2, 3, 4, 5, 6 (no observation for state 1 - it acts as a "transition" state)
        """
        import numpy as np

        observations = np.array(self.posture_records)
        n_obs = len(observations)

        # Define states
        STATES = [0, 1, 2, 3, 4, 5, 6]
        OBSERVATIONS = [0, 2, 3, 4, 5, 6]
        STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
        OBS_TO_IDX = {o: i for i, o in enumerate(OBSERVATIONS)}

        # Load GT data to estimate transition and emission probabilities
        import json
        from pathlib import Path
        output_dir = Path(__file__).parent.parent.parent / "output"
        with open(output_dir / "office1_0.json", 'r') as f:
            data = json.load(f)
            gt = np.array(data['gt_result_lst'])
            obs_data = np.array(data['results'])

        # Estimate transition probabilities from GT
        # P(true_state_t | true_state_{t-1})
        transition_counts = np.ones((7, 7))  # Laplace smoothing
        for i in range(1, len(gt)):
            prev_state = gt[i - 1]
            curr_state = gt[i]
            transition_counts[prev_state, curr_state] += 1

        # Normalize to get transition probabilities
        trans_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        # Estimate emission probabilities from the data
        # P(observed_state | true_state)
        emission_counts = np.ones((7, 6))  # Laplace smoothing (6 possible observations)
        for i in range(n_obs):
            true_state = gt[i]
            obs = observations[i]
            if obs in OBSERVATIONS:
                obs_idx = OBS_TO_IDX[obs]
                emission_counts[true_state, obs_idx] += 1

        # Normalize to get emission probabilities
        emission_probs = emission_counts / emission_counts.sum(axis=1, keepdims=True)

        # Prior probabilities (initial state distribution)
        prior_counts = np.ones(7)
        prior_counts[gt[0]] += 1
        prior_probs = prior_counts / prior_counts.sum()

        # Viterbi algorithm
        n_states = 7
        T = n_obs

        # viterbi[n_states][T] = log probability
        viterbi = np.full((n_states, T), -np.inf)
        backpointer = np.full((n_states, T), -1, dtype=int)

        # Initialize
        for s in range(n_states):
            if observations[0] in OBSERVATIONS:
                obs_idx = OBS_TO_IDX[observations[0]]
                viterbi[s, 0] = np.log(prior_probs[s]) + np.log(emission_probs[s, obs_idx])
            else:
                viterbi[s, 0] = np.log(prior_probs[s])

        # Forward pass
        for t in range(1, T):
            obs = observations[t]
            obs_idx = OBS_TO_IDX[obs] if obs in OBSERVATIONS else -1
            for s in range(n_states):
                if obs_idx >= 0:
                    obs_prob = emission_probs[s, obs_idx]
                else:
                    obs_prob = 1.0  # State 1 (unclassified) doesn't emit
                if obs_prob <= 0:
                    obs_prob = 1e-10
                best_prev = np.argmax(viterbi[:, t - 1] + np.log(trans_probs[:, s]))
                viterbi[s, t] = viterbi[best_prev, t - 1] + np.log(trans_probs[best_prev, s]) + np.log(obs_prob)
                backpointer[s, t] = best_prev

        # Backtrack
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(viterbi[:, T - 1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[best_path[t + 1], t + 1]

        self.posture_records = best_path.tolist()

    def _state_machine_smoothing(self):
        """
        Apply rule-based state machine smoothing to posture records.
        Uses majority voting within a sliding window to decide the state for each frame.
        """
        import numpy as np
        WINDOW_SIZE = 5  # window for majority voting

        records = np.array(self.posture_records)
        n = len(records)
        smoothed = np.zeros(n, dtype=int)

        # For each frame, look at surrounding frames and take majority vote
        half = WINDOW_SIZE // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = records[start:end]
            # Count each unique value in the window
            values, counts = np.unique(window, return_counts=True)
            # Use argmax to get most common value (np.unique returns sorted values)
            majority_idx = np.argmax(counts)
            smoothed[i] = values[majority_idx]

        self.posture_records = smoothed.tolist()

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

    def visualize_frame(ira, blobs, posture_label, index, waittime = 0):
        # visualize the blobs and posture label on the ira_highres image, and save the image to disk
        thermal_prepared = dv._prepare_thermal_for_colormap(ira)
        ira_color = cv2.applyColorMap(thermal_prepared, cv2.COLORMAP_JET)
        # rescale the image to 5 times
        scale_factor = 10
        ira_color = cv2.resize(ira_color, (ira_color.shape[1] * scale_factor, ira_color.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

        corr = 0

        # plot the blobs on the image, use green circle for human and yellow circle for residual heat
        for blob in blobs:
            if blob.mean_temp is None or blob.centroid is None:
                continue
            color = (0, 255, 0) if blob.is_residual == False else (0, 255, 255)
            # cv2.circle(ira_color, (int(blob.centroid[1]), int(blob.centroid[0])), 10, color, 2)
            # mark the blob id on the image
            x_min, y_min, x_max, y_max = blob.get_state()
            cv2.putText(ira_color, f'ID: {blob.id_fixed}', (int(x_min*scale_factor), int(y_min*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(ira_color, f'corr: {blob.corr:.1f}', (int(x_min*scale_factor), int((y_min+15)*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # draw bbox
            corr = blob.corr
            x_min = int(x_min * scale_factor)
            y_min = int(y_min * scale_factor)
            x_max = int(x_max * scale_factor)
            y_max = int(y_max * scale_factor)
            cv2.rectangle(ira_color, (x_min, y_min), (x_max, y_max), color, 2)

        cv2.putText(ira_color, f'Frame: {index}| Posture: {posture_label}|Corr: {corr:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('ira', ira_color)
        key = cv2.waitKey(waittime)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        # cv2.destroyAllWindows()

    data_name = "office0_3"

    def test_postprocessor():
        # use a data entry as test: /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall5
        # 1. load the dataset
        # dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/office1_0")
        dataset = ThermalDataset(f"/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/{data_name}", noCam = True)
        # 2. initialize our system's components ==================================
        #   2.1. heatsource detection module: load the module
        heat_detector = HeatSourceDetector()
        thermalinvariantpreprocessor = ThermalInvariantPreprocessor()
        #   2.1. posture detector module: load the model
        posture_detector_model = SimpleIRA_CNN()
        # load the pretrained weights for posture detection model
        posture_detector_model.load_state_dict(torch.load('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/all_current_data.pth'))
        #   2.2. kalman tracker module: load the module
        tracker = Tracker()
        #   2.4. postprocessor module: load the module
        postprocessor = PostProcessor()
        
        gt_result_lst = []

        print("dataset length: ", len(dataset))

        waittime = 1
        # 3. loop through each fraome
        for idx in range(2000, len(dataset), 1):
            # if idx == 8720:
            #     waittime = 0
            label = dataset.annotations_expanded[idx]
            # if label == -1:
            #      continue
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
            visualize_frame(ira_highres, tracker.blobs, posture_str, idx, waittime)
        
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

    import matplotlib.pyplot as plt

    def test_pipeline_gridsearch():
        # load yaml content from /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml
        with open('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml', 'r') as f:
            config = yaml.safe_load(f)
            train_all = config['train_all'][:3]
        TEMP_DECREASE_THRESH = [-0.9, -0.92, -0.95]
        K_THRESH = [0.003, 0.004, 0.005]

        for temp_thresh in TEMP_DECREASE_THRESH:
            for k_thresh in K_THRESH:
                pred = []
                gt = []
                print(f"Testing with TEMP_DECREASE_THRESH: {temp_thresh}, K_THRESH: {k_thresh}")
                for folder in train_all:
                    gt_result_lst, pred_res_lst = _test_pipeline_gridsearch(folder)
                    pred.extend(pred_res_lst)
                    gt.extend(gt_result_lst)
                    print("accuracy: ", sum([1 if p == 1 and g == 1 else 0 for p, g in zip(pred_res_lst, gt_result_lst)]) / len(gt_result_lst))
                    print("present predicted as absent: ", sum([1 if p == 0 and g == 1 else 0 for p, g in zip(pred_res_lst, gt_result_lst)]) / len(gt_result_lst))
                    plt.plot(pred_res_lst, alpha = 0.5, label = "pred")
                    plt.plot(gt_result_lst, alpha = 0.5, label = "GT")
                    plt.legend()
                    plt.show()
                    print("absent predicted as present: ", sum([1 if p == 1 and g == 0 else 0 for p, g in zip(pred_res_lst, gt_result_lst)]) / len(gt_result_lst)) 
                print("===summary: present predicted as absent: ", sum([1 if p == 0 and g == 1 else 0 for p, g in zip(pred, gt)]) / len(gt))
                print("===summary: absent predicted as present: ", sum([1 if p == 1 and g == 0 else 0 for p, g in zip(pred, gt)]) / len(gt))
                print("=== summary: accuracy: ", sum([1 if p == 1 and g == 1 else 0 for p, g in zip(pred, gt)]) / len(gt)) 
    from tqdm import tqdm
    def _test_pipeline_gridsearch(dataset_path, HUMAN_ENV_THRESH = 4, TEMP_DECREASE_THRESH = -0.9, K_THRESH = 0.004):
        """
        Using grid search to find the best parameters for track_kalman
        """
        dataset = ThermalDataset(dataset_path, noCam = True)
        heat_detector = HeatSourceDetector()
        thermalinvariantpreprocessor = ThermalInvariantPreprocessor()
        # # posture_detector_model = SimpleIRA_CNN()
        # posture_detector_model.load_state_dict(torch.load('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/all_current_data.pth'))
        
        #   2.2. kalman tracker module: load the module
        tracker = Tracker()
        #   2.4. postprocessor module: load the module
        postprocessor = PostProcessor()
        
        gt_result_lst = []
        pred_res_lst = []
        print("dataset length: ", len(dataset))

        waittime = 1
        # 3. loop through each fraome
        for idx in tqdm(range(len(dataset))):
            # if idx == 8720:
            #     waittime = 0
            label = dataset.annotations_expanded[idx]
            label = 1 if label >= 1 else label
            # if label == -1:
            #      continue
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
            hasHuman = 0
            for blob in tracker.blobs:
                if blob.is_residual == False: # if it is classified as human
                    hasHuman = 1
            pred_res_lst.append(hasHuman)
            # visualize the result for this frame
            # visualize_frame(ira_highres, tracker.blobs, posture_str, idx, waittime)
        
        #print("DEBUG: posture records: ", len(postprocessor.posture_records), len(gt_result_lst))
        return gt_result_lst, pred_res_lst


    
    def test_results():
        import matplotlib.pyplot as plt
        from organizer_module.postprocessor import PostProcessor
        import numpy as np

        # load from json
        with open(f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/{data_name}.json', 'r') as f:
            data = json.load(f)
            results = data['results']
            gt_result_lst = data['gt_result_lst']

        print("DEBUG: results: ", len(results))
        print("DEBUG: gt_result_lst: ", len(gt_result_lst))

        # Apply Markov smoothing to the results
        pp = PostProcessor()
        pp.posture_records = results.copy()
        # pp._markov_smoothing()
        smoothed_results = pp.posture_records

        def compute_accuracy(pred, gt):
            correct = 0
            total = 0
            for i in range(len(gt)):
                g = gt[i]
                p = pred[i]
                if g == -1:
                    g = 0
                if g == 1:
                    if p in [2, 3, 4, 5, 6]:
                        correct += 1
                else:
                    if p == g:
                        correct += 1
                total += 1
            return correct / total

        accuracy_before = compute_accuracy(results, gt_result_lst)
        accuracy_after = compute_accuracy(smoothed_results, gt_result_lst)

        print("DEBUG: smoothed results: ", len(smoothed_results))
        plt.plot(smoothed_results, alpha = 0.5)
        # plt.plot(results, alpha = 0.5)
        # plt.plot(gt_result_lst, alpha = 0.5)
        plt.show()
        print(f"Accuracy before smoothing: {accuracy_before:.4f}")
        print(f"Accuracy after smoothing: {accuracy_after:.4f}")

        # count the accuracy
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


        # plot confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        # presence classification confusion matrix
        y_true = [1 if r in [2, 3, 4, 5, 6] else 0 for r in gt_result_lst]
        y_pred = [1 if r in [2, 3, 4, 5, 6] else 0 for r in results]
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix for Presence Classification:", cm)

        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Human', 'Human'])
        # disp.plot(cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix for Presence Classification (Binary)")
        # plt.show()

        # plot posture classification confusion matrix
        # remove ambiguous labels in gt_result_lst and results
        y_true = [r for r in gt_result_lst if r != -1 and r != 1]
        y_pred = [results[i] for i in range(len(results)) if gt_result_lst[i] != -1 and gt_result_lst[i] != 1]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 2, 3, 4, 5, 6])
        print("Confusion Matrix for Posture Classification (Detailed):", cm)

        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Human', 'Standing', 'Sitting by Bed', 'Sitting on Bed', 'Lying w/o Cover', 'Lying with Cover'])
        # disp.plot(cmap=plt.cm.Blues)
        # # rotate x label by 45 degree
        # plt.xticks(rotation=45)
        # plt.title("Confusion Matrix for Posture Classification (Detailed)")
        # plt.show()


    # test_results()
    # test_postprocessor()
    test_pipeline_gridsearch() 



import json
from pathlib import Path
import sys
sys.path.insert(0, "/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src")

# Load datasets using ThermalDataset (not the slow Aggregator)
from dataset.dataset import ThermalDataset
from posture_detection_module.utils import filter_dataset, train_model, ThermalNormalize
from posture_detection_module.CNN_model import SimpleIRA_CNN

from organizer_module.track_kalman import Tracker
from heat_patch_tracker_module.kalman_blob import KalmanBlob, mask_to_bbox
from residual_heat_detection_module.residual_detect import ResidualHeatDetector
from posture_detection_module.utils import remap_labels_simple, inverse_remap_labels_simple, label_to_text_simple, ThermalInvariantPreprocessor
from organizer_module.postprocessor import PostProcessor
from heatsource_detection_module.extract import HeatSourceDetector
# load the yaml file containing experiment setup
# path: /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml
def get_exp_training_testing_setup():

    import yaml
    with open('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml', 'r') as f:
        exp_setup = yaml.safe_load(f)
    
    
    return exp_setup['hyperparam_tune']


env_or_entry_name = "user5"
type = "user"

# env 0 cross-env hyperparameter grid search
res_path = f"posture_cnn_cross_{type}_{env_or_entry_name}_0417.yaml"

# load the datasets for training and testing
train_path_lst = get_exp_training_testing_setup()

kept_labels = [0, 2, 3, 4, 5, 6]
label_to_index = {
    0: 0,
    2: 1,
    3: 1, 
    4: 1,
    5: 1,
    6: 1
}


# run the entire pipeline on one data entry with visualizer
def test_inference(data_name, k = None, corr = None, v = None):
    # use a data entry as test: /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/hall5
    
    # 1. load the dataset ===============================================
    # dataset = ThermalDataset("/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/office1_0")
    dataset = ThermalDataset(f"/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data/{data_name}", noCam = True)
    print(f"dataset {data_name}; length:", len(dataset))


    # 2. initialize our system's components ==================================
    #   2.1. heatsource detection module: load the module
    heat_detector = HeatSourceDetector()
    # thermalinvariantpreprocessor = ThermalInvariantPreprocessor()
    #   2.1. posture detector module: load the model
    
    # load the pretrained weights for posture detection model
    # posture_detector_model.load_state_dict(torch.load('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/all_current_data.pth'))
    #   2.2. kalman tracker module: load the module
    if k is not None:
        tracker = Tracker(k_thresh = k, temp_decrease_thresh = corr, velocity_thresh = v)
    else:
        tracker = Tracker()
    #   2.4. postprocessor module: load the module
    postprocessor = PostProcessor()

    from tqdm import tqdm
    

    # 3. loop through each fraome ========================================
    #   3.1. prepare the ground truth label list
    gt_result_lst = []
    pred_result_lst = []
    waittime = 1
    visualization = False
    for idx in tqdm(range(0, len(dataset), 1)):
        # gt
        label = dataset.annotations_expanded[idx]
        presence = 1 if label > 0 else 0
        gt_result_lst.append(presence)

        # data
        ira_highres = dataset.get_ira_highres(idx)
        # make pixels less than 17 deg be 17 deg
        ira_highres[ira_highres < 18] = 18

        #   3.1. detect heat source
        thresh, mask = heat_detector.get_thresh_mask_otsu(ira_highres)
        mask_individual = heat_detector.process_frame_connected_components(ira_highres, min_size=100)

        #   3.2. detect presence with kalman tracker
        tracker.update_blobs(mask_individual, ira_highres, heat_detector.get_unmasked_mean(ira_highres, mask), idx)
        # postprocessor.get_blobs(tracker.blobs, idx)

        #   3.3. posture detection if kalman shows presence; record it in postprocessor
        hasHuman = False
        for blob in tracker.blobs:
            if blob.is_residual == False: # if it is classified as human
                hasHuman = True

        # if hasHuman:
        #     # clip and normalize the ira_highres image, and convert to tensor before feeding into the posture detection model
        #     ira_highres = thermalinvariantpreprocessor(ira_highres)
        #     posture = posture_detector_model(torch.tensor(ira_highres, dtype=torch.float32).unsqueeze(0)) # add batch and channel dimension
        #     posture_label = torch.argmax(posture, dim=1).item()
        #     # print("DEBUG: posture label: ", posture_label)
        #     posture_label = inverse_remap_labels_simple(posture_label)  # remap the posture label
        #     # print("DEBUG: inverse remap posture label: ", posture_label)
        #     postprocessor.get_posture(posture_label, idx)  # inverse remap the posture label
        #     posture_str = label_to_text_simple(posture_label)
        # else:
        #     posture_label = 0
        #     posture_str = label_to_text_simple(posture_label)
        #     postprocessor.get_posture(0, idx)
        pred_result_lst.append(1 if hasHuman else 0)


        # visualize the result for this frame
    # 4. save the results to /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/{data_name}.json
    with open(f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/gridsearch/{data_name}_{k}_{corr}_{v}.json', 'w') as f:
        json.dump({'results': pred_result_lst, 'gt_result_lst': gt_result_lst}, f, indent=4)
    return gt_result_lst, pred_result_lst
    # draw the confusion matrix for posture classificatino result
    # confusion_matrix_draw(postprocessor.posture_records, gt_result_lst)

def _vis_presence_absence(gt_lst, pred_lst):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 4))
    plt.plot(gt_lst, label='Ground Truth', alpha=0.7)
    plt.plot(pred_lst, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title('Presence vs Absence over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Presence (1) / Absence (0)')
    plt.show()

def vis_presence_absence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_lst = data['gt_result_lst']
    pred_lst = data['results']
    _vis_presence_absence(gt_lst, pred_lst)

def load_inference_result(entry_name, k = None, corr = None, v = None):
    entry_name = entry_name.split('/')[-1]
    json_path = f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/gridsearch/{entry_name}_{k}_{corr}_{v}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_lst = data['gt_result_lst']
    pred_lst = data['results']
    return gt_lst, pred_lst

def entry_exist(entry_name, k = None, corr = None, v = None):
    entry_name = entry_name.split('/')[-1]
    json_path = f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/gridsearch/{entry_name}_{k}_{corr}_{v}.json'
    return Path(json_path).exists()
# vis_presence_absence('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/output/gridsearch/hall5_0.5_0.5_0.5.json')
ks = [0.002, 0.003, 0.004, 0.005, 0.006]
corrs = [-0.65, -0.70, -0.75, -0.80, -0.85, -0.90, -0.95]
ks = [0.005]
corrs = [-0.075]
vs = [0.6, 0.8, 1.0, 1.2, 1.4]
for k in ks:
    for corr in corrs:
        for v in vs:
            preds = []
            actuals = []
            for train_path in train_path_lst:
                # gt, pred = test_inference(train_path.split('/')[-1], k=k, corr=corr, v=1)
                # gt, pred = load_inference_result(train_path, k = k, corr = corr, v = 1)
                if entry_exist(train_path, k = k, corr = corr, v = 1):
                    gt, pred = load_inference_result(train_path, k = k, corr = corr, v = v)
                else:
                    gt, pred = test_inference(train_path.split('/')[-1], k=k, corr=corr, v=v)
                preds.extend(pred)
                actuals.extend(gt)
                print("k value:", k, "corr value:", corr, "v value:", v)
                print("pred acc", sum([1 if p == a else 0 for p, a in zip(preds, actuals)]) / len(preds))
            print("SUMMARY: == k value:", k, "corr value:", corr, "v value:", v)
            print("SUMMARY: == pred acc", sum([1 if p == a else 0 for p, a in zip(preds, actuals)]) / len(preds))
            # print presence-absence confusion
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(actuals, preds)
            print("Confusion Matrix:")
            print(cm)
            print("====================")
        

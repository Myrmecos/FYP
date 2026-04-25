# Privacy-preserving Real-time Bed Monitoring and Exit Detection Using Low-resolution Thermal Array
This project leverages a low-resolution (80x62) thermal array to monitor bed and detect exit events. 

## Backgrounds and Motivations
1. Sleep is a vital physiological indicator. Monitoring sleep enables early detection of many abnormalities that are otherwise difficult to perceive.
2. Monitoring bed-side activities also enable real-time warning of activities related to fall risks, such as bed exit events.
3. Clinical monitoring solutions require specialized personnel, venues, and equipment, making it unsuitable for long-term, continuous self-monitoring at home.

Therefore, we hope to provide a novel solution that is affordable, non-intrusive, privacy-preserving, and human-sensitive, to transparently monitor sleep and report relevant statistics.


## Sensor Choice
We select a low-cost thermal array (IRA) sensor that produces heat maps, IRA properties:

1. Sensitve to human presence by detecting the thermal signals.
2. Functional in complete darkness.
3. Privacy preserving due to low-resolution.
4. Relatively low-cost.

Which make it a suitable modality for home-based monitoring.

## Results
Presence detection achieves over 95% accuracy. Posture detection achieves over 89% accuracy, and overall accuracy exceeds 89% for all held-out users and environments. This demonstrates the system’s robustness and effectiveness.

## Quick Start
1. put all your data into `data/` (one data entry is uploaded, but to preserve the subject's privacy, the RGB images are removed)
2. put all your weights into `weights/`
3. run the demo in `src/experiments/demo.ipynb`

Note: you can download demo data and weights from:
`https://1drv.ms/u/c/7ece577cc7f199eb/IQAGVsJ4uNjRTIanGfKYznbLAa9ByaxHTIW6fNsWwrPSa34?e=rGhg08`

## Project Structure
1. `config/`: configuration files
2. `data/`: example data, running output, input
3. `presence_detector/`: contains different modules
4. `scripts/`: scripts for tuning
5. `test/`: test examples

```
├── data_collection_module
│   ├── __init__.py
│   ├── display_data.py    # for displaying collected data
│   ├── PC_receiver_new.py # main data collection code
│   ├── senxor/    # IRA library
│   └── utils.py   # utilities for data collection
├── data_visualization_module
│   ├── imgs_to_vid.py # convert a data entry to a mp4 video. for visualization.
│   ├── plot_postures.ipynb # plot ira image of each posture
│   ├── plot.py   # plot the statitics across time
│   └── visualize_detection_result.ipynb # visualize the presence detection results
├── dataset 
│   ├── __init__.py
│   └── dataset.py # dataset class
├── exit_detection_module
│   ├── __init__.py
│   └── exit_detect.py   # detection of exit event based on DL. No longer used.
├── experiments
│   ├── acc_recall_f1_precision.py. # calculate the statistics of output
│   ├── CNN_5class.py # training 5-class CNN for posture module
│   ├── CNN_baseline.py # baseline 6-class CNN to show non-trivial nature
│   ├── demo.ipynb # demo for visualization
│   ├── experiment_final_model_gridsearch.py # gridsearch to optimize params
│   ├── experiment_final_model_training.ipynb # train models (jupyter notebook)
│   ├── experiment_final_model_training.py # train models (python)
│   ├── experiment_heldout.ipynb # heldout environment and user; exit detection experiment
│   ├── full_pipeline_eval.py # inferencing an entire data entry
│   └── LSTM_baseline.py # baseline LSTM model.
├── heat_patch_tracker_module
│   ├── __init__.py
│   ├── blob.py # naive heatpatch tracker
│   ├── convert.py # bounding box and heat patch mask conversion utility
│   ├── kalman_blob.py # tracker leveraging Kalman Filter
│   ├── particle_blob.py # tracker leveraging particle filter
│   └── utils.py # other utility functions
├── heatsource_detection_module
│   ├── __init__.py
│   ├── extract.ipynb # testing/visualizing extraction code
│   └── extract.py # code for extracting heat patches
├── organizer_module # equivalent to residual detection module + integration module
│   ├── __init__.py
│   ├── postprocessor.py # for tracking and performing smoothing
│   ├── track_kalman_noPostProcess.py # does not perform history-based blob type checking
│   ├── track_kalman_noResDet.py # does not perform split-based residual determination
│   ├── track_kalman_test.py # test function for kalman filter tracking
│   ├── track_kalman.py # performing residual detection based on kalman filter tracking result
│   ├── track_no_filter.py # perform residual detection based on naive tracking result
│   ├── track_particle.py # perform residual detection based on particle filter result
│   └── visualize.py # visualize plotted trajectory
├── posture_detection_module
│   ├── __init__.py
│   ├── CNN_model.py # CNN model architecture (for classifying postures)
│   ├── main_cross_validation.py # cross env/user validation of model's classification capability
│   ├── posture.ipynb # training CNN model
│   └── utils.py
└── residual_heat_detection_module # currently not used
    ├── __init__.py
    ├── residual_detect.ipynb # checking and visualizing residual
    └── residual_detect.py # checking residual
```

# Data collection
1. `PC_receiver_new.py`
   1. `self.thermalHighresCollector = Collector("/dev/cu.usbmodem1301")`: fill in the correct dev name
   2. `def __init__(self, save_path="presence_detection_workspace/data", target_file="hall0")`: fill in the correct path name
2. `sketch_nov7a.ino`
   1. `const char* ssid = "JH300-254ACC";`
   2. `const char* udpAddress = "192.168.11.61";`
3. connect to JH300
4. `python ./presence_detection_workspace/src/data_collection_module/PC_receiver_new.py`

# Heatsource detection
1. Otsu's method: maximizing inter-group difference between foreground and background
2. cleaning: get connected components, remove components with small area

# Tracking
1. for every blob, try to find match in detected heat regions (consider max-flow, min-cut problem): COMPLETED
2. for new hot regions with no match to previous, if it is hot enough (larger than bg by 3), create a new blob: DONE
3. TODO: use particle filter to track and predict human motion.

# Motion detection
4. TODO: classify postures (features + SVM)
   1. can experiment with different models

# Update
1. renamed organizer_module to heat_patch_tracker_module
2. renamed tracking_module to organizer_module



# Project Structure
1. `config/`: configuration files
2. `data/`: example data, running output, input
3. `presence_detector/`: contains different modules
4. `scripts/`: scripts for tuning
5. `test/`: test examples

# Data collection
1. `PC_receiver_new.py`
   1. `self.thermalHighresCollector = Collector("/dev/cu.usbmodem1301")`: fill in the correct dev name
   2. `def __init__(self, save_path="presence_detection_workspace/data", target_file="hall0")`: correct it
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

# TODO
TODO: add experiment where we carry a bottle of hot water
TODO: benchmark inference speed

# Goal: 
0. A window, first select where bed is, then start inferencing
1. when human present, track him with a cross
   1. when covered by blanket, mark blanket
2. when human ready to stand up, issue warning
3. when human leaves, do not mark residual
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
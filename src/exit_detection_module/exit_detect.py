




if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # this module detects the exit segments
    # 1. load the dataset and the annotations
    from dataset.dataset import ThermalDataset
    # load all data from the yaml file /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml
    import yaml

    exit_segments = []
    with open('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml', 'r') as f:
        config = yaml.safe_load(f)
        data_path = config['train_all'][0]
    for data_path in config['train_all']: # test on the first 3 videos
        dataset = ThermalDataset(data_path)
        print(dataset.exit_frame_indices)
        for exit_indices in dataset.exit_frame_indices:
            exit_frames = [dataset.get_ira_highres(i) for i in range(exit_indices[0], exit_indices[1]+1)] # take 30 frames before and after the exit frame as the exit segment
            exit_segments.append(exit_frames)
    
    # in total, we have 24 exit segments
    # we want to train a model to detect the exit segments
    # using other frames as negative samples, and frames around the exit segments as hard negative samples
    # we can use the heat source detection and tracking results to extract features for each frame
    # we can use the features to train a model to detect the exit segments

    print(len(exit_segments))
    import matplotlib.pyplot as plt
    for i in exit_segments[4]:
    
        plt.imshow(i)
        plt.show()

    

from pathlib import Path
import sys
sys.path.insert(0, "/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src")

# Load datasets using ThermalDataset (not the slow Aggregator)
from dataset.dataset import ThermalDataset
from posture_detection_module.utils import filter_dataset, train_model, ThermalNormalize
from posture_detection_module.CNN_model import SimpleIRA_CNN
from torch.utils.data import Subset, ConcatDataset
from sklearn.model_selection import train_test_split
import torch

# load the yaml file containing experiment setup
# path: /Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml
def get_exp_training_testing_setup(env_name, type):
    ret = {
        'train': [],
        'test': []
    }
    import yaml
    with open('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml', 'r') as f:
        exp_setup = yaml.safe_load(f)
    
    envs = [f"{type}{i}" for i in range(0, 6)]
    for env in envs:
        if env != env_name:
            ret['train'].extend(exp_setup[env])
        else:
            ret['test'].extend(exp_setup[env])
    
    return ret


env_name = "user5"
type = "user"

# env 0 cross-env model training
model_path = f"posture_cnn_cross_{type}_{env_name}_0417.pth"
# load the datasets for training and testing
train_test = get_exp_training_testing_setup(env_name, type=type)
train_path_lst = train_test['train']
test_path_lst = train_test['test']

kept_labels = [2, 3, 4, 5, 6]
label_to_index = {label: idx for idx, label in enumerate(kept_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# ==================================================
# Build list of filtered datasets from train paths
train_datasets = []
all_entries = []
all_labels = []

for path in train_path_lst:
    ds = ThermalDataset(path)
    ds_filtered = filter_dataset(ds, label_to_index)
    train_datasets.append(ds_filtered)
    print(f"Loaded {path.split('/')[-1]}: {len(ds_filtered)} samples")
    for i in range(len(ds_filtered)):
        all_entries.append((ds_filtered, i))
        all_labels.append(ds_filtered.dataset.annotations_expanded[ds_filtered.indices[i]])

print(f"\nTotal training samples: {len(all_entries)}")

test_datasets = []
for path in test_path_lst:
    ds = ThermalDataset(path)
    ds_filtered = filter_dataset(ds, label_to_index)
    test_datasets.append(ds_filtered)
    
print(f"Loaded {path.split('/')[-1]}: {len(ds_filtered)} samples")

# ==================================================

# Stratified train/val split
train_entries, val_entries = train_test_split(
    all_entries, test_size=0.2, stratify=all_labels, random_state=42
)

# Create dataloaders
def make_loader(entries, batch_size=32, augment=False):
    indices_by_ds = {}
    for ds, i in entries:
        if ds not in indices_by_ds:
            indices_by_ds[ds] = []
        indices_by_ds[ds].append(i)
    subsets = [Subset(ds, indices) for ds, indices in indices_by_ds.items()]
    concat = ConcatDataset(subsets) if len(subsets) > 1 else subsets[0]
    return torch.utils.data.DataLoader(
        ThermalNormalize(concat, augment=augment), batch_size=batch_size, shuffle=True
    )

train_loader = make_loader(train_entries, augment=True)
val_loader = make_loader(val_entries, augment=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Create and train model
model = SimpleIRA_CNN(num_classes=len(kept_labels))
train_model(model, train_loader, val_loader, label_to_index, num_epochs=15, learning_rate=1e-3, save_path=model_path)

# Save model
# save_path = f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/posture_cnn_cross_env_{env_name}_0417.pth'
# torch.save(model.state_dict(), save_path)
# print(f"Model saved to {save_path}")
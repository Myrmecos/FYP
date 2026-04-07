# CNN posture classification — SimpleIRA_CNN + per-frame percentile normalization
# Cross-environment generalization via adaptive normalization (no augmentation)

import sys
sys.path.append('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src/')
sys.path.append('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src/posture_detection_module')

from dataset.dataset import ThermalDataset
from utils import remap_labels, train_model, test_model, filter_dataset, ThermalNormalize
from CNN_model import SimpleIRA_CNN
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch

# classes: 
# -1: unknown or unlabeled; 
# 0: absence; 
# 1: presence, unclassified; 
# 2: standing; 
# 3: sitting by bed; 
# 4: sitting on bed; 
# 5: lying w/o cover; 
# 6: lying with cover

# ==============================================================================
# 1. Load dataset
# ==============================================================================
data_root = '/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data'
dataset = ThermalDataset(f'{data_root}/office0_2')

kept_labels = [2, 3, 4, 5, 6]
label_to_index = {label: idx for idx, label in enumerate(kept_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
print("Label mapping:", label_to_index)

label_counts = Counter(dataset.annotations_expanded)
print("Label distribution:", label_counts)


# ==============================================================================
# 2. Stratified train/val/test split
# ==============================================================================
valid_indices = [i for i, label in enumerate(dataset.annotations_expanded) if label in kept_labels]
labels_valid = [dataset.annotations_expanded[i] for i in valid_indices]

train_val_idx, test_idx = train_test_split(
    valid_indices, test_size=0.20, stratify=labels_valid, random_state=42
)
labels_train = [dataset.annotations_expanded[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.20, stratify=labels_train, random_state=42
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


# ==============================================================================
# 3. Datasets with per-frame percentile normalization (no augmentation)
# ==============================================================================
train_dataset = ThermalNormalize(Subset(dataset, train_idx), augment=False)
val_dataset   = ThermalNormalize(Subset(dataset, val_idx),   augment=False)
test_dataset  = ThermalNormalize(Subset(dataset, test_idx),  augment=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False)

for batch in train_dataloader:
    thermal = batch[0]['ira_highres']
    print(f"Thermal shape: {thermal.shape}, range: [{thermal.min():.3f}, {thermal.max():.3f}]")
    break


# ==============================================================================
# 4. Train
# ==============================================================================
model = SimpleIRA_CNN(num_classes=len(label_to_index))
train_model(model, train_dataloader, val_dataloader, label_to_index, num_epochs=2, learning_rate=1e-3)

torch.save(model.state_dict(),
    f'{data_root}/../weights/posture_cnn.pth')


# ==============================================================================
# 5. Evaluate — same environment
# ==============================================================================
test_model(model, test_dataloader, label_to_index)


# ==============================================================================
# 6. Evaluate — cross environment (hall5)
# ==============================================================================
hall5_ds = ThermalDataset(f'{data_root}/hall5')
hall5_ds = filter_dataset(hall5_ds, label_to_index)
hall5_dataset = ThermalNormalize(hall5_ds, augment=False)
hall5_loader = torch.utils.data.DataLoader(hall5_dataset, batch_size=32, shuffle=False)

print("\nCross-environment (hall5):")
test_model(model, hall5_loader, label_to_index)

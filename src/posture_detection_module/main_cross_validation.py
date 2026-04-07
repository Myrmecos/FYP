# CNN posture classification — Cross-user and Cross-environment validation
# Based on config/exp_setup.yaml

import sys
sys.path.append('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src/')
sys.path.append('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/src/posture_detection_module')

from dataset.dataset import ThermalDataset
from utils import remap_labels, train_model, test_model, filter_dataset, ThermalNormalize
from CNN_model import SimpleIRA_CNN
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset
import torch
import yaml
from pathlib import Path

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
# 1. Load experiment configuration
# ==============================================================================
config_path = '/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_root = '/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/data'
kept_labels = [2, 3, 4, 5, 6]
label_to_index = {label: idx for idx, label in enumerate(kept_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
print("Label mapping:", label_to_index)

# Resolve environment paths
envs = {}
for env_key, paths in config.items():
    if env_key.startswith('env'):
        envs[env_key] = [Path(p) for p in paths]

# Resolve user paths
users = {}
for user_key, paths in config.items():
    if user_key.startswith('user'):
        users[user_key] = [Path(p) for p in paths]

train_envs = config.get('train_envs', [])
test_envs = config.get('test_envs', [])
train_users = config.get('train_users', [])
test_users = config.get('test_users', [])

print(f"\nTrain environments: {train_envs}")
print(f"Test environments: {test_envs}")
print(f"Train users: {train_users}")
print(f"Test users: {test_users}")


# ==============================================================================
# 2. Helper: load and combine datasets for given envs or users
# ==============================================================================
def load_datasets_for_envs(env_keys, label_map):
    """Load all datasets for given environment keys. Returns list of (env_key, dataset) tuples."""
    result = []
    for env_key in env_keys:
        if env_key in envs:
            for path in envs[env_key]:
                if path.exists():
                    ds = ThermalDataset(str(path))
                    ds = filter_dataset(ds, label_map)
                    result.append((env_key, ds))
                    print(f"  Loaded {env_key}: {path.name} ({len(ds)} samples)")
    return result


def load_datasets_for_users(user_keys, label_map):
    """Load all datasets for given user keys. Returns list of (user_key, dataset) tuples."""
    result = []
    for user_key in user_keys:
        if user_key in users:
            for path in users[user_key]:
                if path.exists():
                    ds = ThermalDataset(str(path))
                    ds = filter_dataset(ds, label_map)
                    result.append((user_key, ds))
                    print(f"  Loaded {user_key}: {path.name} ({len(ds)} samples)")
    return result


def get_label(ds, i):
    """Get label from a dataset or Subset, handling both cases."""
    if isinstance(ds, Subset):
        return ds.dataset.annotations_expanded[ds.indices[i]]
    return ds.annotations_expanded[i]


def combine_and_split(dataset_tuples, test_size=0.2, val_size=0.2, random_state=42):
    """Combine multiple (key, dataset) tuples and create stratified train/val/test splits.
    Returns (key, dataset, index) tuples for each split."""
    # Flatten all indices with their labels
    all_entries = []
    all_labels = []
    for key, ds in dataset_tuples:
        for i in range(len(ds)):
            all_entries.append((key, ds, i))
            all_labels.append(get_label(ds, i))

    valid_mask = [label in kept_labels for label in all_labels]
    valid_entries = [e for e, m in zip(all_entries, valid_mask) if m]
    valid_labels = [l for l, m in zip(all_labels, valid_mask) if m]

    # Train/val/test split
    train_val_entries, test_entries = train_test_split(
        valid_entries, test_size=test_size, stratify=valid_labels, random_state=random_state
    )
    train_labels = [all_labels[all_entries.index(e)] for e in train_val_entries]
    train_entries, val_entries = train_test_split(
        train_val_entries, test_size=val_size / (1 - test_size), stratify=train_labels, random_state=random_state
    )

    return train_entries, val_entries, test_entries


def create_dataloaders_from_entries(entries, batch_size=32):
    """Create a DataLoader from a list of (key, dataset, index) tuples."""
    indices_by_ds = {}
    for key, ds, i in entries:
        if ds not in indices_by_ds:
            indices_by_ds[ds] = []
        indices_by_ds[ds].append(i)

    wrapped_subsets = [Subset(ds, indices) for ds, indices in indices_by_ds.items()]
    if wrapped_subsets:
        concat = ConcatDataset(wrapped_subsets) if len(wrapped_subsets) > 1 else wrapped_subsets[0]
        return torch.utils.data.DataLoader(
            ThermalNormalize(concat, augment=False), batch_size=batch_size, shuffle=False
        )
    return None


# ==============================================================================
# 3. Experiment modes
# ==============================================================================
def run_within_env_user(env_key, user_key):
    """Baseline: train and test on same environment and user."""
    print(f"\n{'='*60}")
    print(f"Experiment: Within env/user — env={env_key}, user={user_key}")
    print(f"{'='*60}")

    dataset_tuples = load_datasets_for_envs([env_key], label_to_index)
    dataset_tuples = [(k, ds) for k, ds in dataset_tuples if len(ds) > 0]
    if not dataset_tuples:
        dataset_tuples = load_datasets_for_users([user_key], label_to_index)

    if not dataset_tuples:
        print(f"  No data found for {env_key}/{user_key}")
        return None

    train_entries, val_entries, test_entries = combine_and_split(dataset_tuples)
    train_loader = create_dataloaders_from_entries(train_entries)
    val_loader = create_dataloaders_from_entries(val_entries)
    test_loader = create_dataloaders_from_entries(test_entries)

    model = SimpleIRA_CNN(num_classes=len(label_to_index))
    train_model(model, train_loader, val_loader, label_to_index, num_epochs=2, learning_rate=1e-3)

    results = test_model(model, test_loader, label_to_index)
    return results


def run_cross_env(train_envs, test_envs):
    """Cross-environment: train on train_envs, test on test_envs."""
    print(f"\n{'='*60}")
    print(f"Experiment: Cross-environment")
    print(f"  Train: {train_envs}")
    print(f"  Test: {test_envs}")
    print(f"{'='*60}")

    train_tuples = load_datasets_for_envs(train_envs, label_to_index)
    test_tuples = load_datasets_for_envs(test_envs, label_to_index)

    if not train_tuples or not test_tuples:
        print("  Missing train or test data")
        return None

    train_entries, val_entries, _ = combine_and_split(train_tuples)
    train_loader = create_dataloaders_from_entries(train_entries)
    val_loader = create_dataloaders_from_entries(val_entries)

    print(f"  Train loader size: {len(train_loader.dataset)} samples")
    print(f"  Val loader size: {len(val_loader.dataset)} samples")

    # Test on each test env separately
    for env_key in test_envs:
        env_test_entries = [(env_key, ds, i) for env_key_ds, ds in test_tuples if env_key_ds == env_key for i in range(len(ds))]
        if not env_test_entries:
            continue
        test_loader = create_dataloaders_from_entries(env_test_entries)

        print(f"\n  Test on {env_key}:")
        model = SimpleIRA_CNN(num_classes=len(label_to_index))
        torch.save(model.state_dict(),
            f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/posture_cnn_cross_env_{env_key}.pth')
        train_model(model, train_loader, val_loader, label_to_index, num_epochs=2, learning_rate=1e-3)
        results = test_model(model, test_loader, label_to_index)


def run_cross_user(train_users, test_users):
    """Cross-user: train on train_users, test on test_users."""
    print(f"\n{'='*60}")
    print(f"Experiment: Cross-user")
    print(f"  Train: {train_users}")
    print(f"  Test: {test_users}")
    print(f"{'='*60}")

    train_tuples = load_datasets_for_users(train_users, label_to_index)
    test_tuples = load_datasets_for_users(test_users, label_to_index)

    if not train_tuples or not test_tuples:
        print("  Missing train or test data")
        return None

    train_entries, val_entries, _ = combine_and_split(train_tuples)
    train_loader = create_dataloaders_from_entries(train_entries)
    val_loader = create_dataloaders_from_entries(val_entries)

    # Test on each test user separately
    for user_key in test_users:
        user_test_entries = [(user_key, ds, i) for user_key_ds, ds in test_tuples if user_key_ds == user_key for i in range(len(ds))]
        if not user_test_entries:
            continue
        test_loader = create_dataloaders_from_entries(user_test_entries)

        print(f"\n  Test on {user_key}:")
        model = SimpleIRA_CNN(num_classes=len(label_to_index))
        torch.save(model.state_dict(),
            f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/posture_cnn_cross_user_{user_key}.pth')
        train_model(model, train_loader, val_loader, label_to_index, num_epochs=2, learning_rate=1e-3)
        results = test_model(model, test_loader, label_to_index)


def run_cross_user_and_env(train_envs, test_envs, train_users, test_users):
    """Cross-user-and-env: most challenging generalization."""
    print(f"\n{'='*60}")
    print(f"Experiment: Cross-user-and-env")
    print(f"  Train: {train_envs} x {train_users}")
    print(f"  Test: {test_envs} x {test_users}")
    print(f"{'='*60}")

    # Combine env and user datasets for train
    train_tuples = load_datasets_for_envs(train_envs, label_to_index)
    train_user_tuples = load_datasets_for_users(train_users, label_to_index)
    all_train_tuples = train_tuples + [t for t in train_user_tuples if t[0] not in [t2[0] for t2 in train_tuples]]

    test_tuples = load_datasets_for_envs(test_envs, label_to_index)
    test_user_tuples = load_datasets_for_users(test_users, label_to_index)
    all_test_tuples = test_tuples + [t for t in test_user_tuples if t[0] not in [t2[0] for t2 in test_tuples]]

    if not all_train_tuples or not all_test_tuples:
        print("  Missing train or test data")
        return None

    train_entries, val_entries, _ = combine_and_split(all_train_tuples)
    train_loader = create_dataloaders_from_entries(train_entries)
    val_loader = create_dataloaders_from_entries(val_entries)

    test_entries = [(k, ds, i) for k, ds in all_test_tuples for i in range(len(ds))]
    test_loader = create_dataloaders_from_entries(test_entries)

    model = SimpleIRA_CNN(num_classes=len(label_to_index))
    torch.save(model.state_dict(),
        f'/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/weights/posture_cnn_cross_user_env.pth')
    train_model(model, train_loader, val_loader, label_to_index, num_epochs=2, learning_rate=1e-3)
    results = test_model(model, test_loader, label_to_index)
    return results


# ==============================================================================
# 4. Run experiments
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running cross-validation experiments")
    print("="*60)

    # Run cross-environment experiment
    run_cross_env(train_envs, test_envs)

    # Run cross-user experiment
    # run_cross_user(train_users, test_users)

    # Run cross-user-and-env experiment
    # run_cross_user_and_env(train_envs, test_envs, train_users, test_users)

    print("\n" + "="*60)
    print("All experiments complete")
    print("="*60)
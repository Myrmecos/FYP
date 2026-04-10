

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import pickle
import os
from dataset.dataset import ThermalDataset


def extract_features(ira_frame):
    """Extract simple thermal features from an IRA frame."""
    feat = []
    # Basic statistics
    feat.append(np.mean(ira_frame))
    feat.append(np.std(ira_frame))
    feat.append(np.max(ira_frame))
    feat.append(np.min(ira_frame))
    feat.append(np.percentile(ira_frame, 25))
    feat.append(np.percentile(ira_frame, 75))
    # Count hot pixels (above threshold)
    threshold = np.mean(ira_frame) + np.std(ira_frame)
    feat.append(np.sum(ira_frame > threshold))
    feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.array(feat)


class ExitDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.frames[idx]), torch.FloatTensor([self.labels[idx]])


class SimpleExitDetector(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


def load_data(data_paths):
    """Load exit segments and negative samples."""
    positive_frames = []
    negative_frames = []

    for data_path in data_paths:
        dataset = ThermalDataset(data_path)

        # Get positive samples (exit segments)
        for exit_indices in dataset.exit_frame_indices:
            for i in range(exit_indices[0], exit_indices[1] + 1):
                ira = dataset.get_ira_highres(i)
                feat = extract_features(ira)
                positive_frames.append(feat)

        # Get negative samples (frames far from exits)
        all_indices = set(range(len(dataset)))
        exit_ranges = set()
        for exit_indices in dataset.exit_frame_indices:
            for i in range(exit_indices[0], exit_indices[1] + 1):
                exit_ranges.add(i)

        # Sample negative frames (not in exit regions)
        negative_indices = list(all_indices - exit_ranges)
        np.random.seed(42)
        np.random.shuffle(negative_indices)
        for i in negative_indices[:len(positive_frames)]:  # Balance classes
            ira = dataset.get_ira_highres(i)
            feat = extract_features(ira)
            negative_frames.append(feat)

    # Combine and create labels
    all_frames = positive_frames + negative_frames
    all_labels = [1.0] * len(positive_frames) + [0.0] * len(negative_frames)

    return np.array(all_frames), np.array(all_labels)


def train(data_paths, epochs=20, lr=0.001, batch_size=16):
    """Train the exit detector."""
    X, y = load_data(data_paths)
    print(f"Total samples: {len(X)} (positive: {sum(y)}, negative: {len(y) - sum(y)})")

    dataset = ExitDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleExitDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")

    return model


def evaluate(model, data_paths):
    """Evaluate the model on test data."""
    X, y = load_data(data_paths)
    dataset = ExitDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = (model(X_batch) > 0.5).float()
            correct += (pred == y_batch).sum().item()

    accuracy = correct / len(X)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    with open('/Users/entomophile/Desktop/FYP/entry_exit_detection/presence_detection_workspace/config/exp_setup.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Use first 3 videos for quick training
    train_paths = config['train_all'][:3]
    print(f"Training on {len(train_paths)} datasets")

    # Train
    model = train(train_paths, epochs=20)

    # Evaluate on same data (replace with test paths for real evaluation)
    print("\nEvaluation:")
    evaluate(model, train_paths)

    # Save model
    save_path = Path(__file__).parent / "exit_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

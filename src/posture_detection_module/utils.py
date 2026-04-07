
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "heatsource_detection_module"))
from heatsource_detection_module.extract import normalize_temperature
from tqdm import tqdm

# -1: unknown or unlabeled; 
# 0: absence; 
# 1: presence, unclassified; 
# 2: standing; 
# 3: sitting by bed; 
# 4: sitting on bed; 
# 5: lying w/o cover; 
# 6: lying with cover
kept_labels = [2, 3, 4, 5, 6]
label_to_index = {label: idx for idx, label in enumerate(kept_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
label_to_text_dict = {
    -1: "unknown",
    0: "absence",
    1: "presence, unclassified",
    2: "standing",
    3: "sitting by bed",
    4: "sitting on bed",
    5: "lying w/o cover",
    6: "lying with cover"
}
TEMP_MIN = 10.0
TEMP_MAX = 40.0


# map label to index
def remap_labels(labels, label_to_index=label_to_index, device='cpu'):
    labels_tensor = torch.as_tensor(labels)
    if labels_tensor.dim() == 0 or labels_tensor.numel() == 1:
        print("DEBUG: int(labels): ", int(labels))
        return torch.tensor(label_to_index[int(labels)], dtype=torch.long, device=device)
    return torch.tensor(
        [label_to_index[int(label)] for label in labels],
        dtype=torch.long,
        device=device,
    )

def remap_labels_simple(label, label_to_index=label_to_index):
    return label_to_index[label]

def inverse_remap_labels_simple(indices, index_to_label=index_to_label):
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    if np.isscalar(indices):
        return index_to_label[int(indices)]
    return [index_to_label[int(idx)] for idx in indices]

def label_to_text_simple(labels, label_to_text_dict=label_to_text_dict):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if np.isscalar(labels):
        return label_to_text_dict[int(labels)]
    return [label_to_text_dict[int(label)] for label in labels]


class ThermalInvariantPreprocessor:
    """Per-frame percentile-based normalization.

    Adapts to each frame's own distribution rather than fixed [10, 40]°C bounds.
    Robust to cross-environment temperature shifts.
    """

    def __init__(self, clip_percentiles=(1, 99)):
        self.p_low, self.p_high = clip_percentiles

    def __call__(self, thermal):
        p_low, p_high = np.percentile(thermal, (self.p_low, self.p_high))
        clipped = np.clip(thermal, p_low, p_high)
        range_val = p_high - p_low + 1e-8
        normalized = (clipped - p_low) / range_val
        return normalized.astype(np.float32)


class ThermalAugment:
    """Applies domain-generalizing augmentations to thermal images.

    Augmentations are designed to simulate cross-environment temperature variations
    and partial occlusions without introducing environment-specific biases.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, thermal):
        # thermal is a numpy array of shape (H, W) in [0, 1]
        if np.random.rand() < self.p:
            thermal = self._histogram_distortion(thermal)
        if np.random.rand() < self.p:
            thermal = self._random_blackout(thermal)
        if np.random.rand() < self.p:
            thermal = self._gaussian_noise(thermal)
        if np.random.rand() < self.p:
            thermal = self._random_affine(thermal)
        return thermal

    def _histogram_distortion(self, img):
        # Simulate different ambient temperatures by scaling the histogram
        scale = np.random.uniform(0.7, 1.3)
        shifted = img * scale
        return np.clip(shifted, 0.0, 1.0)

    def _random_blackout(self, img, n_holes=2, length=8):
        h, w = img.shape
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            img = img.copy()
            img[y1:y2, x1:x2] = 0.0
        return img

    def _gaussian_noise(self, img, sigma=0.02):
        noise = np.random.randn(*img.shape) * sigma
        return np.clip(img + noise, 0.0, 1.0)

    def _random_affine(self, img, angle=10, translate=(0.05, 0.05)):
        h, w = img.shape
        center = (w // 2, h // 2)
        angle_deg = np.random.uniform(-angle, angle)
        scale = np.random.uniform(0.95, 1.05)
        M = self._get_affine_matrix(center, angle_deg, scale, translate, (w, h))
        import cv2
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    def _get_affine_matrix(self, center, angle_deg, scale, translate, shape):
        import cv2
        angle_rad = angle_deg * np.pi / 180
        alpha = scale * np.cos(angle_rad)
        beta = scale * np.sin(angle_rad)
        tx = translate[0] * shape[0]
        ty = translate[1] * shape[1]
        M = np.array([
            [alpha, beta, (1 - alpha) * center[0] - beta * center[1] + tx],
            [-beta, alpha, beta * center[0] + (1 - alpha) * center[1] + ty],
        ], dtype=np.float32)
        return M


class ThermalNormalize(torch.utils.data.Dataset):
    """Dataset wrapper that applies per-frame percentile normalization + augmentation."""

    def __init__(self, subset, augment=False):
        self.subset = subset
        self.preprocessor = ThermalInvariantPreprocessor()
        self.augmenter = ThermalAugment() if augment else None

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data_dict, label = self.subset[idx]
        ira = data_dict['ira_highres']
        if isinstance(ira, np.ndarray):
            ira_norm = self.preprocessor(ira)
        else:
            ira_norm = self.preprocessor(ira.cpu().numpy())
        if self.augmenter is not None:
            ira_norm = self.augmenter(ira_norm)
        data_dict['ira_highres'] = torch.from_numpy(ira_norm).float()
        return data_dict, label


def train_model(model, train_dataloader, val_dataloader, label_to_index = label_to_index, num_epochs=10, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            thermal = batch[0]['ira_highres'].to(device=device, dtype=torch.float32)
            labels = remap_labels(batch[1], label_to_index, device)
            
            outputs = model(thermal)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                thermal = batch[0]['ira_highres'].to(device=device, dtype=torch.float32)
                labels = remap_labels(batch[1], label_to_index, device)
                
                outputs = model(thermal)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")


def test_model(model, test_dataloader, label_to_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_dataloader:
            thermal = batch[0]['ira_highres'].to(device=device, dtype=torch.float32)
            labels = remap_labels(batch[1], label_to_index, device)
            
            outputs = model(thermal)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


def filter_dataset(dataset, label_to_index):
    valid_indices = [
        i for i, label in enumerate(dataset.annotations_expanded)
        if label in label_to_index
    ]
    
    filtered_dataset = torch.utils.data.Subset(dataset, valid_indices)
    
    return filtered_dataset
#print("job started successfully")
"""
Multi-Modal Garbage Classification Script

This script trains and evaluates a deep learning model that combines:
- Image features extracted by EfficientNetV2-S
- Text features extracted from file names using DistilBERT

Dataset Structure:
  /garbage_data/
    Train/
    Validation/
    Test/

Each folder contains subfolders for each class with image files.

Usage:
- Update dataset paths for your local or HPC environment.
- This code is for HPCs which have nodes whitout internet connection. For nodes with internet connection loading models are easier (You can also use this code but make sure you are using right paths)
- For Google Colab, mount Google Drive and adjust paths accordingly.
  (Colab-specific setup code omitted here)
"""

# ========================================
# Setup & Imports
# ========================================
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True

# ========================================
# Image Data Loading
# ========================================

data_dir = "/home/alika/projects/def-csimo/alika/ENEL645AS2/garbage_data"
train_dir = os.path.join(data_dir, "Train")
val_dir = os.path.join(data_dir, "Validation")
test_dir = os.path.join(data_dir, "Test")

#  Define image transforms for train, validation, and test
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load image datasets
datasets = {
    "train": datasets.ImageFolder(train_dir, transform=transform["train"]),
    "val": datasets.ImageFolder(val_dir, transform=transform["val"]),
    "test": datasets.ImageFolder(test_dir, transform=transform["test"]),
}

# Define DataLoaders
workers = 12
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=50, shuffle=True, num_workers=workers),
    "val": DataLoader(datasets["val"], batch_size=50, shuffle=False, num_workers=workers),
    "test": DataLoader(datasets["test"], batch_size=50, shuffle=False, num_workers=workers),
}

# ========================================
# Text Data Loading
# ========================================

# ========================================
# Dataset & DataLoader Setup for Text
# ========================================

class CustomDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    Tokenizes texts using a given tokenizer and returns input_ids, attention_mask, and label tensors.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load DistilBERT tokenizer (cached locally)
tokenizer = DistilBertTokenizer.from_pretrained('/home/alika/projects/def-csimo/alika/ENEL645AS2/distilbert_cache') 

# Tokenize data
max_len = 24

# ========================================
# Combined Multi-Modal Dataset & DataLoaders
# ========================================

class MultiModalDataset(Dataset):
    """
    Returns aligned (image, text, label) items derived from ImageFolder's `samples` list,
    ensuring labels are exactly the indices used by ImageFolder (0..num_classes-1)
    and the text is parsed from the image filename.
    """
    def __init__(self, image_dataset, tokenizer, max_len):
        self.image_dataset = image_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Cache paths and labels from ImageFolder to preserve exact ordering
        # Each entry in samples is (path, class_index)
        self.samples = list(image_dataset.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image and target using ImageFolder's transform/path
        image, target = self.image_dataset[idx]  # target is already an int in [0..num_classes-1]
        img_path, _ = self.samples[idx]
        # Build text from filename (without extension), remove digits, replace underscores with spaces
        base = os.path.basename(img_path)
        file_name_no_ext, _ = os.path.splitext(base)
        text_raw = file_name_no_ext.replace('_', ' ')
        text_clean = re.sub(r'\d+', '', text_raw)

        enc = self.tokenizer.encode_plus(
            text_clean,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'image': image,
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(int(target), dtype=torch.long),
            'text': text_clean,
            'path': img_path,
        }

train_dataset = MultiModalDataset(datasets["train"], tokenizer, max_len)
val_dataset   = MultiModalDataset(datasets["val"],   tokenizer, max_len)
test_dataset  = MultiModalDataset(datasets["test"],  tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True,  num_workers=workers)
val_loader   = DataLoader(val_dataset,   batch_size=50, shuffle=False, num_workers=workers)
test_loader  = DataLoader(test_dataset,  batch_size=50, shuffle=False, num_workers=workers)

# ========================================
# Model Loading & Initialization
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained image model (EfficientNetV2-S) without classifier
imgmodel = models.efficientnet_v2_s(pretrained=False)
imgmodel.load_state_dict(torch.load('/home/alika/projects/def-csimo/alika/ENEL645AS2/efficientnet_v2_s-dd5fe13b.pth'))

# Load pretrained DistilBERT model for text
txtmodel = DistilBertModel.from_pretrained('/home/alika/projects/def-csimo/alika/ENEL645AS2/distilbert_cache')

# Freeze pretrained parameters to use models as fixed feature extractors
for param in imgmodel.features.parameters():
    param.requires_grad = False
for param in txtmodel.parameters():
    param.requires_grad = False

img_features = imgmodel.classifier[1].in_features
txt_features = txtmodel.config.hidden_size
num_classes = len(datasets["train"].classes)

print(f"Image feature size: {img_features}")
print(f"Text feature size: {txt_features}")

# ========================================
# Multi-Modal Model Definition
# ========================================
class MultiInputModel(nn.Module):
    """
    Multi-modal model that combines image and text features,
    projects them to a common space, concatenates, and classifies.
    """
    def __init__(self, text_model, image_model, text_output_dim, image_output_dim, num_classes):
        super(MultiInputModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.image_model.classifier=nn.Identity()
        # Freeze pretrained models initially
        for param in self.image_model.parameters():
            param.requires_grad = False

        for param in self.text_model.parameters():
            param.requires_grad = False

        for param in list(self.image_model.features[-2:].parameters()):
            param.requires_grad = True
        for param in list(self.text_model.transformer.layer[-2:].parameters()):
            param.requires_grad = True

        # Projection layers to a common embedding dimension
        hidden_dim = max(text_output_dim, image_output_dim)
        self.text_proj = nn.Linear(text_output_dim, hidden_dim)
        self.image_proj = nn.Linear(image_output_dim, hidden_dim)
        self.dropout_text = nn.Dropout(p=0.5)
        self.dropout_image = nn.Dropout(p=0.5)
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_image = nn.LayerNorm(hidden_dim)
        # Final classification layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids,attention_mask, image):
       # Process image features
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        image_features = self.image_proj(image_features)
        image_features = self.dropout_image(self.ln_image(image_features))
        # Process text features (use CLS token)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_features)
        text_features = self.dropout_text(self.ln_text(text_features))
        # Fuse image and text features
        combined = torch.cat((image_features, text_features), dim=1)
        output=self.fc(combined)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi_input_model = MultiInputModel(
    txtmodel, imgmodel, txt_features, img_features, num_classes).to(device)

# ========================================
# Training Function
# ========================================

# GPU Parallelization
if torch.cuda.device_count() > 1 :
    print(f"using {torch.cuda.device_count()} GPUs!")
    multi_input_model=nn.DataParallel(multi_input_model)
    print("Parallel!")
multi_input_model = multi_input_model.to(device)
print(device)
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Train with aligned multi-modal batches from a single loader per phase."""
    best_acc = 0.0
    patience = 3
    patience_counter = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch in loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # --- Sanity check: labels must be in [0, num_classes-1] ---
                if labels.min().item() < 0 or labels.max().item() >= num_classes:
                    # Move small debug info to CPU for safe printing
                    bad_min = int(labels.min().detach().cpu().item())
                    bad_max = int(labels.max().detach().cpu().item())
                    print(f"[LabelError] Found labels outside range [0,{num_classes-1}]: min={bad_min}, max={bad_max}")
                    print("Batch paths (first 5):", [batch['path'][i] for i in range(min(5, len(batch['path'])) )])
                    raise ValueError("Labels out of range for CrossEntropyLoss")

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(input_ids, attention_mask, images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Save in a way that's safe with DataParallel
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state, "best_model_combined.pth")
                patience_counter = 0
            elif phase == "val":
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                return model

    print(f"Best validation accuracy: {best_acc:.4f}")
    return model

# ========================================
# Testing Function
# ========================================

def test_model(model, test_loader):
    """Evaluate the trained model on test data."""
    # Load in a way that's safe with DataParallel
    state_dict = torch.load("best_model_combined.pth", map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ========================================
# Run Training and Testing
# ========================================
optimizer = torch.optim.AdamW(
    multi_input_model.parameters(),
    lr=2e-5,
    weight_decay=1e-4,
)
criterion = nn.CrossEntropyLoss()
model = train_model(multi_input_model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
test_model(model, test_loader)

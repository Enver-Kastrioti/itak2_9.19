#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path

# Solve MKL library conflict
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 20
DEFAULT_MAX_LENGTH = 1000
VOCAB_SIZE = 21
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_DATASET_JSON = SCRIPT_DIR / "Lecrk_dataset.json"

# Amino acid mapping
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
aa_to_idx['X'] = len(AMINO_ACIDS)

# --- Model Definition (Copied from predict.py) ---
class OneHotLightweightCNN(nn.Module):
    """Multi-scale CNN model using One-Hot encoding"""
    def __init__(self, vocab_size=21, num_classes=2, max_length=1000):
        super(OneHotLightweightCNN, self).__init__()
        
        # Config 1: (4x21, 4x1, 16x1)
        self.branch1_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=4, padding=2)
        self.branch1_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)
        self.branch1_conv3 = nn.Conv1d(128, 256, kernel_size=16, padding=8)
        
        # Config 2: (12x21, 8x1, 4x1)
        self.branch2_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=12, padding=6)
        self.branch2_conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.branch2_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)
        
        # Config 3: (16x21, 4x1, 4x1)
        self.branch3_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=16, padding=8)
        self.branch3_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)
        self.branch3_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)
        
        # Batch Normalization
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn1_3 = nn.BatchNorm1d(256)
        
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(256)
        
        self.bn3_1 = nn.BatchNorm1d(64)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.bn3_3 = nn.BatchNorm1d(256)
        
        # Fusion
        self.fusion_conv = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm1d(512)
        
        # Deep features
        self.deep_conv1 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.deep_conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.deep_bn1 = nn.BatchNorm1d(256)
        self.deep_bn2 = nn.BatchNorm1d(128)
        
        # Attention
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        # Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, max_length, vocab_size)
        x = x.transpose(1, 2)
        
        # Branch 1
        branch1 = torch.relu(self.bn1_1(self.branch1_conv1(x)))
        branch1 = self.dropout1(branch1)
        branch1 = torch.relu(self.bn1_2(self.branch1_conv2(branch1)))
        branch1 = self.dropout2(branch1)
        branch1 = torch.relu(self.bn1_3(self.branch1_conv3(branch1)))
        branch1 = self.dropout1(branch1)
        
        # Branch 2
        branch2 = torch.relu(self.bn2_1(self.branch2_conv1(x)))
        branch2 = self.dropout1(branch2)
        branch2 = torch.relu(self.bn2_2(self.branch2_conv2(branch2)))
        branch2 = self.dropout2(branch2)
        branch2 = torch.relu(self.bn2_3(self.branch2_conv3(branch2)))
        branch2 = self.dropout1(branch2)
        
        # Branch 3
        branch3 = torch.relu(self.bn3_1(self.branch3_conv1(x)))
        branch3 = self.dropout1(branch3)
        branch3 = torch.relu(self.bn3_2(self.branch3_conv2(branch3)))
        branch3 = self.dropout2(branch3)
        branch3 = torch.relu(self.bn3_3(self.branch3_conv3(branch3)))
        branch3 = self.dropout1(branch3)
        
        # Fusion
        x = torch.cat([branch1, branch2, branch3], dim=1)
        x = torch.relu(self.fusion_bn(self.fusion_conv(x)))
        x = self.dropout3(x)
        
        # Deep
        x = torch.relu(self.deep_bn1(self.deep_conv1(x)))
        x = self.dropout2(x)
        x = torch.relu(self.deep_bn2(self.deep_conv2(x)))
        x = self.dropout1(x)
        
        # Attention
        x = x.transpose(1, 2)
        x_att, _ = self.attention(x, x, x)
        x = x + x_att
        x = x.transpose(1, 2)
        
        # Pooling
        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        x = x.squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        return x

# --- Data Handling ---
def sequence_to_onehot(sequence, max_length=1000, vocab_size=21):
    onehot = np.zeros((max_length, vocab_size), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_length]):
        aa_idx = aa_to_idx.get(aa, aa_to_idx['X'])
        onehot[i, aa_idx] = 1.0
    return onehot

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, sample_weights, max_length=1000):
        self.sequences = sequences
        self.labels = labels
        self.sample_weights = sample_weights
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        weight = self.sample_weights[idx]
        onehot = sequence_to_onehot(seq, self.max_length)
        return (
            torch.tensor(onehot),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )

def load_data_from_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
    data = json.loads(json_path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid dataset JSON format: {json_path}")
    sequences = []
    labels = []
    weights = []
    for item in data:
        if not isinstance(item, dict):
            continue
        seq = item.get('sequence')
        tag = item.get('tag')
        if not seq or not tag:
            continue
        tag_u = str(tag).strip().upper()
        if tag_u == 'YES':
            label = 1
            weight = 1.0
        elif tag_u == 'NO':
            label = 0
            weight = 1.0
        elif tag_u == 'GREAT':
            label = 1
            weight = None
        else:
            raise ValueError(f"Unknown tag value: {tag}")
        sequences.append(str(seq))
        labels.append(label)
        weights.append(weight)
    if not sequences:
        raise ValueError(f"No valid samples found in dataset JSON: {json_path}")
    combined = list(zip(sequences, labels, weights))
    random.shuffle(combined)
    sequences, labels, weights = zip(*combined)
    return list(sequences), list(labels), list(weights)

# --- Training ---
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sequences, labels, weights = load_data_from_json(args.dataset_json)
    weights = [args.great_weight if w is None else float(w) for w in weights]
    
    n = len(sequences)
    if n < 3:
        raise ValueError("Dataset must contain at least 3 samples for train/val/test split.")
    train_end = max(1, min(n - 2, int(n * 0.8)))
    val_end = max(train_end + 1, min(n - 1, int(n * 0.9)))
    train_seqs, val_seqs, test_seqs = sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]
    train_labels, val_labels, test_labels = labels[:train_end], labels[train_end:val_end], labels[val_end:]
    train_weights, val_weights, test_weights = weights[:train_end], weights[train_end:val_end], weights[val_end:]
    
    train_dataset = ProteinDataset(train_seqs, train_labels, train_weights, args.max_length)
    val_dataset = ProteinDataset(val_seqs, val_labels, val_weights, args.max_length)
    test_dataset = ProteinDataset(test_seqs, test_labels, test_weights, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(
        f"Training on {len(train_dataset)} samples, "
        f"Validating on {len(val_dataset)} samples, "
        f"Testing on {len(test_dataset)} samples."
    )
    
    model = OneHotLightweightCNN(VOCAB_SIZE, 2, args.max_length).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_acc = -1.0
    
    model_path = Path(args.model_name)
    if not model_path.suffix:
        model_path = model_path.with_suffix(".pth")
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets, sample_weights in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            sample_weights = sample_weights.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_per_sample = criterion(outputs, targets)
            loss = (loss_per_sample * sample_weights).mean()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets, _sample_weights in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss_per_sample = criterion(outputs, targets)
                val_loss += loss_per_sample.mean().item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    if len(test_dataset) > 0 and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets, _sample_weights in test_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss_per_sample = criterion(outputs, targets)
                test_loss += loss_per_sample.mean().item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f"Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--great-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)

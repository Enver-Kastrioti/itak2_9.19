#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Hot Encoding Multi-scale CNN Protein Prediction Script
- Multi-scale OneHotLightweightCNN model based on train_onehot_cnn.py
- Supports transcription factor prediction for single FASTA files
- Outputs protein prediction probability table
- Supports custom confidence threshold
- Adapted for train_data8.28 multi-scale CNN architecture
"""

import os
import sys
import argparse

# Solve MKL library conflict - must be set before importing torch
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from Bio import SeqIO
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



# Amino acid to number mapping
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
aa_to_idx['X'] = len(AMINO_ACIDS)  # Unknown amino acid

class OneHotLightweightCNN(nn.Module):
    """Multi-scale CNN model using One-Hot encoding
    
    Uses three different configurations of multi-layer convolution kernel combinations to achieve multi-scale feature extraction through parallel structures:
    Config 1: (4x21, 4x1, 16x1)
    Config 2: (12x21, 8x1, 4x1) 
    Config 3: (16x21, 4x1, 4x1)
    """
    def __init__(self, vocab_size=21, num_classes=2, max_length=1000):
        super(OneHotLightweightCNN, self).__init__()
        
        # Input: (batch_size, max_length, vocab_size)
        # Needs to be transposed to: (batch_size, vocab_size, max_length) for Conv1d
        
        # Config 1: (4x21, 4x1, 16x1) - Small scale fine features
        self.branch1_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=4, padding=2)  # 4x21 equivalent
        self.branch1_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)        # 4x1
        self.branch1_conv3 = nn.Conv1d(128, 256, kernel_size=16, padding=8)      # 16x1
        
        # Config 2: (12x21, 8x1, 4x1) - Medium scale features
        self.branch2_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=12, padding=6) # 12x21 equivalent
        self.branch2_conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)        # 8x1
        self.branch2_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)       # 4x1
        
        # Config 3: (16x21, 4x1, 4x1) - Large scale global features
        self.branch3_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=16, padding=8) # 16x21 equivalent
        self.branch3_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)        # 4x1
        self.branch3_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)       # 4x1
        
        # Batch Normalization layers - Branch 1
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn1_3 = nn.BatchNorm1d(256)
        
        # Batch Normalization layers - Branch 2
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(256)
        
        # Batch Normalization layers - Branch 3
        self.bn3_1 = nn.BatchNorm1d(64)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.bn3_3 = nn.BatchNorm1d(256)
        
        # Feature fusion layer
        self.fusion_conv = nn.Conv1d(768, 512, kernel_size=3, padding=1)  # 256*3=768
        self.fusion_bn = nn.BatchNorm1d(512)
        
        # Deep feature extraction
        self.deep_conv1 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.deep_conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.deep_bn1 = nn.BatchNorm1d(256)
        self.deep_bn2 = nn.BatchNorm1d(128)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        # Pooling layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dropout layers (enhance regularization)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)
        
        # Classification layer (multi-scale feature classification)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # 128*2=256 (avg+max pooling)
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
        x = x.transpose(1, 2)  # (batch_size, vocab_size, max_length)
        
        # Branch 1: (4x21, 4x1, 16x1) - Small scale fine features
        branch1 = torch.relu(self.bn1_1(self.branch1_conv1(x)))
        branch1 = self.dropout1(branch1)
        branch1 = torch.relu(self.bn1_2(self.branch1_conv2(branch1)))
        branch1 = self.dropout2(branch1)
        branch1 = torch.relu(self.bn1_3(self.branch1_conv3(branch1)))
        branch1 = self.dropout1(branch1)
        
        # Branch 2: (12x21, 8x1, 4x1) - Medium scale features
        branch2 = torch.relu(self.bn2_1(self.branch2_conv1(x)))
        branch2 = self.dropout1(branch2)
        branch2 = torch.relu(self.bn2_2(self.branch2_conv2(branch2)))
        branch2 = self.dropout2(branch2)
        branch2 = torch.relu(self.bn2_3(self.branch2_conv3(branch2)))
        branch2 = self.dropout1(branch2)
        
        # Branch 3: (16x21, 4x1, 4x1) - Large scale global features
        branch3 = torch.relu(self.bn3_1(self.branch3_conv1(x)))
        branch3 = self.dropout1(branch3)
        branch3 = torch.relu(self.bn3_2(self.branch3_conv2(branch3)))
        branch3 = self.dropout2(branch3)
        branch3 = torch.relu(self.bn3_3(self.branch3_conv3(branch3)))
        branch3 = self.dropout1(branch3)
        
        # Multi-scale feature fusion
        x = torch.cat([branch1, branch2, branch3], dim=1)  # (batch_size, 768, max_length)
        x = torch.relu(self.fusion_bn(self.fusion_conv(x)))
        x = self.dropout3(x)
        
        # Deep feature extraction
        x = torch.relu(self.deep_bn1(self.deep_conv1(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.deep_bn2(self.deep_conv2(x)))
        x = self.dropout1(x)
        
        # Attention mechanism
        x = x.transpose(1, 2)  # (batch_size, max_length, 128)
        x_att, _ = self.attention(x, x, x)
        x = x + x_att  # Residual connection
        x = x.transpose(1, 2)  # (batch_size, 128, max_length)
        
        # Dual pooling
        x_avg = self.global_avg_pool(x)  # (batch_size, 128, 1)
        x_max = self.global_max_pool(x)  # (batch_size, 128, 1)
        x = torch.cat([x_avg, x_max], dim=1)  # (batch_size, 256, 1)
        x = x.squeeze(-1)  # (batch_size, 256)
        
        # Classification
        x = self.classifier(x)
        return x

def parse_fasta(fasta_file):
    """Parse FASTA file"""
    sequences = []
    current_header = None
    current_sequence = ""
    
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append({
                        'header': current_header,
                        'sequence': current_sequence
                    })
                current_header = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
        
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': current_sequence
            })
    
    return sequences



def sequence_to_onehot(sequence, max_length=1000, vocab_size=21):
    """Convert amino acid sequence to one-hot encoding"""
    # Initialize one-hot matrix
    onehot = np.zeros((max_length, vocab_size), dtype=np.float32)
    
    # Encode sequence
    for i, aa in enumerate(sequence[:max_length]):
        aa_idx = aa_to_idx.get(aa, aa_to_idx['X'])
        onehot[i, aa_idx] = 1.0
    
    # For sequences shorter than max_length, remaining positions stay zero (padding)
    return onehot

def predict_sequences_batch(model, sequences, device, max_length=1000, batch_size=16):
    """Batch predict sequences"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Encode sequences as One-Hot
            encoded_batch = []
            for seq_data in batch_sequences:
                onehot_encoded = sequence_to_onehot(seq_data['sequence'], max_length)
                encoded_batch.append(onehot_encoded)
            
            # Convert to tensor
            batch_tensor = torch.tensor(np.stack(encoded_batch), dtype=torch.float).to(device)
            
            # Predict
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Extract probabilities
            for j, seq_data in enumerate(batch_sequences):
                non_tf_prob = probabilities[j][0].item()
                tf_prob = probabilities[j][1].item()
                
                all_predictions.append({
                    'header': seq_data['header'],
                    'tf_probability': tf_prob,
                    'non_tf_probability': non_tf_prob
                })
    
    return all_predictions



def get_sequence_fragments(sequence, window_size=1000, step_size=200, mode='fast'):
    """Split sequence into fragments based on mode"""
    length = len(sequence)
    if length <= window_size:
        return [sequence]
    
    fragments = []
    
    if mode == 'full':
        # Sliding window with overlap (Step size 200)
        for start in range(0, length, step_size):
            end = start + window_size
            # If the window goes beyond the end, take from start to the end and stop
            if end >= length:
                fragments.append(sequence[start:])
                break
            fragments.append(sequence[start:end])
            
    elif mode == 'fast':
        # Fast mode: Contiguous 1000aa chunks + one final chunk from the end
        # Example: 1300 -> [0:1000], [300:1300]
        # Example: 3200 -> [0:1000], [1000:2000], [2000:3000], [2200:3200]
        
        # 1. Extract non-overlapping chunks
        num_chunks = length // window_size
        for i in range(num_chunks):
            start = i * window_size
            end = start + window_size
            fragments.append(sequence[start:end])
            
        # 2. Handle remainder by taking the last window_size characters
        if length % window_size != 0:
            fragments.append(sequence[-window_size:])
            
    return fragments

def predict_fasta(fasta_file=None, model=None, device=None, threshold=0.1, max_length=1000, batch_size=16, sequences=None, mode='fast'):
    if sequences is None:
        print(f"\nProcessing file: {fasta_file}")
        sequences = parse_fasta(fasta_file)
        print(f"Number of sequences: {len(sequences)}")
    if not sequences:
        print("Warning: No valid sequences found")
        return [], []
    
    # Pre-process sequences: Split long sequences into fragments
    expanded_sequences = []
    fragment_counts = []  # Record number of fragments for each original sequence
    
    split_seq_count = 0 # Count of sequences that were split (length > window_size)
    total_fragments = 0 # Total number of fragments generated
    
    print(f"Sequence Splitting Mode: {mode}")
    
    for seq_data in sequences:
        fragments = get_sequence_fragments(seq_data['sequence'], window_size=max_length, step_size=200, mode=mode)
        num_frags = len(fragments)
        
        if num_frags > 1:
            split_seq_count += 1
        
        total_fragments += num_frags
        fragment_counts.append(num_frags)
        for frag in fragments:
            expanded_sequences.append({
                'header': seq_data['header'],
                'sequence': frag
            })
            
    print(f"Sequence Splitting Statistics:")
    print(f"  - Original Sequences: {len(sequences)}")
    print(f"  - Sequences Split (> {max_length}aa): {split_seq_count}")
    print(f"  - Total Fragments Generated: {total_fragments}")
    print(f"  - Average Fragments per Sequence: {total_fragments/len(sequences):.2f}")

    # Predict on all fragments
    raw_predictions = predict_sequences_batch(model, expanded_sequences, device, max_length, batch_size)
    
    results = []
    tf_headers = []
    tf_count = 0
    
    # Aggregate results
    current_idx = 0
    for count in fragment_counts:
        # Get predictions for all fragments of this protein
        frag_preds = raw_predictions[current_idx : current_idx + count]
        current_idx += count
        
        # Strategy: Max Pooling
        # If any fragment is predicted as TF (prob >= threshold), the protein is TF.
        # We take the fragment with the highest TF probability as the representative.
        best_pred = max(frag_preds, key=lambda x: x['tf_probability'])
        
        tf_prob = best_pred['tf_probability']
        predicted_class = 'TF' if tf_prob >= threshold else 'Non-TF'
        confidence = max(tf_prob, best_pred['non_tf_probability'])
        
        result = {
            'header': best_pred['header'],
            'predicted_class': predicted_class,
            'tf_probability': tf_prob,
            'non_tf_probability': best_pred['non_tf_probability'],
            'confidence': confidence
        }
        results.append(result)
        if predicted_class == 'TF':
            tf_count += 1
            tf_headers.append(best_pred['header'])
            
    print(f"Prediction results: {tf_count} TF, {len(results) - tf_count} Non-TF")
    print(f"TF ratio: {tf_count/len(results)*100:.2f}%")
    return results, tf_headers

def save_predictions(predictions, output_file):
    """Save prediction results to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Header', 'Predicted_Class', 'TF_Probability', 'Non_TF_Probability', 'Confidence'])
        for pred in predictions:
            writer.writerow([
                pred['header'],
                pred['predicted_class'],
                f"{pred['tf_probability']:.4f}",
                f"{pred['non_tf_probability']:.4f}",
                f"{pred['confidence']:.4f}"
            ])
    print(f"Prediction results saved to: {output_file}")

def save_tf_predictions(predictions, output_file):
    """Save TF prediction results to CSV file"""
    tf_predictions = [p for p in predictions if p['predicted_class'] == 'TF']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Header', 'TF_Probability', 'Confidence'])
        for pred in tf_predictions:
            writer.writerow([
                pred['header'],
                f"{pred['tf_probability']:.4f}",
                f"{pred['confidence']:.4f}"
            ])
    print(f"TF prediction results saved to: {output_file}")

def load_model(model_path, device, max_length=1000):
    """Load trained multi-scale CNN model"""
    print(f"Loading multi-scale CNN model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        vocab_size = config.get('vocab_size', 21)
        max_length = config.get('max_length', 1000)
        num_classes = config.get('num_classes', 2)
    else:
        # Default configuration
        vocab_size = 21
        num_classes = 2
    
    # Create multi-scale CNN model
    model = OneHotLightweightCNN(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_length=max_length
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Multi-scale CNN model loaded successfully:")
    print(f"  - Model Architecture: Three-branch parallel multi-scale CNN")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Vocabulary Size: {vocab_size}")
    print(f"  - Max Sequence Length: {max_length}")
    print(f"  - Number of Classes: {num_classes}")
    if 'val_acc' in checkpoint:
        print(f"  - Validation Accuracy: {checkpoint['val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  - Training Epochs: {checkpoint['epoch'] + 1}")
    
    return model, max_length

def main():
    # Get absolute path of script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, 'model.pth')
    
    parser = argparse.ArgumentParser(description='Multi-scale CNN Model Protein Prediction')
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file path')
    parser.add_argument('--model', type=str, 
                        default=default_model_path,
                        help=f'Model file path (default: {default_model_path})')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='TF prediction threshold (default: 0.1, i.e., 10%%)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated based on input filename)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode, generate CSV file (default: False)')
    parser.add_argument('--output-tf-list', action='store_true',
                       help='Output TF list to stdout for memory passing (default: False)')
    
    parser.add_argument('--use-processed', action='store_true', default=True,
                       help='Use processed protein FASTA for prediction')
    parser.add_argument('--write-predicted-fasta', action='store_true', default=True,
                       help='Write sequences predicted as TF to project output fasta subdirectory')
    parser.add_argument('--project-output', type=str, default=None,
                       help='Project output directory path for writing predicted TF FASTA')
    parser.add_argument('--mode', type=str, choices=['fast', 'full'], default='fast',
                       help='Sequence splitting mode: fast (contiguous chunks + tail) or full (sliding window). Default: fast')
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        return
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Generate output filename
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.fasta))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base_name}_predictions_{timestamp}.csv"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\nParameter Settings:")
    print(f"Input File: {args.fasta}")
    print(f"Model File: {args.model}")
    print(f"Prediction Threshold: {args.threshold} ({args.threshold*100:.1f}%)")
    print(f"Output File: {args.output}")
    print(f"Batch Size: {args.batch_size}")
    
    # Load model
    model, max_length = load_model(args.model, device)
    
    # Predict
    input_name = os.path.splitext(os.path.basename(args.fasta))[0]
    repo_root = os.path.dirname(script_dir)
    sys.path.append(repo_root)
    try:
        from module.get_fasta import get_processed_fasta_path, read_fasta_to_dict, get_output_subdir
        if args.use_processed:
            processed_fasta = get_processed_fasta_path(args.fasta)
            if not os.path.exists(processed_fasta):
                from module.get_fasta import generate_protein_fasta_with_translation
                generate_protein_fasta_with_translation(args.fasta)
            sequences = []
            for header, seq in read_fasta_to_dict(processed_fasta).items():
                sequences.append({'header': header, 'sequence': seq})
            predictions, tf_headers = predict_fasta(
                model=model,
                device=device,
                threshold=args.threshold,
                max_length=max_length,
                batch_size=args.batch_size,
                sequences=sequences,
                mode=args.mode,
            )
        else:
            predictions, tf_headers = predict_fasta(
                fasta_file=args.fasta,
                model=model,
                device=device,
                threshold=args.threshold,
                max_length=max_length,
                batch_size=args.batch_size,
                mode=args.mode
            )
    except Exception:
        predictions, tf_headers = predict_fasta(
            fasta_file=args.fasta,
            model=model,
            device=device,
            threshold=args.threshold,
            max_length=max_length,
            batch_size=args.batch_size,
            mode=args.mode
        )
    
    if predictions:
        # Save prediction results (Always save in normal mode too)
        # Save all prediction results
        save_predictions(predictions, args.output)
        
        # Save TF prediction results
        tf_output = args.output.replace('.csv', '_tf_only.csv')
        save_tf_predictions(predictions, tf_output)

        # Output TF list to stdout (for memory passing)
        if args.output_tf_list:
            tf_headers = [p['header'] for p in predictions if p['predicted_class'] == 'TF']
            print("TF_LIST_START")
            for header in tf_headers:
                print(f"TF_HEADER:{header}")
            print("TF_LIST_END")

        # Write predicted TF FASTA to project output/fasta directory
        if args.write_predicted_fasta:
            try:
                from module.get_fasta import get_processed_fasta_path, read_fasta_to_dict
                project_dir = args.project_output if args.project_output else get_project_output_dir(args.fasta)
                fasta_dir = os.path.join(project_dir, 'protein_model_preclassification')
                os.makedirs(fasta_dir, exist_ok=True)
                processed_fasta = get_processed_fasta_path(args.fasta)
                seq_dict = read_fasta_to_dict(processed_fasta) if os.path.exists(processed_fasta) else {}
                out_fa = os.path.join(fasta_dir, f"{input_name}_tf_sequences.fasta")
                with open(out_fa, 'w') as f:
                    for p in predictions:
                        if p['predicted_class'] == 'TF':
                            header = p['header']
                            seq = seq_dict.get(header)
                            if seq:
                                f.write(f">{header}\n")
                                f.write(seq + "\n")
                print(f"Predicted TF FASTA saved to: {out_fa}")
            except Exception as e:
                print(f"Failed to write predicted TF FASTA: {e}")
        
        print(f"\n=== Prediction Completed ===")
        print(f"Total Sequences: {len(predictions)}")
        tf_count = sum(1 for p in predictions if p['predicted_class'] == 'TF')
        print(f"Predicted TF Count: {tf_count}")
        print(f"TF Ratio: {tf_count/len(predictions)*100:.2f}%")
        print(f"Threshold Used: {args.threshold} ({args.threshold*100:.1f}%)")
    else:
        print("Error: Prediction failed or no valid sequences found")

if __name__ == '__main__':
    main()
def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_project_output_dir(fasta_file):
    root = _repo_root()
    output_base = os.path.join(root, "output")
    base = os.path.splitext(os.path.basename(fasta_file))[0]
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    candidates = []
    for name in os.listdir(output_base):
        if name == base or name.startswith(base + "_"):
            full = os.path.join(output_base, name)
            if os.path.isdir(full):
                candidates.append(full)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    target = os.path.join(output_base, base)
    os.makedirs(target, exist_ok=True)
    return target

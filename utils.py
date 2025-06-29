import torch
import os
import json
import logging

logger = logging.getLogger(__name__)

def create_sample_data_structure():
    """
    sample directory structure for data organization
    """
    directories = [
        './data/hindi/train',
        './data/hindi/test',
        './data/sanskrit/train', 
        './data/sanskrit/test',
        './data/alignments',
        './outputs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    #sample files to show expected format
    sample_files = {
        './data/hindi/train/sample.txt': 'यह एक नमूना वाक्य है',
        './data/sanskrit/train/sample.txt': 'इदं नमूना वाक्यम् अस्ति',
        './data/alignments/sample.ali': '1 2 3 4 5 6 7 8 9 10',
        './data/vocab.txt': 'यह\nएक\nनमूना\nवाक्य\nहै\nइदं\nनमूना\nवाक्यम्\nअस्ति'
    }
    
    for file_path, content in sample_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    logger.info("Sample data structure created. Please replace with actual data.")

def print_model_summary(model):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model type: {type(model).__name__}")
    print(f"Model architecture:\n{model}")

def save_checkpoint(model, optimizer, epoch, loss, accuracy, config, filepath):
    """Save training checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'config': config
    }, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint

def save_training_stats(stats, filepath):
    """Save training statistics to JSON"""
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Training stats saved to {filepath}")

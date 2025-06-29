import torch
import os

# Configuration for cross-lingual ASR training
CONFIG = {
    # Model configuration
    'model_type': 'DSN',  # 'GRL' or 'DSN'
    'input_dim': 1320,    # 40 mel features * 3 (static+delta+delta-delta) * 11 (context)
    'hidden_dim': 1024,
    'private_hidden': 512,
    'num_senones': 3080,  # As specified in paper for Hindi
    'num_domains': 2,     # Hindi and Sanskrit
    
    # Data paths
    'data_dir': './data',
    'hindi_train_dir': './data/hindi/train',
    'hindi_test_dir': './data/hindi/test', 
    'sanskrit_train_dir': './data/sanskrit/train',
    'sanskrit_test_dir': './data/sanskrit/test',
    'senone_alignments_dir': './data/alignments',
    'vocab_file': './data/vocab.txt',
    'output_dir': './outputs',
    
    # Pre-processed data paths
    'hindi_wav_dir': './data/hindi_wav',
    'sanskrit_wav_dir': './data/sanskrit_wav',
    'cache_dir': './data/cache',
    
    # Training parameters
    'batch_size': 16,  # Reduced for stability
    'learning_rate': 0.001,  # Reduced for stability
    'momentum': 0.9,
    'num_epochs': 10,  # Increased for better training
    'max_hindi_utterances': 3229,  # Your actual data size
    'max_sanskrit_utterances': 1620,  # Your actual data size
    
    # Scheduler parameters
    'scheduler_step_size': 1000,  # More frequent updates
    'scheduler_gamma': 0.95,
    
    # DSN loss weights
    'beta': 0.25,    # Reconstruction loss weight
    'gamma': 0.075,  # Similarity loss weight
    'delta': 0.1,    # Dissimilarity loss weight
    
    # Training settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,  # Set to 0 to avoid multiprocessing issues
    'save_interval': 2,
    'log_interval': 50,
    'similarity_start_step': 500,  # When to start similarity losses in DSN
    
    # Feature extraction
    'n_mels': 40,
    'sample_rate': 8000,
    'n_fft': 512,
    'hop_length': 160,
    'context_frames': 5,
    
    # Evaluation
    'eval_batch_size': 8,
    'test_split': 0.1,  # 10% for testing
}

def get_config():
    """Get configuration dictionary"""
    return CONFIG

def update_config(**kwargs):
    """Update configuration with new values"""
    CONFIG.update(kwargs)
    return CONFIG
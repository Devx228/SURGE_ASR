# import torch
# import os

# # Configuration for cross-lingual ASR training
# CONFIG = {
#     # Model configuration
#     'model_type': 'DSN',  # 'GRL' or 'DSN'
#     'input_dim': 1320,    # 40 mel features * 3 (static+delta+delta-delta) * 11 (context)
#     'hidden_dim': 1024,
#     'private_hidden': 512,
#     'num_senones': 3080,  # As specified in paper for Hindi
#     'num_domains': 2,     # Hindi and Sanskrit
    
#     # Data paths
#     'data_dir': './data',
#     'hindi_train_dir': './data/hindi/train',
#     'hindi_test_dir': './data/hindi/test', 
#     'sanskrit_train_dir': './data/sanskrit/train',
#     'sanskrit_test_dir': './data/sanskrit/test',
#     'senone_alignments_dir': './data/alignments',
#     'vocab_file': './data/vocab.txt',
#     'output_dir': './outputs',
    
#     # Training parameters
#     'batch_size': 32,
#     'learning_rate': 0.01,
#     'momentum': 0.9,
#     'num_epochs': 3,
#     'max_hindi_utterances': 15000,
#     'max_sanskrit_utterances': 2837,
    
#     # Scheduler parameters
#     'scheduler_step_size': 20000,
#     'scheduler_gamma': 0.95,
    
#     # DSN loss weights
#     'beta': 0.25,    # Reconstruction loss weight
#     'gamma': 0.075,  # Similarity loss weight
#     'delta': 0.1,    # Dissimilarity loss weight
    
#     # Training settings
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     'num_workers': 4,
#     'save_interval': 5,
#     'log_interval': 100,
#     'similarity_start_step': 10000,  # When to start similarity losses in DSN
    
#     # Feature extraction
#     'n_mels': 40,
#     'sample_rate': 8000,
#     'n_fft': 512,
#     'hop_length': 160,
#     'context_frames': 5,
    
#     # Evaluation
#     'eval_batch_size': 16,
# }

# def get_config():
#     """Get configuration dictionary"""
#     return CONFIG

# def update_config(**kwargs):
#     """Update configuration with new values"""
#     CONFIG.update(kwargs)
#     return CONFIG


# fixed_config.py

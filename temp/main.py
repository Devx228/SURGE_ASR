# #!/usr/bin/env python3
# """
# Main training script for Sanskrit ASR using TSV data format
# """

# import os
# import sys
# import torch
# import logging
# import argparse
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm

# # Import your modules
# from config import get_config, update_config
# from models import GRLModel, DSNModel
# from trainer import Trainer
# from feature_extraction import FeatureExtractor
# from data_loader_m4a import (
#     M4ADataLoader, load_hindi_data_from_tsv, load_sanskrit_data_from_tsv,
#     M4AASRDataset, generate_alignments_for_m4a, update_config_for_tsv
# )
# from dataset import collate_fn
# from utils import save_checkpoint, create_sample_data_structure

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def setup_directories():
#     """Create necessary directories"""
#     dirs = [
#         './data',
#         './data/hindi_audio',
#         './data/sanskrit_audio', 
#         './data/alignments',
#         './outputs',
#         './temp'
#     ]
#     for d in dirs:
#         os.makedirs(d, exist_ok=True)
#     logger.info("Created directory structure")

# def prepare_data_from_tsv(hindi_tsv_path: str, sanskrit_tsv_path: str, 
#                          hindi_audio_dir: str, sanskrit_audio_dir: str,
#                          config: dict):
#     """Prepare data from TSV files"""
    
#     # Check if TSV files exist
#     if not os.path.exists(hindi_tsv_path):
#         logger.error(f"Hindi TSV file not found: {hindi_tsv_path}")
#         logger.info("Please create a TSV file with format: audio_file<TAB>transcript")
#         return None, None, None, None
    
#     if not os.path.exists(sanskrit_tsv_path):
#         logger.error(f"Sanskrit TSV file not found: {sanskrit_tsv_path}")
#         logger.info("Please create a TSV file with format: audio_file<TAB>transcript")
#         return None, None, None, None
    
#     # Initialize feature extractor
#     feature_extractor = FeatureExtractor(
#         n_mels=config['n_mels'],
#         sample_rate=config['sample_rate'],
#         n_fft=config['n_fft'],
#         hop_length=config['hop_length'],
#         context_frames=config['context_frames']
#     )
    
#     # Load Hindi data
#     logger.info("Loading Hindi data...")
#     hindi_audio, hindi_trans, hindi_senones = load_hindi_data_from_tsv(
#         hindi_tsv_path, hindi_audio_dir, 
#         config['senone_alignments_dir'],
#         max_utterances=config['max_hindi_utterances']
#     )
    
#     # Generate alignments if they don't exist
#     if not any(len(s) > 0 for s in hindi_senones):
#         logger.info("Generating pseudo-alignments for Hindi data...")
#         generate_alignments_for_m4a(
#             hindi_audio, hindi_trans,
#             config['senone_alignments_dir'],
#             num_senones=config['num_senones']
#         )
#         # Reload with alignments
#         hindi_audio, hindi_trans, hindi_senones = load_hindi_data_from_tsv(
#             hindi_tsv_path, hindi_audio_dir,
#             config['senone_alignments_dir'],
#             max_utterances=config['max_hindi_utterances']
#         )
    
#     # Load Sanskrit data
#     logger.info("Loading Sanskrit data...")
#     sanskrit_audio, sanskrit_trans = load_sanskrit_data_from_tsv(
#         sanskrit_tsv_path, sanskrit_audio_dir,
#         max_utterances=config['max_sanskrit_utterances']
#     )
    
#     # Create datasets
#     logger.info("Creating datasets...")
    
#     # Hindi dataset with senone labels
#     source_dataset = M4AASRDataset(
#         hindi_audio, hindi_trans, hindi_senones,
#         [0] * len(hindi_audio),  # Domain 0 for Hindi
#         feature_extractor
#     )
    
#     # Sanskrit dataset without senone labels
#     target_dataset = M4AASRDataset(
#         sanskrit_audio, sanskrit_trans,
#         [np.array([])] * len(sanskrit_audio),  # No senone labels
#         [1] * len(sanskrit_audio),  # Domain 1 for Sanskrit
#         feature_extractor
#     )
    
#     logger.info(f"Created datasets: {len(source_dataset)} Hindi, {len(target_dataset)} Sanskrit")
    
#     # Create data loaders
#     source_loader = DataLoader(
#         source_dataset, batch_size=config['batch_size'],
#         shuffle=True, collate_fn=collate_fn,
#         num_workers=0  # Set to 0 for debugging, increase later
#     )
    
#     target_loader = DataLoader(
#         target_dataset, batch_size=config['batch_size'],
#         shuffle=True, collate_fn=collate_fn,
#         num_workers=0
#     )
    
#     return source_dataset, target_dataset, source_loader, target_loader

# def train_model(model_type: str, config: dict, source_loader, target_loader):
#     """Train a single model"""
    
#     logger.info(f"\n{'='*60}")
#     logger.info(f"Training {model_type} Model")
#     logger.info(f"{'='*60}")
    
#     # Initialize model
#     if model_type == 'GRL':
#         model = GRLModel(
#             input_dim=config['input_dim'],
#             hidden_dim=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
#     else:  # DSN
#         model = DSNModel(
#             input_dim=config['input_dim'],
#             private_hidden=config['private_hidden'],
#             shared_hidden=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
    
#     # Print model summary
#     total_params = sum(p.numel() for p in model.parameters())
#     logger.info(f"Model parameters: {total_params:,}")
    
#     # Initialize trainer
#     trainer = Trainer(
#         model,
#         device=config['device'],
#         learning_rate=config['learning_rate'],
#         batch_size=config['batch_size']
#     )
    
#     # Training loop
#     best_loss = float('inf')
#     training_history = []
    
#     for epoch in range(config['num_epochs']):
#         logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
#         # Train for one epoch
#         if model_type == 'GRL':
#             senone_loss, domain_loss = trainer.train_grl_epoch(source_loader, target_loader)
#             epoch_loss = senone_loss + domain_loss
#             logger.info(f"Senone Loss: {senone_loss:.4f}, Domain Loss: {domain_loss:.4f}")
#         else:  # DSN
#             total_loss = trainer.train_dsn_epoch(source_loader, target_loader)
#             epoch_loss = total_loss
#             logger.info(f"Total Loss: {epoch_loss:.4f}")
        
#         training_history.append(epoch_loss)
        
#         # Save best model
#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             save_checkpoint(
#                 model, trainer.optimizer, epoch,
#                 {'loss': epoch_loss}, 0.0, config,
#                 os.path.join(config['output_dir'], f'best_{model_type.lower()}_model.pth')
#             )
#             logger.info(f"Saved best model with loss: {best_loss:.4f}")
        
#         # Regular checkpoint
#         if (epoch + 1) % config['save_interval'] == 0:
#             save_checkpoint(
#                 model, trainer.optimizer, epoch,
#                 {'loss': epoch_loss}, 0.0, config,
#                 os.path.join(config['output_dir'], f'{model_type.lower()}_epoch_{epoch+1}.pth')
#             )
    
#     logger.info(f"\nTraining completed! Best loss: {best_loss:.4f}")
#     return model, training_history

# def main():
#     parser = argparse.ArgumentParser(description='Train Sanskrit ASR using UDA')
#     parser.add_argument('--hindi-tsv', type=str, required=True,
#                        help='Path to Hindi TSV file')
#     parser.add_argument('--sanskrit-tsv', type=str, required=True,
#                        help='Path to Sanskrit TSV file')
#     parser.add_argument('--hindi-audio-dir', type=str, default='./data/hindi_audio',
#                        help='Directory containing Hindi audio files')
#     parser.add_argument('--sanskrit-audio-dir', type=str, default='./data/sanskrit_audio',
#                        help='Directory containing Sanskrit audio files')
#     parser.add_argument('--model', type=str, default='both',
#                        choices=['GRL', 'DSN', 'both'],
#                        help='Which model to train')
#     parser.add_argument('--epochs', type=int, default=3,
#                        help='Number of training epochs')
#     parser.add_argument('--batch-size', type=int, default=32,
#                        help='Batch size')
    
#     args = parser.parse_args()
    
#     # Setup
#     setup_directories()
    
#     # Get config and update with TSV paths
#     config = get_config()
#     config = update_config_for_tsv(config)
#     config['num_epochs'] = args.epochs
#     config['batch_size'] = args.batch_size
    
#     # Prepare data
#     logger.info("Preparing data...")
#     source_dataset, target_dataset, source_loader, target_loader = prepare_data_from_tsv(
#         args.hindi_tsv, args.sanskrit_tsv,
#         args.hindi_audio_dir, args.sanskrit_audio_dir,
#         config
#     )
    
#     if source_dataset is None:
#         logger.error("Failed to prepare data!")
#         return
    
#     # Train models
#     if args.model in ['GRL', 'both']:
#         grl_model, grl_history = train_model('GRL', config, source_loader, target_loader)
    
#     if args.model in ['DSN', 'both']:
#         dsn_model, dsn_history = train_model('DSN', config, source_loader, target_loader)
    
#     logger.info("\n" + "="*60)
#     logger.info("Training Complete!")
#     logger.info("="*60)
#     logger.info(f"Models saved in: {config['output_dir']}")
    
#     # Print expected results from paper
#     logger.info("\nExpected results from paper:")
#     logger.info("- Baseline DNN (Hindi only): 24.58% WER")
#     logger.info("- GRL: 17.87% WER (6.71% improvement)")
#     logger.info("- DSN: 17.26% WER (7.32% improvement)")

# if __name__ == "__main__":
#     # Check dependencies
#     try:
#         import torch
#         import torchaudio
#         import numpy
#     except ImportError as e:
#         print(f"Missing dependency: {e}")
#         print("Install required packages:")
#         print("pip install torch torchaudio numpy librosa soundfile")
#         sys.exit(1)
    
#     # Check for ffmpeg
#     import subprocess
#     try:
#         subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
#     except:
#         print("Warning: ffmpeg not found. Install it for M4A support:")
#         print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
#         print("  macOS: brew install ffmpeg")
#         print("  Windows: Download from https://ffmpeg.org/download.html")
#         print("\nAlternatively, install pydub: pip install pydub")
    
#     main()
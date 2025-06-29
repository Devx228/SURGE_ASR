# # main_fixed.py
# #!/usr/bin/env python3
# """
# Fixed main training script for Sanskrit ASR using UDA
# """

# import os
# import sys
# import torch
# import logging
# import argparse
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm
# import time

# # Import modules
# from fixed_config import get_config, update_config
# from models import GRLModel, DSNModel
# from trainer import Trainer
# from data_preprocessor import preprocess_all_data
# from fixed_dataset import create_data_loaders
# from evaluation_metrics import ASREvaluator, quick_evaluation_demo
# from utils import save_checkpoint, print_model_summary

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def setup_environment():
#     """Setup training environment and check dependencies"""
#     logger.info("Setting up training environment...")
    
#     # Check PyTorch and CUDA
#     logger.info(f"PyTorch version: {torch.__version__}")
#     logger.info(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
#     # Create directories
#     dirs = [
#         './data',
#         './data/hindi_audio',
#         './data/sanskrit_audio',
#         './data/hindi_wav',
#         './data/sanskrit_wav',
#         './data/alignments',
#         './data/cache',
#         './outputs',
#         './logs'
#     ]
    
#     for d in dirs:
#         os.makedirs(d, exist_ok=True)
    
#     logger.info("Environment setup completed")

# def check_data_files(config):
#     """Check if required data files exist"""
#     required_files = {
#         'Hindi TSV': os.path.join(config['data_dir'], 'hindi.tsv'),
#         'Sanskrit TSV': os.path.join(config['data_dir'], 'sanskrit.tsv'),
#         'Hindi audio dir': os.path.join(config['data_dir'], 'hindi_audio'),
#         'Sanskrit audio dir': os.path.join(config['data_dir'], 'sanskrit_audio')
#     }
    
#     missing_files = []
#     for name, path in required_files.items():
#         if not os.path.exists(path):
#             missing_files.append(f"{name}: {path}")
#         else:
#             if name.endswith('TSV'):
#                 # Count lines in TSV
#                 with open(path, 'r', encoding='utf-8') as f:
#                     line_count = sum(1 for line in f if line.strip())
#                 logger.info(f"✅ {name}: {line_count} entries")
#             else:
#                 # Count files in directory
#                 file_count = len([f for f in os.listdir(path) if f.endswith(('.m4a', '.wav'))])
#                 logger.info(f"✅ {name}: {file_count} audio files")
    
#     if missing_files:
#         logger.error("Missing required files:")
#         for missing in missing_files:
#             logger.error(f"  ❌ {missing}")
#         return False
    
#     return True

# def train_model(model_type: str, config: dict, 
#                 hindi_train_loader, sanskrit_train_loader, 
#                 test_loader, test_data):
#     """Train a single model with evaluation"""
    
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
#     print_model_summary(model)
    
#     # Initialize trainer
#     trainer = Trainer(
#         model,
#         device=config['device'],
#         learning_rate=config['learning_rate'],
#         batch_size=config['batch_size']
#     )
    
#     # Initialize evaluator
#     evaluator = ASREvaluator()
    
#     # Training loop
#     best_wer = float('inf')
#     training_history = []
    
#     logger.info(f"Starting training for {config['num_epochs']} epochs...")
    
#     for epoch in range(config['num_epochs']):
#         epoch_start_time = time.time()
#         logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
#         # Train for one epoch
#         try:
#             if model_type == 'GRL':
#                 senone_loss, domain_loss = trainer.train_grl_epoch(
#                     hindi_train_loader, sanskrit_train_loader
#                 )
#                 epoch_loss = senone_loss + domain_loss
#                 logger.info(f"Senone Loss: {senone_loss:.4f}, Domain Loss: {domain_loss:.4f}")
#             else:  # DSN
#                 epoch_loss = trainer.train_dsn_epoch(
#                     hindi_train_loader, sanskrit_train_loader
#                 )
#                 logger.info(f"Total Loss: {epoch_loss:.4f}")
            
#             training_history.append(epoch_loss)
            
#             # Quick evaluation every few epochs
#             if (epoch + 1) % 2 == 0:
#                 logger.info("\nRunning quick evaluation...")
#                 quick_evaluation_demo(model, test_loader, config['device'])
            
#             # Comprehensive evaluation every 5 epochs
#             if (epoch + 1) % 5 == 0:
#                 logger.info("\nRunning comprehensive evaluation...")
#                 metrics = evaluator.evaluate_model(model, test_loader, config['device'])
#                 evaluator.print_evaluation_report(metrics, f"{model_type} (Epoch {epoch+1})")
                
#                 current_wer = metrics.get('word_error_rate', float('inf'))
                
#                 # Save best model
#                 if current_wer < best_wer:
#                     best_wer = current_wer
#                     save_checkpoint(
#                         model, trainer.optimizer, epoch,
#                         {'loss': epoch_loss, 'wer': current_wer}, current_wer, config,
#                         os.path.join(config['output_dir'], f'best_{model_type.lower()}_model.pth')
#                     )
#                     logger.info(f"🏆 New best model saved! WER: {best_wer*100:.2f}%")
                    
#                     # Save evaluation results
#                     evaluator.save_evaluation_results(
#                         metrics, [], [], config['output_dir'], f'best_{model_type.lower()}'
#                     )
            
#             # Regular checkpoint
#             if (epoch + 1) % config['save_interval'] == 0:
#                 save_checkpoint(
#                     model, trainer.optimizer, epoch,
#                     {'loss': epoch_loss}, 0.0, config,
#                     os.path.join(config['output_dir'], f'{model_type.lower()}_epoch_{epoch+1}.pth')
#                 )
            
#             epoch_time = time.time() - epoch_start_time
#             logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
#         except Exception as e:
#             logger.error(f"Error in epoch {epoch+1}: {e}")
#             continue
    
#     # Final evaluation
#     logger.info(f"\n🎯 Final Evaluation for {model_type}")
#     final_metrics = evaluator.evaluate_model(model, test_loader, config['device'])
#     evaluator.print_evaluation_report(final_metrics, f"{model_type} (Final)")
    
#     # Save final results
#     evaluator.save_evaluation_results(
#         final_metrics, [], [], config['output_dir'], f'final_{model_type.lower()}'
#     )
    
#     logger.info(f"Training completed! Best WER: {best_wer*100:.2f}%")
#     return model, training_history, final_metrics

# def main():
#     parser = argparse.ArgumentParser(description='Train Sanskrit ASR using UDA')
#     parser.add_argument('--model', type=str, default='both',
#                        choices=['GRL', 'DSN', 'both'],
#                        help='Which model to train')
#     parser.add_argument('--epochs', type=int, default=10,
#                        help='Number of training epochs')
#     parser.add_argument('--batch-size', type=int, default=16,
#                        help='Batch size')
#     parser.add_argument('--learning-rate', type=float, default=0.001,
#                        help='Learning rate')
#     parser.add_argument('--preprocess-only', action='store_true',
#                        help='Only run preprocessing, skip training')
#     parser.add_argument('--skip-preprocess', action='store_true',
#                        help='Skip preprocessing, use cached data')
    
#     args = parser.parse_args()
    
#     # Setup
#     setup_environment()
    
#     # Get config
#     config = get_config()
#     config['num_epochs'] = args.epochs
#     config['batch_size'] = args.batch_size
#     config['learning_rate'] = args.learning_rate
    
#     logger.info(f"Training configuration:")
#     logger.info(f"  Model: {args.model}")
#     logger.info(f"  Epochs: {args.epochs}")
#     logger.info(f"  Batch size: {args.batch_size}")
#     logger.info(f"  Learning rate: {args.learning_rate}")
#     logger.info(f"  Device: {config['device']}")
    
#     # Check data files
#     if not check_data_files(config):
#         logger.error("Please ensure all required data files are present!")
#         sys.exit(1)
    
#     # Preprocessing
#     if not args.skip_preprocess:
#         logger.info("Starting data preprocessing...")
#         hindi_pairs, sanskrit_pairs = preprocess_all_data(config)
#         if hindi_pairs is None:
#             logger.error("Data preprocessing failed!")
#             sys.exit(1)
    
#     if args.preprocess_only:
#         logger.info("Preprocessing completed. Exiting.")
#         return
    
#     # Create data loaders
#     logger.info("Creating data loaders...")
#     hindi_train_loader, sanskrit_train_loader, test_loader, test_data = create_data_loaders(config)
    
#     if hindi_train_loader is None:
#         logger.error("Failed to create data loaders!")
#         sys.exit(1)
    
#     # Training
#     results = {}
    
#     if args.model in ['GRL', 'both']:
#         logger.info("\n🚀 Training GRL Model")
#         grl_model, grl_history, grl_metrics = train_model(
#             'GRL', config, hindi_train_loader, sanskrit_train_loader, 
#             test_loader, test_data
#         )
#         results['GRL'] = grl_metrics
    
#     if args.model in ['DSN', 'both']:
#         logger.info("\n🚀 Training DSN Model")
#         dsn_model, dsn_history, dsn_metrics = train_model(
#             'DSN', config, hindi_train_loader, sanskrit_train_loader, 
#             test_loader, test_data
#         )
#         results['DSN'] = dsn_metrics
    
#     # Final summary
#     logger.info("\n" + "="*80)
#     logger.info("TRAINING COMPLETE - FINAL SUMMARY")
#     logger.info("="*80)
    
#     for model_name, metrics in results.items():
#         wer = metrics['word_error_rate'] * 100
#         senone_acc = metrics['senone_accuracy'] * 100
#         logger.info(f"\n{model_name} Model Results:")
#         logger.info(f"  Word Error Rate: {wer:.2f}%")
#         logger.info(f"  Senone Accuracy: {senone_acc:.2f}%")
#         logger.info(f"  Total Samples: {metrics['total_samples']}")
    
#     logger.info(f"\nModels saved in: {config['output_dir']}")
#     logger.info(f"Logs saved in: training.log")
    
#     # Performance comparison
#     logger.info(f"\n📊 Performance Comparison:")
#     logger.info(f"  Paper Baseline (Hindi only): 24.58% WER")
#     logger.info(f"  Paper GRL Result:            17.87% WER")
#     logger.info(f"  Paper DSN Result:            17.26% WER")
    
#     if 'GRL' in results:
#         grl_wer = results['GRL']['word_error_rate'] * 100
#         improvement = 24.58 - grl_wer
#         logger.info(f"  Your GRL Result:             {grl_wer:.2f}% WER ({improvement:+.2f}% vs baseline)")
    
#     if 'DSN' in results:
#         dsn_wer = results['DSN']['word_error_rate'] * 100  
#         improvement = 24.58 - dsn_wer
#         logger.info(f"  Your DSN Result:             {dsn_wer:.2f}% WER ({improvement:+.2f}% vs baseline)")
    
#     logger.info("\n🎉 Training pipeline completed successfully!")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         logger.info("\nTraining interrupted by user")
#     except Exception as e:
#         logger.error(f"Training failed with error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
# main_fixed.py
#!/usr/bin/env python3
"""
Fixed main training script for Sanskrit ASR using UDA
"""

import os
import sys
import torch
import logging
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import json

# Import modules
from fixed_config import get_config, update_config
from models import GRLModel, DSNModel
from trainer import Trainer
from data_preprocessor import preprocess_all_data
from fixed_dataset import create_data_loaders
from evaluation_metrics import ASREvaluator
from utils import save_checkpoint, print_model_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_data_dimensions(config):
    """Verify that feature dimensions are correct"""
    logger.info("\n🔍 Verifying data dimensions...")
    
    # Expected dimensions
    expected_feature_dim = config['n_mels'] * 3 * (2 * config['context_frames'] + 1)
    logger.info(f"Expected feature dimension: {expected_feature_dim}")
    logger.info(f"  - n_mels: {config['n_mels']}")
    logger.info(f"  - features: static + delta + delta-delta = 3")
    logger.info(f"  - context: {2 * config['context_frames'] + 1} frames")
    
    if expected_feature_dim != config['input_dim']:
        logger.error(f"❌ Mismatch! Config input_dim={config['input_dim']} but expected {expected_feature_dim}")
        logger.info(f"Updating config input_dim to {expected_feature_dim}")
        config['input_dim'] = expected_feature_dim
    else:
        logger.info(f"✅ Feature dimensions match: {expected_feature_dim}")
    
    return config

def test_data_loader_dimensions(data_loader, loader_name="DataLoader"):
    """Test a single batch from data loader to verify dimensions"""
    logger.info(f"\n📊 Testing {loader_name} dimensions...")
    
    try:
        # Get one batch
        batch = next(iter(data_loader))
        
        # Log shapes
        logger.info(f"Batch shapes:")
        logger.info(f"  - features: {batch['features'].shape}")
        logger.info(f"  - senones: {batch['senones'].shape}")
        logger.info(f"  - domains: {batch['domains'].shape}")
        logger.info(f"  - feature_lengths: {batch['feature_lengths'].shape}")
        logger.info(f"  - senone_lengths: {batch['senone_lengths'].shape}")
        
        # Verify feature dimension
        batch_size, seq_len, feature_dim = batch['features'].shape
        logger.info(f"\nFeature analysis:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Max sequence length: {seq_len}")
        logger.info(f"  - Feature dimension: {feature_dim}")
        
        # Check senone alignment
        senone_batch_size, senone_seq_len = batch['senones'].shape
        logger.info(f"\nSenone analysis:")
        logger.info(f"  - Senone sequence length: {senone_seq_len}")
        logger.info(f"  - Valid senones (not -1): {(batch['senones'] != -1).sum().item()}")
        
        # Verify consistency
        assert batch_size == senone_batch_size, "Batch size mismatch between features and senones"
        
        # Check actual lengths vs padded lengths
        logger.info(f"\nLength analysis:")
        for i in range(min(3, batch_size)):  # Check first 3 samples
            feat_len = batch['feature_lengths'][i].item()
            senone_len = batch['senone_lengths'][i].item()
            logger.info(f"  Sample {i}: feature_len={feat_len}, senone_len={senone_len}")
        
        return True, feature_dim
        
    except Exception as e:
        logger.error(f"❌ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def setup_environment():
    """Setup training environment and check dependencies"""
    logger.info("Setting up training environment...")
    
    # Check PyTorch and CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    dirs = [
        './data',
        './data/hindi_audio',
        './data/sanskrit_audio', 
        './data/hindi_wav',
        './data/sanskrit_wav',
        './data/alignments',
        './data/cache',
        './data/cache/features',
        './outputs',
        './logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    logger.info("Environment setup completed")

def check_data_files(config):
    """Check if required data files exist"""
    required_files = {
        'Hindi TSV': os.path.join(config['data_dir'], 'hindi.tsv'),
        'Sanskrit TSV': os.path.join(config['data_dir'], 'sanskrit.tsv'),
        'Hindi audio dir': os.path.join(config['data_dir'], 'hindi_audio'),
        'Sanskrit audio dir': os.path.join(config['data_dir'], 'sanskrit_audio')
    }
    
    missing_files = []
    data_stats = {}
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
        else:
            if name.endswith('TSV'):
                # Count lines in TSV
                with open(path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f if line.strip())
                logger.info(f"✅ {name}: {line_count} entries")
                data_stats[name] = line_count
            else:
                # Count files in directory
                file_count = len([f for f in os.listdir(path) if f.endswith(('.m4a', '.wav'))])
                logger.info(f"✅ {name}: {file_count} audio files")
                data_stats[name] = file_count
    
    if missing_files:
        logger.error("Missing required files:")
        for missing in missing_files:
            logger.error(f"  ❌ {missing}")
        return False
    
    return True

def quick_model_test(model, config):
    """Quick test to ensure model forward pass works"""
    logger.info("\n🧪 Testing model forward pass...")
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 100
        feature_dim = config['input_dim']
        
        dummy_features = torch.randn(batch_size, seq_len, feature_dim).to(config['device'])
        
        model.eval()
        with torch.no_grad():
            if isinstance(model, GRLModel):
                senone_logits, domain_logits = model(dummy_features, alpha=0.5)
                logger.info(f"GRL output shapes:")
                logger.info(f"  - senone_logits: {senone_logits.shape}")
                logger.info(f"  - domain_logits: {domain_logits.shape}")
            else:  # DSN
                outputs = model(dummy_features, domain_id=0, alpha=0.5)
                logger.info(f"DSN output shapes:")
                logger.info(f"  - senone_logits: {outputs[0].shape}")
                logger.info(f"  - domain_logits: {outputs[1].shape}")
                logger.info(f"  - reconstructed: {outputs[2].shape}")
                logger.info(f"  - shared_features: {outputs[3].shape}")
                logger.info(f"  - private_features: {outputs[4].shape}")
        
        logger.info("✅ Model forward pass successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model(model_type: str, config: dict, 
                hindi_train_loader, sanskrit_train_loader, 
                test_loader, test_data):
    """Train a single model with evaluation and dimension checking"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type} Model")
    logger.info(f"{'='*60}")
    
    # Initialize model
    if model_type == 'GRL':
        model = GRLModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_senones=config['num_senones'],
            num_domains=config['num_domains']
        )
    else:  # DSN
        model = DSNModel(
            input_dim=config['input_dim'],
            private_hidden=config['private_hidden'],
            shared_hidden=config['hidden_dim'],
            num_senones=config['num_senones'],
            num_domains=config['num_domains']
        )
    model.to(config['device'])
    
    # Print model summary
    print_model_summary(model)
    
    # Test model forward pass
    if not quick_model_test(model, config):
        logger.error("Model test failed! Check model architecture.")
        return None, None, None
    
    # Initialize trainer
    trainer = Trainer(
        model,
        device=config['device'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size']
    )
    
    # Initialize evaluator
    evaluator = ASREvaluator(config.get('vocab_file', '/data/vocab.txt'))
    
    # Training loop
    best_wer = float('inf')
    training_history = {
        'loss': [],
        'wer': [],
        'senone_acc': []
    }
    
    logger.info(f"Starting training for {config['num_epochs']} epochs...")
    logger.info(f"Training on device: {config['device']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        logger.info(f"{'='*50}")
        
        try:
            # Train for one epoch
            if model_type == 'GRL':
                senone_loss, domain_loss = trainer.train_grl_epoch(
                    hindi_train_loader, sanskrit_train_loader
                )
                epoch_loss = senone_loss + domain_loss
                logger.info(f"Epoch Summary - Senone Loss: {senone_loss:.4f}, Domain Loss: {domain_loss:.4f}")
            else:  # DSN
                epoch_loss = trainer.train_dsn_epoch(
                    hindi_train_loader, sanskrit_train_loader
                )
                logger.info(f"Epoch Summary - Total Loss: {epoch_loss:.4f}")
            
            training_history['loss'].append(epoch_loss)
            
            # Validation every 2 epochs
            if (epoch + 1) % 2 == 0:
                logger.info("\n📊 Running validation...")
                val_metrics = trainer.validate(test_loader)
                logger.info(f"Validation - Senone Accuracy: {val_metrics['senone_accuracy']*100:.2f}%, Loss: {val_metrics['avg_loss']:.4f}")
                training_history['senone_acc'].append(val_metrics['senone_accuracy'])
            
            # Comprehensive evaluation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
                logger.info("\n📊 Running comprehensive evaluation...")
                epoch_metrics,epoch_examples = evaluator.evaluate_model(model, test_loader, config['device'])

                evaluator.print_evaluation_report(epoch_metrics, epoch_examples, f"{model_type} @ Epoch {epoch+1}")
                
                current_wer = epoch_metrics.get('word_error_rate', float('inf'))
                training_history['wer'].append(current_wer)
                
                # Save best model
                if current_wer < best_wer:
                    best_wer = current_wer
                    # trainer.save_checkpoint(
                    #     os.path.join(config['output_dir'], f'best_{model_type.lower()}_model.pth'),
                    #     epoch,
                    #     epoch_metrics
                    # )
                    save_checkpoint(
                            model, trainer.optimizer, epoch,
                            epoch_loss,  # This is the "loss" argument in your utils.py, but you're passing metrics. See note below.
                            0.0,            # accuracy (you can pass 0.0 or actual accuracy if available)
                            config,
                            os.path.join(config['output_dir'], f'best_{model_type.lower()}_model.pth')
                    )
                    logger.info(f"🏆 New best model saved! WER: {best_wer*100:.2f}%")
            
            # Regular checkpoint
            if (epoch + 1) % config['save_interval'] == 0:
                save_checkpoint(
                    model, trainer.optimizer, epoch,
                    epoch_loss,  # loss
                    0.0,         # accuracy
                    config,
                    os.path.join(config['output_dir'], f'{model_type.lower()}_epoch_{epoch+1}.pth')
                )
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"⏱️  Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("❌ CUDA out of memory! Try reducing batch size.")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.error(f"Runtime error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final evaluation
    logger.info(f"\n🎯 Final Evaluation for {model_type}")
    final_metrics,final_examples = evaluator.evaluate_model(model, test_loader, config['device'])
    evaluator.print_evaluation_report(final_metrics, final_examples, f"Final {model_type}")
    
    # Save training history
    history_file = os.path.join(config['output_dir'], f'{model_type.lower()}_history.json')
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"✅ Training completed! Best WER: {best_wer*100:.2f}%")
    return model, training_history, final_metrics

def main():
    parser = argparse.ArgumentParser(description='Train Sanskrit ASR using UDA')
    parser.add_argument('--model', type=str, default='both',
                       choices=['GRL', 'DSN', 'both'],
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run preprocessing')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip preprocessing, use cached data')
    parser.add_argument('--test-dimensions', action='store_true',
                       help='Test data dimensions and exit')
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Get config
    config = get_config()
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    
    # Set device
    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device
    
    # Verify and fix dimensions
    config = verify_data_dimensions(config)
    
    logger.info(f"\n🚀 Training Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Input dimension: {config['input_dim']}")
    
    # Check data files
    if not check_data_files(config):
        logger.error("Please ensure all required data files are present!")
        sys.exit(1)
    
    # Preprocessing
    if not args.skip_preprocess:
        logger.info("\n📦 Starting data preprocessing...")
        hindi_pairs, sanskrit_pairs = preprocess_all_data(config)
        if hindi_pairs is None:
            logger.error("Data preprocessing failed!")
            sys.exit(1)
        logger.info("✅ Preprocessing completed successfully!")
    else:
        logger.info("⏭️  Skipping preprocessing, using cached data")
    
    if args.preprocess_only:
        logger.info("Preprocessing completed. Exiting.")
        return
    
    # Create data loaders
    logger.info("\n📊 Creating data loaders...")
    hindi_train_loader, sanskrit_train_loader, test_loader, test_data = create_data_loaders(config)
    
    if hindi_train_loader is None:
        logger.error("Failed to create data loaders!")
        sys.exit(1)
    
    logger.info(f"✅ Data loaders created:")
    logger.info(f"  Hindi train batches: {len(hindi_train_loader)}")
    logger.info(f"  Sanskrit train batches: {len(sanskrit_train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Test dimensions
    if args.test_dimensions or config.get('debug_dimensions', False):
        logger.info("\n🔬 Testing data dimensions...")
        
        # Test Hindi loader
        success, feature_dim = test_data_loader_dimensions(hindi_train_loader, "Hindi Train")
        if not success:
            logger.error("Hindi data loader test failed!")
            sys.exit(1)
        
        # Test Sanskrit loader
        success, feature_dim = test_data_loader_dimensions(sanskrit_train_loader, "Sanskrit Train")
        if not success:
            logger.error("Sanskrit data loader test failed!")
            sys.exit(1)
        
        # Test loader
        success, feature_dim = test_data_loader_dimensions(test_loader, "Test")
        if not success:
            logger.error("Test data loader test failed!")
            sys.exit(1)
        
        # Verify against config
        if feature_dim != config['input_dim']:
            logger.error(f"❌ Feature dimension mismatch! Data: {feature_dim}, Config: {config['input_dim']}")
            logger.info(f"Updating config to match data...")
            config['input_dim'] = feature_dim
        
        if args.test_dimensions:
            logger.info("✅ Dimension testing completed. Exiting.")
            return
    
    # Training
    results = {}
    
    if args.model in ['GRL', 'both']:
        logger.info("\n🚀 Training GRL Model")
        grl_model, grl_history, grl_metrics = train_model(
            'GRL', config, hindi_train_loader, sanskrit_train_loader, 
            test_loader, test_data
        )
        if grl_metrics:
            results['GRL'] = grl_metrics
    
    if args.model in ['DSN', 'both']:
        logger.info("\n🚀 Training DSN Model")
        dsn_model, dsn_history, dsn_metrics = train_model(
            'DSN', config, hindi_train_loader, sanskrit_train_loader, 
            test_loader, test_data
        )
        if dsn_metrics:
            results['DSN'] = dsn_metrics
    
    # Final summary
    if results:
        logger.info("\n" + "="*80)
        logger.info("🎉 TRAINING COMPLETE - FINAL SUMMARY")
        logger.info("="*80)
        
        for model_name, metrics in results.items():
            if metrics:
                wer = metrics.get('word_error_rate', 1.0) * 100
                senone_acc = metrics.get('senone_accuracy', 0.0) * 100
                logger.info(f"\n{model_name} Model Results:")
                logger.info(f"  Word Error Rate: {wer:.2f}%")
                logger.info(f"  Senone Accuracy: {senone_acc:.2f}%")
                logger.info(f"  Total Samples: {metrics.get('total_samples', 0)}")
        
        logger.info(f"\n📁 Output Files:")
        logger.info(f"  Models saved in: {config['output_dir']}")
        logger.info(f"  Training logs: training.log")
    
    logger.info("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
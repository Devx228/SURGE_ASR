# # test_everything.py
# #!/usr/bin/env python3
# """
# Comprehensive test script to verify everything works before training
# """

# import os
# import sys
# import torch
# import logging
# import traceback
# from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def test_imports():
#     """Test all required imports"""
#     logger.info("🔍 Testing imports...")
    
#     try:
#         import torch
#         import torchaudio
#         import numpy as np
#         import pandas as pd
#         from tqdm import tqdm
#         import matplotlib.pyplot as plt
#         import editdistance
        
#         logger.info("✅ All basic packages imported successfully")
        
#         # Test project modules
#         from fixed_config import get_config
#         from models import GRLModel, DSNModel
#         from feature_extraction import FeatureExtractor
#         from data_preprocessor import AudioPreprocessor
#         from evaluation_metrics import ASREvaluator
        
#         logger.info("✅ All project modules imported successfully")
#         return True
        
#     except ImportError as e:
#         logger.error(f"❌ Import error: {e}")
#         return False
#     except Exception as e:
#         logger.error(f"❌ Unexpected error: {e}")
#         return False

# def test_config():
#     """Test configuration loading"""
#     logger.info("🔍 Testing configuration...")
    
#     try:
#         from fixed_config import get_config
#         config = get_config()
        
#         required_keys = [
#             'input_dim', 'hidden_dim', 'num_senones', 'num_domains',
#             'batch_size', 'learning_rate', 'num_epochs', 'device'
#         ]
        
#         for key in required_keys:
#             if key not in config:
#                 logger.error(f"❌ Missing config key: {key}")
#                 return False
        
#         logger.info(f"✅ Configuration loaded successfully")
#         logger.info(f"  Device: {config['device']}")
#         logger.info(f"  Input dim: {config['input_dim']}")
#         logger.info(f"  Batch size: {config['batch_size']}")
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Configuration error: {e}")
#         return False

# def test_feature_extraction():
#     """Test feature extraction"""
#     logger.info("🔍 Testing feature extraction...")
    
#     try:
#         from feature_extraction import FeatureExtractor
#         from fixed_config import get_config
        
#         config = get_config()
#         extractor = FeatureExtractor(
#             n_mels=config['n_mels'],
#             sample_rate=config['sample_rate'],
#             n_fft=config['n_fft'],
#             hop_length=config['hop_length'],
#             context_frames=config['context_frames']
#         )
        
#         # Create a dummy audio file for testing
#         import torchaudio
#         import numpy as np
        
#         # Generate 1 second of dummy audio (8kHz)
#         dummy_audio = np.random.randn(8000).astype(np.float32)
#         dummy_path = './temp/test_audio.wav'
#         os.makedirs('./temp', exist_ok=True)
        
#         # Save dummy audio
#         torchaudio.save(dummy_path, torch.tensor(dummy_audio).unsqueeze(0), 8000)
        
#         # Test feature extraction
#         features = extractor.extract_features(dummy_path)
        
#         logger.info(f"✅ Feature extraction successful")
#         logger.info(f"  Feature shape: {features.shape}")
#         logger.info(f"  Expected: (time_frames, 1320)")
        
#         if features.shape[1] == 1320:
#             logger.info("✅ Feature dimensions correct")
#         else:
#             logger.error(f"❌ Wrong feature dimensions: {features.shape[1]} != 1320")
#             return False
        
#         # Clean up
#         os.remove(dummy_path)
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Feature extraction error: {e}")
#         traceback.print_exc()
#         return False

# def test_models():
#     """Test model creation and forward pass"""
#     logger.info("🔍 Testing models...")
    
#     try:
#         from models import GRLModel, DSNModel
#         from fixed_config import get_config
        
#         config = get_config()
        
#         # Test GRL model
#         grl_model = GRLModel(
#             input_dim=config['input_dim'],
#             hidden_dim=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
        
#         # Test DSN model
#         dsn_model = DSNModel(
#             input_dim=config['input_dim'],
#             private_hidden=config['private_hidden'],
#             shared_hidden=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
        
#         # Test forward pass with dummy data
#         batch_size, seq_len, input_dim = 2, 100, config['input_dim']
#         dummy_input = torch.randn(batch_size, seq_len, input_dim)
        
#         # Test GRL forward pass
#         grl_senone_logits, grl_domain_logits = grl_model(dummy_input, alpha=0.5)
#         logger.info(f"✅ GRL forward pass successful")
#         logger.info(f"  Senone logits shape: {grl_senone_logits.shape}")
#         logger.info(f"  Domain logits shape: {grl_domain_logits.shape}")
        
#         # Test DSN forward pass
#         dsn_senone_logits, dsn_domain_logits, reconstructed, shared, private = dsn_model(dummy_input, domain_id=0)
#         logger.info(f"✅ DSN forward pass successful")
#         logger.info(f"  Senone logits shape: {dsn_senone_logits.shape}")
#         logger.info(f"  Reconstructed shape: {reconstructed.shape}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Model error: {e}")
#         traceback.print_exc()
#         return False

# def test_data_structure():
#     """Test data directory structure and sample files"""
#     logger.info("🔍 Testing data structure...")
    
#     required_dirs = [
#         './data',
#         './data/hindi_audio',
#         './data/sanskrit_audio',
#         './data/cache',
#         './outputs'
#     ]
    
#     for d in required_dirs:
#         if os.path.exists(d):
#             logger.info(f"✅ {d}")
#         else:
#             logger.error(f"❌ Missing directory: {d}")
#             return False
    
#     # Check for TSV files
#     tsv_files = ['./data/hindi.tsv', './data/sanskrit.tsv']
#     for tsv in tsv_files:
#         if os.path.exists(tsv):
#             with open(tsv, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#             logger.info(f"✅ {tsv} ({len(lines)} entries)")
#         else:
#             logger.warning(f"⚠️ {tsv} not found (create your data files)")
    
#     return True

# def test_audio_preprocessing():
#     """Test audio preprocessing with dummy files"""
#     logger.info("🔍 Testing audio preprocessing...")
    
#     try:
#         from data_preprocessor import AudioPreprocessor
#         from fixed_config import get_config
        
#         config = get_config()
#         preprocessor = AudioPreprocessor(config)
        
#         # Test with dummy WAV file
#         import torchaudio
#         import numpy as np
        
#         # Create dummy audio
#         dummy_audio = np.random.randn(16000).astype(np.float32)  # 2 seconds at 8kHz
#         dummy_wav = './temp/test_preprocessing.wav'
#         os.makedirs('./temp', exist_ok=True)
        
#         torchaudio.save(dummy_wav, torch.tensor(dummy_audio).unsqueeze(0), 8000)
        
#         # Test file verification
#         is_valid = preprocessor.verify_wav_file(dummy_wav)
#         if is_valid:
#             logger.info("✅ Audio file verification works")
#         else:
#             logger.error("❌ Audio file verification failed")
#             return False
        
#         # Test alignment generation
#         test_pairs = [(dummy_wav, "यह एक परीक्षा वाक्य है")]
#         alignments = preprocessor.generate_basic_alignments(test_pairs, "Test")
        
#         if len(alignments) > 0 and len(alignments[0]) > 0:
#             logger.info(f"✅ Alignment generation works (length: {len(alignments[0])})")
#         else:
#             logger.error("❌ Alignment generation failed")
#             return False
        
#         # Clean up
#         os.remove(dummy_wav)
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Preprocessing error: {e}")
#         traceback.print_exc()
#         return False

# def test_evaluation():
#     """Test evaluation metrics"""
#     logger.info("🔍 Testing evaluation metrics...")
    
#     try:
#         from evaluation_metrics import ASREvaluator
        
#         evaluator = ASREvaluator()
        
#         # Test with dummy predictions
#         predictions = ["यह एक परीक्षा है", "नमूना वाक्य"]
#         references = ["यह एक परीक्षण है", "नमूना वाक्य"]
        
#         wer = evaluator.calculate_wer(predictions, references)
#         cer = evaluator.calculate_cer(predictions, references)
        
#         logger.info(f"✅ Evaluation metrics work")
#         logger.info(f"  Test WER: {wer*100:.2f}%")
#         logger.info(f"  Test CER: {cer*100:.2f}%")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Evaluation error: {e}")
#         traceback.print_exc()
#         return False

# def test_full_pipeline():
#     """Test a minimal version of the full pipeline"""
#     logger.info("🔍 Testing minimal full pipeline...")
    
#     try:
#         # This will test creating a small dataset and running one forward pass
#         from fixed_config import get_config
#         from models import DSNModel
#         from fixed_dataset import ASRDataset, collate_fn
#         from feature_extraction import FeatureExtractor
#         from torch.utils.data import DataLoader
#         import numpy as np
#         import torchaudio
        
#         config = get_config()
        
#         # Create dummy data
#         os.makedirs('./temp', exist_ok=True)
#         dummy_files = []
#         dummy_transcripts = []
#         dummy_alignments = []
        
#         for i in range(3):  # 3 dummy files
#             # Create dummy audio
#             audio_data = np.random.randn(8000).astype(np.float32)  # 1 second
#             audio_path = f'./temp/dummy_{i}.wav'
#             torchaudio.save(audio_path, torch.tensor(audio_data).unsqueeze(0), 8000)
            
#             dummy_files.append(audio_path)
#             dummy_transcripts.append(f"परीक्षा वाक्य {i}")
#             dummy_alignments.append(np.random.randint(0, 100, size=50))  # 50 frames
        
#         # Create feature extractor
#         feature_extractor = FeatureExtractor(
#             n_mels=config['n_mels'],
#             sample_rate=config['sample_rate'],
#             n_fft=config['n_fft'],
#             hop_length=config['hop_length'],
#             context_frames=config['context_frames']
#         )
        
#         # Create dataset
#         dataset = ASRDataset(
#             dummy_files, dummy_transcripts, dummy_alignments,
#             [0, 1, 0], feature_extractor, cache_features=False
#         )
        
#         # Create dataloader
#         dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
#         # Create model
#         model = DSNModel(
#             input_dim=config['input_dim'],
#             private_hidden=config['private_hidden'],
#             shared_hidden=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
        
#         # Test one batch
#         batch = next(iter(dataloader))
#         features = batch['features']
        
#         # Forward pass
#         with torch.no_grad():
#             senone_logits, domain_logits, reconstructed, shared, private = model(features, domain_id=0)
        
#         logger.info(f"✅ Full pipeline test successful")
#         logger.info(f"  Batch size: {features.shape[0]}")
#         logger.info(f"  Sequence length: {features.shape[1]}")
#         logger.info(f"  Output shape: {senone_logits.shape}")
        
#         # Clean up
#         for f in dummy_files:
#             if os.path.exists(f):
#                 os.remove(f)
        
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Full pipeline error: {e}")
#         traceback.print_exc()
#         return False

# def main():
#     print("""
# ╔══════════════════════════════════════════════════════════════╗
# ║              Comprehensive System Test                       ║
# ║              Testing all components before training          ║
# ╚══════════════════════════════════════════════════════════════╝
# """)
    
#     tests = [
#         ("Package Imports", test_imports),
#         ("Configuration", test_config),
#         ("Feature Extraction", test_feature_extraction),
#         ("Model Creation", test_models),
#         ("Data Structure", test_data_structure),
#         ("Audio Preprocessing", test_audio_preprocessing),
#         ("Evaluation Metrics", test_evaluation),
#         ("Full Pipeline", test_full_pipeline)
#     ]
    
#     passed = 0
#     total = len(tests)
    
#     for test_name, test_func in tests:
#         print(f"\n{'='*60}")
#         print(f"Running: {test_name}")
#         print(f"{'='*60}")
        
#         try:
#             if test_func():
#                 passed += 1
#                 print(f"✅ {test_name} PASSED")
#             else:
#                 print(f"❌ {test_name} FAILED")
#         except Exception as e:
#             print(f"❌ {test_name} CRASHED: {e}")
    
#     # Summary
#     print(f"\n{'='*60}")
#     print(f"TEST SUMMARY")
#     print(f"{'='*60}")
#     print(f"Passed: {passed}/{total} tests")
    
#     if passed == total:
#         print(f"""
# 🎉 ALL TESTS PASSED! 

# Your system is ready for training. Next steps:

# 1. Prepare your data:
#    - Place 3229 Hindi M4A files in ./data/hindi_audio/
#    - Place 1620 Sanskrit M4A files in ./data/sanskrit_audio/
#    - Create ./data/hindi.tsv with your transcripts
#    - Create ./data/sanskrit.tsv with your transcripts

# 2. Run preprocessing:
#    python main_fixed.py --preprocess-only

# 3. Start training:
#    python main_fixed.py --model both --epochs 10

# 4. Monitor results:
#    Check training.log and ./outputs/ for results
# """)
#     else:
#         print(f"""
# ⚠️  {total - passed} tests failed. Please fix the issues before training.

# Common solutions:
# - Install missing packages: pip install -r requirements.txt
# - Check file paths and permissions
# - Ensure sufficient disk space
# - Verify Python version (3.7+)
# """)
        
#     # Additional system info
#     print(f"\n💻 System Information:")
#     print(f"  Python: {sys.version}")
#     print(f"  PyTorch: {torch.__version__ if 'torch' in sys.modules else 'Not loaded'}")
#     print(f"  CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else 'Unknown'}")
#     if 'torch' in sys.modules and torch.cuda.is_available():
#         print(f"  CUDA device: {torch.cuda.get_device_name()}")

# if __name__ == "__main__":
#     main()
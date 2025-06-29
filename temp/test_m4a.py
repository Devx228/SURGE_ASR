# #!/usr/bin/env python3
# """
# Test script to verify M4A audio processing works correctly
# """

# import os
# import sys
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def test_m4a_conversion():
#     """Test M4A to WAV conversion"""
#     print("\n🔊 Testing M4A Audio Conversion...")
    
#     from data_loader_m4a import M4ADataLoader
    
#     # Create test M4A file path (you'll need to provide actual file)
#     test_m4a = "./data/sanskrit_audio/844424932806112-643-f.m4a"
    
#     if not os.path.exists(test_m4a):
#         print(f"❌ Test file not found: {test_m4a}")
#         print("   Please add at least one M4A file to test")
#         return False
    
#     loader = M4ADataLoader()
    
#     try:
#         # Test conversion
#         wav_path = loader.convert_m4a_to_wav(test_m4a)
#         print(f"✅ Successfully converted to: {wav_path}")
        
#         # Check if WAV file exists and is valid
#         if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
#             print("✅ WAV file is valid")
#             return True
#         else:
#             print("❌ WAV file is invalid")
#             return False
            
#     except Exception as e:
#         print(f"❌ Conversion failed: {e}")
#         print("\nTrying with pydub...")
        
#         try:
#             wav_path = loader.convert_with_pydub(test_m4a, test_m4a.replace('.m4a', '.wav'))
#             print("✅ Pydub conversion successful")
#             return True
#         except:
#             print("❌ Pydub also failed. Please install ffmpeg or pydub")
#             return False

# def test_feature_extraction():
#     """Test feature extraction from audio"""
#     print("\n🎵 Testing Feature Extraction...")
    
#     from feature_extraction import FeatureExtractor
#     from data_loader_m4a import M4ADataLoader
    
#     # Find a test audio file
#     test_files = []
#     for ext in ['.m4a', '.wav']:
#         if os.path.exists('./data/sanskrit_audio'):
#             test_files.extend([
#                 os.path.join('./data/sanskrit_audio', f) 
#                 for f in os.listdir('./data/sanskrit_audio') 
#                 if f.endswith(ext)
#             ])
    
#     if not test_files:
#         print("❌ No audio files found for testing")
#         return False
    
#     test_audio = test_files[0]
#     print(f"   Using: {test_audio}")
    
#     # Convert if M4A
#     if test_audio.endswith('.m4a'):
#         loader = M4ADataLoader()
#         test_audio = loader.convert_m4a_to_wav(test_audio)
    
#     # Extract features
#     try:
#         extractor = FeatureExtractor()
#         features = extractor.extract_features(test_audio)
        
#         print(f"✅ Features extracted successfully")
#         print(f"   Shape: {features.shape}")
#         print(f"   Expected: (time_frames, 1320)")
        
#         if features.shape[1] == 1320:
#             print("✅ Feature dimensions are correct")
#             return True
#         else:
#             print("❌ Feature dimensions are incorrect")
#             return False
            
#     except Exception as e:
#         print(f"❌ Feature extraction failed: {e}")
#         return False

# def test_data_loading():
#     """Test loading data from TSV"""
#     print("\n📁 Testing TSV Data Loading...")
    
#     from data_loader_m4a import load_sanskrit_data_from_tsv
    
#     tsv_path = './data/sanskrit.tsv'
#     if not os.path.exists(tsv_path):
#         # Create test TSV
#         test_tsv = """844424932806112-643-f.m4a	किन्तु एषः यल्लाप्रगडसुब्बरावः जीवनस्य बहुभागम् अमेरिकादेशे एव अयापयत्
# 844424932812543-643-f.m4a	तस्य मनः अपि इन्द्रियैः सह युक्तं सत् बुद्धिं विचलितां कर्तुं न प्रभवति"""
        
#         with open(tsv_path, 'w', encoding='utf-8') as f:
#             f.write(test_tsv)
#         print(f"   Created test TSV: {tsv_path}")
    
#     try:
#         audio_files, transcripts = load_sanskrit_data_from_tsv(
#             tsv_path, './data/sanskrit_audio', max_utterances=10
#         )
        
#         print(f"✅ Loaded {len(audio_files)} audio files")
#         print(f"✅ Loaded {len(transcripts)} transcripts")
        
#         if len(audio_files) == len(transcripts) and len(audio_files) > 0:
#             print("✅ Data loading successful")
#             print(f"\n   Sample:")
#             print(f"   Audio: {os.path.basename(audio_files[0])}")
#             print(f"   Text: {transcripts[0][:50]}...")
#             return True
#         else:
#             print("❌ Data loading failed")
#             return False
            
#     except Exception as e:
#         print(f"❌ Data loading error: {e}")
#         return False

# def test_model_creation():
#     """Test model initialization"""
#     print("\n🤖 Testing Model Creation...")
    
#     from models import GRLModel, DSNModel
#     from code.ASR_m.temp_model.config import get_config
    
#     config = get_config()
    
#     try:
#         # Test GRL
#         grl_model = GRLModel(
#             input_dim=config['input_dim'],
#             hidden_dim=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
#         grl_params = sum(p.numel() for p in grl_model.parameters())
#         print(f"✅ GRL model created: {grl_params:,} parameters")
        
#         # Test DSN
#         dsn_model = DSNModel(
#             input_dim=config['input_dim'],
#             private_hidden=config['private_hidden'],
#             shared_hidden=config['hidden_dim'],
#             num_senones=config['num_senones'],
#             num_domains=config['num_domains']
#         )
#         dsn_params = sum(p.numel() for p in dsn_model.parameters())
#         print(f"✅ DSN model created: {dsn_params:,} parameters")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Model creation failed: {e}")
#         return False

# def main():
#     print("""
# ╔══════════════════════════════════════════════════════════════╗
# ║              M4A Audio Processing Test Suite                 ║
# ╚══════════════════════════════════════════════════════════════╝

# This will test all components of the ASR pipeline.
# """)
    
#     tests = [
#         ("M4A Conversion", test_m4a_conversion),
#         ("Feature Extraction", test_feature_extraction),
#         ("Data Loading", test_data_loading),
#         ("Model Creation", test_model_creation)
#     ]
    
#     results = []
    
#     for name, test_func in tests:
#         try:
#             result = test_func()
#             results.append((name, result))
#         except Exception as e:
#             print(f"\n❌ {name} crashed: {e}")
#             results.append((name, False))
    
#     # Summary
#     print("\n" + "="*60)
#     print("TEST SUMMARY")
#     print("="*60)
    
#     passed = sum(1 for _, result in results if result)
#     total = len(results)
    
#     for name, result in results:
#         status = "✅ PASSED" if result else "❌ FAILED"
#         print(f"{name:.<40} {status}")
    
#     print(f"\nTotal: {passed}/{total} tests passed")
    
#     if passed == total:
#         print("\n🎉 All tests passed! You're ready to start training.")
#         print("\nNext step: Run the training script")
#         print("python main_tsv.py --hindi-tsv ./data/hindi.tsv --sanskrit-tsv ./data/sanskrit.tsv")
#     else:
#         print("\n⚠️  Some tests failed. Please fix the issues above.")

# if __name__ == "__main__":
#     main()
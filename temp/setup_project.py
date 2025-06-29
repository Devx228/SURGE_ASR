# # setup_project.py
# #!/usr/bin/env python3
# """
# Setup script for Sanskrit ASR project
# """

# import os
# import sys
# import subprocess
# import logging
# from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_python_version():
#     """Check if Python version is compatible"""
#     if sys.version_info < (3, 7):
#         logger.error("Python 3.7 or higher is required")
#         return False
#     logger.info(f"✅ Python version: {sys.version}")
#     return True

# def install_requirements():
#     """Install required packages"""
#     requirements = [
#         'torch>=1.9.0',
#         'torchaudio>=0.9.0', 
#         'numpy>=1.21.0',
#         'pandas>=1.3.0',
#         'tqdm>=4.62.0',
#         'matplotlib>=3.4.0',
#         'seaborn>=0.11.0',
#         'editdistance>=0.5.3',
#         'soundfile>=0.10.0',
#         'librosa>=0.8.0',
#         'pydub>=0.25.0'
#     ]
    
#     logger.info("Installing required packages...")
    
#     for req in requirements:
#         try:
#             subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
#             logger.info(f"✅ Installed {req}")
#         except subprocess.CalledProcessError:
#             logger.error(f"❌ Failed to install {req}")
#             return False
    
#     return True

# def check_ffmpeg():
#     """Check if ffmpeg is available"""
#     try:
#         subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
#         logger.info("✅ ffmpeg is available")
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         logger.warning("⚠️ ffmpeg not found")
#         logger.info("Install ffmpeg for M4A support:")
#         logger.info("  Ubuntu/Debian: sudo apt-get install ffmpeg")
#         logger.info("  macOS: brew install ffmpeg")
#         logger.info("  Windows: Download from https://ffmpeg.org/")
#         logger.info("  Alternative: Will use pydub as fallback")
#         return False

# def create_directory_structure():
#     """Create project directory structure"""
#     dirs = [
#         './data',
#         './data/hindi_audio',
#         './data/sanskrit_audio',
#         './data/hindi_wav',
#         './data/sanskrit_wav',
#         './data/alignments',
#         './data/cache',
#         './data/cache/features',
#         './outputs',
#         './logs',
#         './temp'
#     ]
    
#     logger.info("Creating directory structure...")
#     for d in dirs:
#         os.makedirs(d, exist_ok=True)
#         logger.info(f"✅ Created {d}")

# def create_sample_tsv_files():
#     """Create sample TSV files to show format"""
    
#     # Sample Hindi TSV
#     hindi_sample = """audio001.m4a	यह एक हिंदी वाक्य है
# audio002.m4a	मैं हिंदी में बोल रहा हूं
# audio003.m4a	यह प्रशिक्षण का डेटा है
# audio004.m4a	हम भाषा पहचान सीख रहे हैं
# audio005.m4a	यह स्वचालित भाषण पहचान है"""
    
#     # Sample Sanskrit TSV (using realistic Sanskrit)
#     sanskrit_sample = """844424932806112-643-f.m4a	किन्तु एषः यल्लाप्रगडसुब्बरावः जीवनस्य बहुभागम् अमेरिकादेशे एव अयापयत्
# 844424932812543-643-f.m4a	तस्य मनः अपि इन्द्रियैः सह युक्तं सत् बुद्धिं विचलितां कर्तुं न प्रभवति
# 844424932880843-1102-f.m4a	५१ ॥ ॐ श्रीं क्रीं हूँ च स्मेरास्या ॐ श्रीं स्मरविवद्धिनी
# 844424932814588-643-f.m4a	यदि विदितं स नरः स्वकमोहं तरति शिवं विशति प्रियरूपम् ॥
# 844424932855748-643-f.m4a	सद्यः काले श्री मुरुघराजेन्द्रस्वामी मल्लाडिहळ्ळि वर्तमानानां सर्वसङ्घटनानां दायित्वं स्वीकृत्य प्रचालयन् अस्ति"""
    
#     # Save TSV files
#     with open('./data/hindi_sample.tsv', 'w', encoding='utf-8') as f:
#         f.write(hindi_sample)
    
#     with open('./data/sanskrit_sample.tsv', 'w', encoding='utf-8') as f:
#         f.write(sanskrit_sample)
    
#     logger.info("✅ Created sample TSV files")
#     logger.info("  - ./data/hindi_sample.tsv")
#     logger.info("  - ./data/sanskrit_sample.tsv")

# def create_vocabulary_file():
#     """Create a basic vocabulary file"""
#     vocab = [
#         '<unk>', '<pad>', '<sos>', '<eos>',
#         # Hindi words
#         'यह', 'एक', 'है', 'हैं', 'की', 'के', 'को', 'में', 'से', 'पर',
#         'और', 'या', 'तो', 'जो', 'कि', 'अगर', 'तब', 'फिर', 'अब', 'यहाँ', 'वहाँ',
#         'कैसे', 'क्यों', 'कब', 'कहाँ', 'क्या', 'कौन', 'कितना', 'कितनी', 'कुछ', 'सब',
#         'मैं', 'तुम', 'वह', 'हम', 'आप', 'वे', 'मेरा', 'तुम्हारा', 'उसका', 'हमारा',
#         'बोल', 'बोलना', 'कहना', 'सुनना', 'देखना', 'आना', 'जाना', 'करना', 'होना',
#         'वाक्य', 'शब्द', 'भाषा', 'हिंदी', 'संस्कृत', 'प्रशिक्षण', 'डेटा', 'मॉडल',
#         # Sanskrit words  
#         'इदं', 'तत्', 'एतत्', 'वाक्यम्', 'अस्ति', 'भवति', 'गच्छति', 'आगच्छति',
#         'श्री', 'ॐ', 'नमः', 'स्वाहा', 'मन्त्र', 'यज्ञ', 'होम', 'पूजा', 'देव', 'देवी',
#         'किन्तु', 'एषः', 'तस्य', 'मनः', 'अपि', 'सत्', 'न', 'यदि', 'स', 'नरः',
#         'जीवनस्य', 'बहुभागम्', 'अमेरिकादेशे', 'एव', 'युक्तं', 'बुद्धिं', 'विचलितां',
#         'कर्तुं', 'प्रभवति', 'विदितं', 'स्वकमोहं', 'तरति', 'शिवं', 'विशति', 'प्रियरूपम्'
#     ]
    
#     with open('./data/vocab.txt', 'w', encoding='utf-8') as f:
#         for word in vocab:
#             f.write(word + '\n')
    
#     logger.info(f"✅ Created vocabulary file with {len(vocab)} words")

# def verify_setup():
#     """Verify that setup is complete"""
#     logger.info("\nVerifying setup...")
    
#     # Check directories
#     required_dirs = [
#         './data', './data/hindi_audio', './data/sanskrit_audio',
#         './outputs', './data/cache'
#     ]
    
#     for d in required_dirs:
#         if os.path.exists(d):
#             logger.info(f"✅ {d}")
#         else:
#             logger.error(f"❌ {d}")
#             return False
    
#     # Check sample files
#     if os.path.exists('./data/vocab.txt'):
#         logger.info("✅ Vocabulary file")
#     else:
#         logger.error("❌ Vocabulary file")
#         return False
    
#     return True

# def print_next_steps():
#     """Print next steps for the user"""
#     print(f"""
# {'='*60}
# 🎉 SETUP COMPLETE!
# {'='*60}

# 📁 Directory structure created:
#    ./data/hindi_audio/     - Place your Hindi M4A files here
#    ./data/sanskrit_audio/  - Place your Sanskrit M4A files here
#    ./outputs/              - Trained models will be saved here
#    ./data/cache/           - Cached preprocessed data

# 📋 Next steps:

# 1. Prepare your data:
#    a) Place your 3229 Hindi M4A files in ./data/hindi_audio/
#    b) Place your 1620 Sanskrit M4A files in ./data/sanskrit_audio/
#    c) Create ./data/hindi.tsv with format: audio_file.m4a<TAB>transcript
#    d) Create ./data/sanskrit.tsv with format: audio_file.m4a<TAB>transcript

# 2. Run preprocessing (one-time setup):
#    python main_fixed.py --preprocess-only

# 3. Start training:
#    python main_fixed.py --model both --epochs 10

# 4. Quick test (fewer epochs):
#    python main_fixed.py --model GRL --epochs 5

# 🔧 Useful commands:

# # Preprocess data only
# python main_fixed.py --preprocess-only

# # Train specific model
# python main_fixed.py --model DSN --epochs 10

# # Resume training (skip preprocessing)
# python main_fixed.py --skip-preprocess --model both

# # Custom settings
# python main_fixed.py --model DSN --epochs 20 --batch-size 8 --learning-rate 0.0005

# 📊 Expected results:
#    - Baseline (Hindi only): ~24.58% WER
#    - With UDA (GRL/DSN): ~17-18% WER

# ⚠️  Important notes:
#    - First run will take time to convert M4A files to WAV
#    - Converted files are cached to avoid re-conversion
#    - Basic alignments are generated automatically
#    - For better results, consider using Kaldi alignments later

# {'='*60}
# """)

# def main():
#     print("""
# ╔══════════════════════════════════════════════════════════════╗
# ║           Sanskrit ASR Project Setup                         ║
# ║           Cross-lingual ASR using UDA                        ║
# ╚══════════════════════════════════════════════════════════════╝
# """)
    
#     # Check Python version
#     if not check_python_version():
#         sys.exit(1)
    
#     # Install requirements
#     install_response = input("Install required Python packages? (y/n): ").lower()
#     if install_response == 'y':
#         if not install_requirements():
#             logger.error("Failed to install requirements")
#             sys.exit(1)
    
#     # Check ffmpeg
#     check_ffmpeg()
    
#     # Create directory structure
#     create_directory_structure()
    
#     # Create sample files
#     create_sample_tsv_files()
#     create_vocabulary_file()
    
#     # Verify setup
#     if verify_setup():
#         logger.info("✅ Setup verification passed")
#     else:
#         logger.error("❌ Setup verification failed")
#         sys.exit(1)
    
#     # Print next steps
#     print_next_steps()

# if __name__ == "__main__":
#     main()
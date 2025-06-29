# #!/usr/bin/env python3
# """
# Quick start script for training Sanskrit ASR with TSV data
# """

# import os
# import sys

# def create_sample_tsv_files():
#     """Create sample TSV files to show the format"""
    
#     # Sample Hindi TSV
#     hindi_sample = """audio1.m4a	यह एक नमूना वाक्य है
# audio2.m4a	मैं हिंदी में बोल रहा हूं
# audio3.m4a	यह प्रशिक्षण डेटा है"""
    
#     # Sample Sanskrit TSV (using your data)
#     sanskrit_sample = """844424932806112-643-f.m4a	किन्तु एषः यल्लाप्रगडसुब्बरावः जीवनस्य बहुभागम् अमेरिकादेशे एव अयापयत्
# 844424932812543-643-f.m4a	तस्य मनः अपि इन्द्रियैः सह युक्तं सत् बुद्धिं विचलितां कर्तुं न प्रभवति
# 844424932880843-1102-f.m4a	५१ ॥ ॐ श्रीं क्रीं हूँ च स्मेरास्या ॐ श्रीं स्मरविवद्धिनी
# 844424932814588-643-f.m4a	यदि विदितं स नरः स्वकमोहं तरति शिवं विशति प्रियरूपम् ॥
# 844424932855748-643-f.m4a	सद्यः काले श्री मुरुघराजेन्द्रस्वामी मल्लाडिहळ्ळि वर्तमानानां सर्वसङ्घटनानां दायित्वं स्वीकृत्य प्रचालयन् अस्ति"""
    
#     os.makedirs('./data', exist_ok=True)
    
#     with open('./data/hindi_sample.tsv', 'w', encoding='utf-8') as f:
#         f.write(hindi_sample)
    
#     with open('./data/sanskrit_sample.tsv', 'w', encoding='utf-8') as f:
#         f.write(sanskrit_sample)
    
#     print("✅ Created sample TSV files in ./data/")

# def main():
#     print("""
# ╔══════════════════════════════════════════════════════════════╗
# ║        Sanskrit ASR using UDA - Quick Start Guide            ║
# ╚══════════════════════════════════════════════════════════════╝

# This will help you train Sanskrit ASR models using your TSV data.

# 📋 Your data format:
#    audio_file.m4a<TAB>transcript_in_devanagari

# 🎯 Expected results (from paper):
#    - Baseline: 24.58% WER
#    - GRL: 17.87% WER (-6.71%)
#    - DSN: 17.26% WER (-7.32%)
# """)
    
#     # Step 1: Check dependencies
#     print("\n" + "="*60)
#     print("Step 1: Checking Dependencies")
#     print("="*60)
    
#     required_packages = {
#         'torch': 'PyTorch',
#         'torchaudio': 'TorchAudio',
#         'numpy': 'NumPy',
#         'pandas': 'Pandas',
#         'tqdm': 'TQDM',
#         'soundfile': 'SoundFile'
#     }
    
#     missing = []
#     for package, name in required_packages.items():
#         try:
#             __import__(package)
#             print(f"✅ {name} is installed")
#         except ImportError:
#             print(f"❌ {name} is NOT installed")
#             missing.append(package)
    
#     if missing:
#         print(f"\n⚠️  Install missing packages:")
#         print(f"   pip install {' '.join(missing)}")
#         return
    
#     # Check for audio conversion tools
#     import subprocess
#     try:
#         subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
#         print("✅ ffmpeg is installed (for M4A conversion)")
#     except:
#         print("⚠️  ffmpeg not found - install it or use pydub")
#         print("   Ubuntu: sudo apt-get install ffmpeg")
#         print("   macOS: brew install ffmpeg")
#         print("   Alternative: pip install pydub")
    
#     # Step 2: Setup directories
#     print("\n" + "="*60)
#     print("Step 2: Setting Up Directory Structure")
#     print("="*60)
    
#     dirs = [
#         './data',
#         './data/hindi_audio',
#         './data/sanskrit_audio',
#         './data/alignments',
#         './outputs'
#     ]
    
#     for d in dirs:
#         os.makedirs(d, exist_ok=True)
#         print(f"✅ Created {d}")
    
#     # Step 3: Check for TSV files
#     print("\n" + "="*60)
#     print("Step 3: Checking for TSV Data Files")
#     print("="*60)
    
#     tsv_files = {
#         'Hindi': './data/hindi.tsv',
#         'Sanskrit': './data/sanskrit.tsv'
#     }
    
#     all_found = True
#     for lang, path in tsv_files.items():
#         if os.path.exists(path):
#             with open(path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#             print(f"✅ {lang} TSV found: {len(lines)} utterances")
#         else:
#             print(f"❌ {lang} TSV not found at {path}")
#             all_found = False
    
#     if not all_found:
#         print("\n📝 Creating sample TSV files...")
#         create_sample_tsv_files()
#         print("\nPlease replace these with your actual data:")
#         print("  - ./data/hindi.tsv")
#         print("  - ./data/sanskrit.tsv")
#         return
    
#     # Step 4: Quick training command
#     print("\n" + "="*60)
#     print("Step 4: Training Commands")
#     print("="*60)
    
#     print("\n🚀 To start training, run one of these commands:\n")
    
#     print("# Train both GRL and DSN models (recommended):")
#     print("python main_tsv.py --hindi-tsv ./data/hindi.tsv --sanskrit-tsv ./data/sanskrit.tsv --model both\n")
    
#     print("# Train only GRL model:")
#     print("python main_tsv.py --hindi-tsv ./data/hindi.tsv --sanskrit-tsv ./data/sanskrit.tsv --model GRL\n")
    
#     print("# Train only DSN model:")
#     print("python main_tsv.py --hindi-tsv ./data/hindi.tsv --sanskrit-tsv ./data/sanskrit.tsv --model DSN\n")
    
#     print("# Quick test with 5 epochs:")
#     print("python main_tsv.py --hindi-tsv ./data/hindi.tsv --sanskrit-tsv ./data/sanskrit.tsv --model GRL --epochs 5\n")
    
#     # Step 5: Important notes
#     print("\n" + "="*60)
#     print("Step 5: Important Notes")
#     print("="*60)
    
#     print("""
# 📌 Key Points:

# 1. Audio Files:
#    - Place M4A files in ./data/hindi_audio/ and ./data/sanskrit_audio/
#    - Files will be automatically converted to 8kHz WAV

# 2. Training Time:
#    - Full training (20 epochs) takes several hours
#    - Start with fewer epochs (--epochs 5) for testing

# 3. Memory Issues:
#    - Reduce batch size if you run out of memory: --batch-size 16
#    - Reduce max utterances in config.py

# 4. Alignments:
#    - The script will generate pseudo-alignments automatically
#    - For better results, use Kaldi to generate real alignments

# 5. Expected Output:
#    - Models saved in ./outputs/
#    - Training logs show loss values
#    - Lower loss = better model
# """)
    
#     # Interactive option
#     response = input("\n🎯 Do you want to start training now? (y/n): ").lower()
    
#     if response == 'y':
#         print("\n📊 Which model do you want to train?")
#         print("1. GRL (Gradient Reversal Layer)")
#         print("2. DSN (Domain Separation Networks)")
#         print("3. Both (recommended)")
        
#         choice = input("\nEnter your choice (1/2/3): ")
#         model_map = {'1': 'GRL', '2': 'DSN', '3': 'both'}
#         model = model_map.get(choice, 'both')
        
#         epochs = input("\nNumber of epochs (default 20, use 5 for quick test): ")
#         epochs = int(epochs) if epochs.isdigit() else 20
        
#         # Check if TSV files exist
#         hindi_tsv = './data/hindi.tsv'
#         sanskrit_tsv = './data/sanskrit.tsv'
        
#         if not os.path.exists(hindi_tsv) or not os.path.exists(sanskrit_tsv):
#             print("\n❌ TSV files not found! Please add your data first.")
#             return
        
#         # Build command
#         cmd = f"python main_tsv.py --hindi-tsv {hindi_tsv} --sanskrit-tsv {sanskrit_tsv} --model {model} --epochs {epochs}"
        
#         print(f"\n🚀 Starting training with command:")
#         print(f"   {cmd}")
        
#         # Execute
#         os.system(cmd)
#     else:
#         print("\n✅ Setup complete! Run the training commands above when ready.")

# if __name__ == "__main__":
#     main()
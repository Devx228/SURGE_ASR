# import torch
# import numpy as np
# import os
# import logging
# from typing import List, Tuple

# logger = logging.getLogger(__name__)

# def load_hindi_data(data_dir: str, alignments_dir: str, max_utterances: int = 15000):
#     """
#     Load Hindi training data with senone alignments from HMM-GMM system
    
#     Args:
#         data_dir: Directory containing Hindi audio files
#         alignments_dir: Directory containing senone alignment files
#         max_utterances: Maximum number of utterances to load
    
#     Returns:
#         audio_files: List of audio file paths
#         transcripts: List of transcription strings
#         senones: List of senone alignment arrays
#     """
#     logger.info(f"Loading Hindi data from {data_dir}")
    
#     audio_files = []
#     transcripts = []
#     senones = []
    
#     # Find all audio files
#     audio_extensions = ['.wav', '.flac', '.mp3']
#     all_audio_files = []
    
#     for ext in audio_extensions:
#         all_audio_files.extend(
#             [f for f in os.listdir(data_dir) if f.endswith(ext)]
#         )
    
#     # Randomly select utterances
#     np.random.shuffle(all_audio_files)
#     selected_files = all_audio_files[:max_utterances]
    
#     for audio_file in selected_files:
#         file_id = os.path.splitext(audio_file)[0]
        
#         # Audio file path
#         audio_path = os.path.join(data_dir, audio_file)
        
#         # Transcript file (assuming .txt extension)
#         transcript_path = os.path.join(data_dir, f"{file_id}.txt")
        
#         # Senone alignment file (assuming .ali extension)
#         alignment_path = os.path.join(alignments_dir, f"{file_id}.ali")
        
#         # Check if all files exist
#         if (os.path.exists(audio_path) and 
#             os.path.exists(transcript_path) and 
#             os.path.exists(alignment_path)):
            
#             # Load transcript
#             with open(transcript_path, 'r', encoding='utf-8') as f:
#                 transcript = f.read().strip()
            
#             # Load senone alignments (assuming space-separated integers)
#             senone_alignment = np.loadtxt(alignment_path, dtype=int)
            
#             audio_files.append(audio_path)
#             transcripts.append(transcript)
#             senones.append(senone_alignment)
    
#     logger.info(f"Loaded {len(audio_files)} Hindi utterances")
#     return audio_files, transcripts, senones

# def load_sanskrit_data(data_dir: str, max_utterances: int = 2837):
#     """
#     Load Sanskrit training data (target domain - no senone labels needed)
    
#     Args:
#         data_dir: Directory containing Sanskrit audio files
#         max_utterances: Maximum number of utterances to load
    
#     Returns:
#         audio_files: List of audio file paths
#         transcripts: List of transcription strings
#     """
#     logger.info(f"Loading Sanskrit data from {data_dir}")
    
#     audio_files = []
#     transcripts = []
    
#     # Find all audio files
#     audio_extensions = ['.wav', '.flac', '.mp3']
#     all_audio_files = []
    
#     for ext in audio_extensions:
#         all_audio_files.extend(
#             [f for f in os.listdir(data_dir) if f.endswith(ext)]
#         )
    
#     # Randomly select utterances
#     np.random.shuffle(all_audio_files)
#     selected_files = all_audio_files[:max_utterances]
    
#     for audio_file in selected_files:
#         file_id = os.path.splitext(audio_file)[0]
        
#         # Audio file path
#         audio_path = os.path.join(data_dir, audio_file)
        
#         # Transcript file
#         transcript_path = os.path.join(data_dir, f"{file_id}.txt")
        
#         if os.path.exists(audio_path) and os.path.exists(transcript_path):
#             # Load transcript
#             with open(transcript_path, 'r', encoding='utf-8') as f:
#                 transcript = f.read().strip()
            
#             audio_files.append(audio_path)
#             transcripts.append(transcript)
    
#     logger.info(f"Loaded {len(audio_files)} Sanskrit utterances")
#     return audio_files, transcripts

# def load_sanskrit_test_data(data_dir: str):
#     """
#     Load Sanskrit test data with senone alignments for evaluation
    
#     Args:
#         data_dir: Directory containing Sanskrit test files
    
#     Returns:
#         audio_files: List of audio file paths
#         transcripts: List of transcription strings  
#         senones: List of senone alignment arrays
#     """
#     logger.info(f"Loading Sanskrit test data from {data_dir}")
    
#     audio_files = []
#     transcripts = []
#     senones = []
    
#     audio_extensions = ['.wav', '.flac', '.mp3']
#     all_audio_files = []
    
#     for ext in audio_extensions:
#         all_audio_files.extend(
#             [f for f in os.listdir(data_dir) if f.endswith(ext)]
#         )
    
#     for audio_file in all_audio_files:
#         file_id = os.path.splitext(audio_file)[0]
#         audio_path = os.path.join(data_dir, audio_file)
#         transcript_path = os.path.join(data_dir, f"{file_id}.txt")
        
#         if os.path.exists(audio_path) and os.path.exists(transcript_path):
#             with open(transcript_path, 'r', encoding='utf-8') as f:
#                 transcript = f.read().strip()
            
#             # For evaluation, you would need proper senone alignments
#             # This is a placeholder - would need alignment from trained model
#             dummy_senones = np.array([])  # Replace with actual alignments
            
#             audio_files.append(audio_path)
#             transcripts.append(transcript)
#             senones.append(dummy_senones)
    
#     logger.info(f"Loaded {len(audio_files)} Sanskrit test utterances")
#     return audio_files, transcripts, senones

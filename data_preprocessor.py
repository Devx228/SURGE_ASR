# data_preprocessor.py
import os
import subprocess
import logging
import pickle
import numpy as np
import torchaudio
from pathlib import Path
from tqdm import tqdm
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Preprocessor to convert M4A files to WAV once and cache features
    """
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = config['cache_dir']
        self.hindi_wav_dir = config['hindi_wav_dir']
        self.sanskrit_wav_dir = config['sanskrit_wav_dir']
        
        # Create directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.hindi_wav_dir, exist_ok=True)
        os.makedirs(self.sanskrit_wav_dir, exist_ok=True)
        os.makedirs(config['senone_alignments_dir'], exist_ok=True)
        
        # Cache files
        self.conversion_cache_file = os.path.join(self.cache_dir, 'conversion_cache.json')
        self.alignment_cache_file = os.path.join(self.cache_dir, 'alignment_cache.pkl')
        
        # Load existing caches
        self.conversion_cache = self.load_conversion_cache()
        
    def load_conversion_cache(self):
        """Load conversion cache to avoid re-converting files"""
        if os.path.exists(self.conversion_cache_file):
            with open(self.conversion_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_conversion_cache(self):
        """Save conversion cache"""
        with open(self.conversion_cache_file, 'w') as f:
            json.dump(self.conversion_cache, f)
    
    def get_file_hash(self, file_path):
        """Get hash of file for caching"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def convert_m4a_to_wav(self, m4a_path, output_dir):
        """Convert M4A to WAV with caching"""
        base_name = os.path.splitext(os.path.basename(m4a_path))[0]
        wav_path = os.path.join(output_dir, f"{base_name}.wav")
        
        # Check if already converted and file hasn't changed
        file_hash = self.get_file_hash(m4a_path)
        cache_key = m4a_path
        
        if (cache_key in self.conversion_cache and 
            self.conversion_cache[cache_key]['hash'] == file_hash and
            os.path.exists(wav_path)):
            logger.debug(f"Using cached WAV: {wav_path}")
            return wav_path
        
        # Convert using ffmpeg
        cmd = [
            'ffmpeg', '-i', m4a_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',  # mono
            '-ar', str(self.config['sample_rate']),  # 8kHz
            wav_path,
            '-y'  # overwrite
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Converted {m4a_path} to {wav_path}")
            
            # Update cache
            self.conversion_cache[cache_key] = {
                'hash': file_hash,
                'wav_path': wav_path,
                'converted_at': str(Path(wav_path).stat().st_mtime)
            }
            self.save_conversion_cache()
            
            return wav_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed for {m4a_path}: {e.stderr}")
            return self.convert_with_pydub(m4a_path, wav_path)
    
    def convert_with_pydub(self, m4a_path, wav_path):
        """Fallback conversion using pydub"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio = audio.set_channels(1)  # mono
            audio = audio.set_frame_rate(self.config['sample_rate'])  # 8kHz
            audio.export(wav_path, format="wav")
            logger.info(f"Converted with pydub: {wav_path}")
            return wav_path
        except Exception as e:
            logger.error(f"Pydub conversion failed: {e}")
            raise
    
    def preprocess_audio_files(self, tsv_path, audio_dir, output_wav_dir, language):
        """
        Preprocess all audio files from TSV
        Returns: list of (wav_path, transcript) tuples
        """
        logger.info(f"Preprocessing {language} audio files...")
        
        # Read TSV file
        audio_transcript_pairs = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"Line {line_num}: Invalid format (expected 2 columns, got {len(parts)})")
                    continue
                    
                audio_file, transcript = parts
                audio_path = os.path.join(audio_dir, audio_file)
                
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue
                
                audio_transcript_pairs.append((audio_path, transcript))
        
        logger.info(f"Found {len(audio_transcript_pairs)} valid {language} entries")
        
        # Convert audio files
        processed_pairs = []
        for audio_path, transcript in tqdm(audio_transcript_pairs, desc=f"Converting {language} audio"):
            try:
                if audio_path.endswith('.m4a'):
                    wav_path = self.convert_m4a_to_wav(audio_path, output_wav_dir)
                else:
                    # Copy or symlink if already WAV
                    wav_path = audio_path
                
                # Verify the WAV file
                if self.verify_wav_file(wav_path):
                    processed_pairs.append((wav_path, transcript))
                else:
                    logger.warning(f"Invalid WAV file: {wav_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_pairs)} {language} files")
        return processed_pairs
    
    def verify_wav_file(self, wav_path):
        """Verify that WAV file is valid and readable"""
        try:
            waveform, sample_rate = torchaudio.load(wav_path)
            if waveform.shape[1] > 0 and sample_rate > 0:
                return True
        except Exception as e:
            logger.debug(f"WAV verification failed for {wav_path}: {e}")
        return False
    
    def generate_basic_alignments(self, audio_transcript_pairs, language):
        """
        Generate basic hash-based alignments for training
        This is temporary until we implement Kaldi alignments
        """
        logger.info(f"Generating basic alignments for {language}...")
        
        alignments = []
        for wav_path, transcript in tqdm(audio_transcript_pairs, desc=f"Generating {language} alignments"):
            try:
                # Get audio duration
                waveform, sample_rate = torchaudio.load(wav_path)
                duration = waveform.shape[1] / sample_rate
                
                # Estimate frames (100 frames per second)
                num_frames = max(10, int(duration * 100))
                
                # Generate alignment based on transcript
                words = transcript.split()
                if not words:
                    # Silence alignment
                    alignment = np.zeros(num_frames, dtype=np.int32)
                else:
                    alignment = self.create_word_based_alignment(words, num_frames)
                
                # Save alignment file
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                ali_file = os.path.join(self.config['senone_alignments_dir'], f'{base_name}.ali')
                np.savetxt(ali_file, alignment, fmt='%d')
                
                alignments.append(alignment)
                
            except Exception as e:
                logger.error(f"Failed to generate alignment for {wav_path}: {e}")
                # Create dummy alignment
                dummy_alignment = np.zeros(50, dtype=np.int32)
                alignments.append(dummy_alignment)
        
        logger.info(f"Generated {len(alignments)} alignments for {language}")
        return alignments
    
    def create_word_based_alignment(self, words, num_frames):
        """Create a more sophisticated alignment based on words"""
        alignment = np.zeros(num_frames, dtype=np.int32)
        
        frames_per_word = max(1, num_frames // len(words))
        
        for i, word in enumerate(words):
            start_frame = i * frames_per_word
            end_frame = min((i + 1) * frames_per_word, num_frames)
            
            # Generate senone sequence for this word
            word_hash = abs(hash(word))
            base_senone = word_hash % (self.config['num_senones'] // 3)
            
            # Create varied senones for the word duration
            for frame in range(start_frame, end_frame):
                frame_offset = (frame - start_frame) % 10
                senone = (base_senone + frame_offset) % self.config['num_senones']
                alignment[frame] = senone
        
        return alignment

def preprocess_all_data(config):
    """
    Main preprocessing function
    """
    logger.info("Starting data preprocessing...")
    
    preprocessor = AudioPreprocessor(config)
    
    # Define TSV files
    hindi_tsv = os.path.join(config['data_dir'], 'hindi.tsv')
    sanskrit_tsv = os.path.join(config['data_dir'], 'sanskrit.tsv')
    
    # Check if TSV files exist
    if not os.path.exists(hindi_tsv):
        logger.error(f"Hindi TSV file not found: {hindi_tsv}")
        return None, None
    
    if not os.path.exists(sanskrit_tsv):
        logger.error(f"Sanskrit TSV file not found: {sanskrit_tsv}")
        return None, None
    
    # Preprocess Hindi data
    hindi_audio_dir = os.path.join(config['data_dir'], 'hindi_audio')
    hindi_pairs = preprocessor.preprocess_audio_files(
        hindi_tsv, hindi_audio_dir, config['hindi_wav_dir'], 'Hindi'
    )
    
    # Preprocess Sanskrit data
    sanskrit_audio_dir = os.path.join(config['data_dir'], 'sanskrit_audio')
    sanskrit_pairs = preprocessor.preprocess_audio_files(
        sanskrit_tsv, sanskrit_audio_dir, config['sanskrit_wav_dir'], 'Sanskrit'
    )
    
    # Generate alignments
    hindi_alignments = preprocessor.generate_basic_alignments(hindi_pairs, 'Hindi')
    sanskrit_alignments = preprocessor.generate_basic_alignments(sanskrit_pairs, 'Sanskrit')
    
    # Save processed data info
    processed_data = {
        'hindi_pairs': hindi_pairs,
        'sanskrit_pairs': sanskrit_pairs,
        'hindi_alignments': hindi_alignments,
        'sanskrit_alignments': sanskrit_alignments,
        'config': config
    }
    
    cache_file = os.path.join(config['cache_dir'], 'processed_data.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    logger.info("Data preprocessing completed!")
    logger.info(f"Hindi: {len(hindi_pairs)} files")
    logger.info(f"Sanskrit: {len(sanskrit_pairs)} files")
    
    return hindi_pairs, sanskrit_pairs

if __name__ == "__main__":
    from fixed_config import get_config
    config = get_config()
    preprocess_all_data(config)
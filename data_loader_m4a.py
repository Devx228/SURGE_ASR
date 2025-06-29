import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import subprocess
import tempfile
import logging
from typing import List, Tuple, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class M4ADataLoader:
    """Handle M4A audio files and TSV transcripts"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def convert_m4a_to_wav(self, m4a_path: str, wav_path: str = None) -> str:
        """Convert M4A to WAV using ffmpeg"""
        if wav_path is None:
            base_name = os.path.basename(m4a_path).replace('.m4a', '.wav')
            wav_path = os.path.join(self.temp_dir, base_name)
        
        # Use ffmpeg to convert m4a to wav
        cmd = [
            'ffmpeg', '-i', m4a_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',  # mono
            '-ar', '8000',  # 8kHz as per paper
            wav_path,
            '-y'  # overwrite
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return wav_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {m4a_path}: {e}")
            # Try using pydub as fallback
            return self.convert_with_pydub(m4a_path, wav_path)
    
    def convert_with_pydub(self, m4a_path: str, wav_path: str) -> str:
        """Fallback conversion using pydub"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio = audio.set_channels(1)  # mono
            audio = audio.set_frame_rate(8000)  # 8kHz
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.error(f"Pydub conversion also failed: {e}")
            raise
    
    def load_data_from_tsv(self, tsv_path: str, audio_dir: str = None) -> Tuple[List[str], List[str]]:
        """Load audio paths and transcripts from TSV file"""
        # Read TSV file
        data = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    audio_file, transcript = parts
                    data.append((audio_file, transcript))
        
        audio_files = []
        transcripts = []
        
        # If audio_dir is provided, prepend it to audio file names
        for audio_file, transcript in data:
            if audio_dir:
                audio_path = os.path.join(audio_dir, audio_file)
            else:
                audio_path = audio_file
            
            audio_files.append(audio_path)
            transcripts.append(transcript)
        
        return audio_files, transcripts

def load_hindi_data_from_tsv(tsv_path: str, audio_dir: str, 
                            alignments_dir: str, max_utterances: int = 15000):
    """
    Load Hindi data from TSV file format
    """
    loader = M4ADataLoader()
    audio_files, transcripts = loader.load_data_from_tsv(tsv_path, audio_dir)
    
    # Limit to max_utterances
    if len(audio_files) > max_utterances:
        indices = np.random.choice(len(audio_files), max_utterances, replace=False)
        audio_files = [audio_files[i] for i in indices]
        transcripts = [transcripts[i] for i in indices]
    
    # For now, create dummy senone alignments
    # In practice, you need to generate these using HMM-GMM
    senones = []
    for i, audio_file in enumerate(audio_files):
        # Check if alignment file exists
        base_name = os.path.basename(audio_file).replace('.m4a', '')
        ali_file = os.path.join(alignments_dir, f'{base_name}.ali')
        
        if os.path.exists(ali_file):
            senone_alignment = np.loadtxt(ali_file, dtype=int)
        else:
            # Create dummy alignment based on transcript length
            # This is temporary - you should generate real alignments
            num_frames = len(transcripts[i].split()) * 50  # Rough estimate
            senone_alignment = np.random.randint(0, 3080, size=num_frames)
        
        senones.append(senone_alignment)
    
    logger.info(f"Loaded {len(audio_files)} Hindi utterances from TSV")
    return audio_files, transcripts, senones

def load_sanskrit_data_from_tsv(tsv_path: str, audio_dir: str, 
                               max_utterances: int = 2837):
    """
    Load Sanskrit data from TSV file format
    """
    loader = M4ADataLoader()
    audio_files, transcripts = loader.load_data_from_tsv(tsv_path, audio_dir)
    
    # Limit to max_utterances
    if len(audio_files) > max_utterances:
        indices = np.random.choice(len(audio_files), max_utterances, replace=False)
        audio_files = [audio_files[i] for i in indices]
        transcripts = [transcripts[i] for i in indices]
    
    logger.info(f"Loaded {len(audio_files)} Sanskrit utterances from TSV")
    return audio_files, transcripts

class M4AASRDataset(Dataset):
    """Dataset for M4A audio files"""
    
    def __init__(self, audio_files: List[str], transcripts: List[str], 
                 senones: List[np.ndarray], domain_labels: List[int], 
                 feature_extractor, convert_m4a: bool = True):
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.senones = senones
        self.domain_labels = domain_labels
        self.feature_extractor = feature_extractor
        self.convert_m4a = convert_m4a
        
        if convert_m4a:
            self.loader = M4ADataLoader()
            # Pre-convert all M4A files to WAV for faster loading
            logger.info("Converting M4A files to WAV...")
            self.wav_files = []
            for audio_file in tqdm(audio_files, desc="Converting audio"):
                if audio_file.endswith('.m4a'):
                    wav_file = self.loader.convert_m4a_to_wav(audio_file)
                    self.wav_files.append(wav_file)
                else:
                    self.wav_files.append(audio_file)
        else:
            self.wav_files = self.audio_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Use converted WAV file
        audio_file = self.wav_files[idx]
        
        # Extract features
        try:
            features = self.feature_extractor.extract_features(audio_file)
        except Exception as e:
            logger.error(f"Error extracting features from {audio_file}: {e}")
            # Return dummy features
            features = np.zeros((100, 1320))  # Dummy features
        
        return {
            'features': torch.FloatTensor(features),
            'senones': torch.LongTensor(self.senones[idx]) if self.senones else torch.LongTensor([]),
            'domain': torch.LongTensor([self.domain_labels[idx]]),
            'transcript': self.transcripts[idx]
        }

# Update the config to handle TSV files
def update_config_for_tsv(config: Dict):
    """Update configuration for TSV data format"""
    config.update({
        'hindi_tsv': './data/hindi_train.tsv',
        'sanskrit_tsv': './data/sanskrit_train.tsv',
        'hindi_audio_dir': './data/hindi_audio/',
        'sanskrit_audio_dir': './data/sanskrit_audio/',
    })
    return config

# Quick data preparation script
def prepare_tsv_data(tsv_content: str, output_path: str, 
                     train_ratio: float = 0.8):
    """
    Split TSV data into train/test sets
    
    Args:
        tsv_content: String containing TSV data
        output_path: Base path for output files
        train_ratio: Ratio of training data
    """
    lines = tsv_content.strip().split('\n')
    np.random.shuffle(lines)
    
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    
    # Save train set
    train_path = output_path.replace('.tsv', '_train.tsv')
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    # Save test set
    test_path = output_path.replace('.tsv', '_test.tsv')
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))
    
    logger.info(f"Split data: {len(train_lines)} train, {len(test_lines)} test")
    return train_path, test_path

def generate_alignments_for_m4a(audio_files: List[str], transcripts: List[str], 
                               output_dir: str, num_senones: int = 3080):
    """
    Generate pseudo-alignments for M4A files
    This is a simplified version - real alignments need HMM-GMM
    """
    os.makedirs(output_dir, exist_ok=True)
    loader = M4ADataLoader()
    
    for i, (audio_file, transcript) in enumerate(tqdm(
            zip(audio_files, transcripts), 
            total=len(audio_files), 
            desc="Generating alignments")):
        
        try:
            # Convert to WAV first
            if audio_file.endswith('.m4a'):
                wav_file = loader.convert_m4a_to_wav(audio_file)
            else:
                wav_file = audio_file
            
            # Get audio duration
            waveform, sample_rate = torchaudio.load(wav_file)
            duration = waveform.shape[1] / sample_rate
            
            # Estimate number of frames (100 frames per second)
            num_frames = int(duration * 100)
            
            # Generate pseudo-alignments based on transcript
            words = transcript.split()
            frames_per_word = max(1, num_frames // len(words)) if words else num_frames
            
            senones = []
            for j, word in enumerate(words):
                # Simple hash-based senone assignment
                base_senone = hash(word) % (num_senones // 2)
                for k in range(frames_per_word):
                    offset = k % 20
                    senone = (base_senone + offset) % num_senones
                    senones.append(senone)
            
            # Adjust to match frame count
            if len(senones) < num_frames:
                senones.extend([senones[-1] if senones else 0] * (num_frames - len(senones)))
            else:
                senones = senones[:num_frames]
            
            # Save alignment
            base_name = os.path.basename(audio_file).replace('.m4a', '')
            ali_file = os.path.join(output_dir, f'{base_name}.ali')
            np.savetxt(ali_file, senones, fmt='%d')
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            # Save dummy alignment
            base_name = os.path.basename(audio_file).replace('.m4a', '')
            ali_file = os.path.join(output_dir, f'{base_name}.ali')
            np.savetxt(ali_file, [0], fmt='%d')
    
    logger.info(f"Generated alignments for {len(audio_files)} files")

if __name__ == "__main__":
    # Example: Process your TSV data
    tsv_data = """844424932806112-643-f.m4a	किन्तु एषः यल्लाप्रगडसुब्बरावः जीवनस्य बहुभागम् अमेरिकादेशे एव अयापयत्
844424932812543-643-f.m4a	तस्य मनः अपि इन्द्रियैः सह युक्तं सत् बुद्धिं विचलितां कर्तुं न प्रभवति
844424932880843-1102-f.m4a	५१ ॥ ॐ श्रीं क्रीं हूँ च स्मेरास्या ॐ श्रीं स्मरविवद्धिनी
844424932814588-643-f.m4a	यदि विदितं स नरः स्वकमोहं तरति शिवं विशति प्रियरूपम् ॥
844424932855748-643-f.m4a	सद्यः काले श्री मुरुघराजेन्द्रस्वामी मल्लाडिहळ्ळि वर्तमानानां सर्वसङ्घटनानां दायित्वं स्वीकृत्य प्रचालयन् अस्ति"""
    
    # Save as TSV file
    with open('./data/sanskrit_sample.tsv', 'w', encoding='utf-8') as f:
        f.write(tsv_data)
    
    print("Sample TSV data saved!")
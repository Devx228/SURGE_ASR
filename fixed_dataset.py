# fixed_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple
import logging
import os
import pickle
from feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)

class ASRDataset(Dataset):
    """Enhanced Dataset class with better error handling and caching"""
    
    def __init__(self, audio_files: List[str], transcripts: List[str], 
                 senones: List[np.ndarray], domain_labels: List[int], 
                 feature_extractor: FeatureExtractor, cache_features: bool = True):
        
        assert len(audio_files) == len(transcripts) == len(senones) == len(domain_labels), \
            "All input lists must have the same length"
        
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.senones = senones
        self.domain_labels = domain_labels
        self.feature_extractor = feature_extractor
        self.cache_features = cache_features
        
        # Feature cache
        self.feature_cache = {}
        self.cache_dir = './data/cache/features'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Dataset created with {len(self.audio_files)} samples")
        
    def __len__(self):
        return len(self.audio_files)
    
    def _get_cache_path(self, idx):
        """Get cache file path for features"""
        base_name = os.path.splitext(os.path.basename(self.audio_files[idx]))[0]
        return os.path.join(self.cache_dir, f"{base_name}_features.npy")
    
    def _load_features(self, idx):
        """Load or extract features with caching"""
        cache_path = self._get_cache_path(idx)
        
        # Try to load from cache
        if self.cache_features and os.path.exists(cache_path):
            try:
                features = np.load(cache_path)
                logger.debug(f"Loaded cached features for {self.audio_files[idx]}")
                return features
            except Exception as e:
                logger.warning(f"Failed to load cached features: {e}")
        
        # Extract features
        try:
            features = self.feature_extractor.extract_features(self.audio_files[idx])
            
            # Save to cache
            if self.cache_features:
                try:
                    np.save(cache_path, features)
                    logger.debug(f"Cached features for {self.audio_files[idx]}")
                except Exception as e:
                    logger.warning(f"Failed to cache features: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {self.audio_files[idx]}: {e}")
            # Return dummy features as fallback
            return np.zeros((100, 1320), dtype=np.float32)
    
    def __getitem__(self, idx):
        # Load features
        features = self._load_features(idx)
        
        # Get senone alignment
        senone_seq = self.senones[idx] if len(self.senones[idx]) > 0 else np.array([0])
        
        # Ensure features and senones have compatible lengths
        min_len = min(len(features), len(senone_seq))
        if min_len > 0:
            features = features[:min_len]
            senone_seq = senone_seq[:min_len]
        else:
            # Fallback for empty sequences
            features = np.zeros((1, 1320), dtype=np.float32)
            senone_seq = np.array([0])
        
        return {
            'features': torch.FloatTensor(features),
            'senones': torch.LongTensor(senone_seq),
            'domain': torch.LongTensor([self.domain_labels[idx]]),
            'transcript': self.transcripts[idx],
            'audio_file': self.audio_files[idx]
        }

def collate_fn(batch):
    """Enhanced collate function with better padding handling"""
    features = [item['features'] for item in batch]
    senones = [item['senones'] for item in batch]
    domains = torch.cat([item['domain'] for item in batch])
    transcripts = [item['transcript'] for item in batch]
    audio_files = [item['audio_file'] for item in batch]
    
    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    senones_padded = pad_sequence(senones, batch_first=True, padding_value=-1)
    
    # Create length tensors
    feature_lengths = torch.LongTensor([len(f) for f in features])
    senone_lengths = torch.LongTensor([len(s) for s in senones])
    
    return {
        'features': features_padded,
        'senones': senones_padded,
        'domains': domains,
        'feature_lengths': feature_lengths,
        'senone_lengths': senone_lengths,
        'transcripts': transcripts,
        'audio_files': audio_files
    }

def create_data_loaders(config):
    """Create train and test data loaders from preprocessed data"""
    
    # Load preprocessed data
    cache_file = os.path.join(config['cache_dir'], 'processed_data.pkl')
    if not os.path.exists(cache_file):
        logger.error("Preprocessed data not found. Run data preprocessing first.")
        return None, None, None, None
    
    with open(cache_file, 'rb') as f:
        processed_data = pickle.load(f)
    
    hindi_pairs = processed_data['hindi_pairs']
    sanskrit_pairs = processed_data['sanskrit_pairs']
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        n_mels=config['n_mels'],
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        context_frames=config['context_frames']
    )
    
    # Split data into train/test
    test_split = config.get('test_split', 0.1)
    
    # Split Hindi data
    hindi_split_idx = int(len(hindi_pairs) * (1 - test_split))
    hindi_train_pairs = hindi_pairs[:hindi_split_idx]
    hindi_test_pairs = hindi_pairs[hindi_split_idx:]
    
    # Split Sanskrit data  
    sanskrit_split_idx = int(len(sanskrit_pairs) * (1 - test_split))
    sanskrit_train_pairs = sanskrit_pairs[:sanskrit_split_idx]
    sanskrit_test_pairs = sanskrit_pairs[sanskrit_split_idx:]
    
    # Load alignments for training data
    def load_alignments(pairs, alignments_dir):
        alignments = []
        for wav_path, _ in pairs:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            ali_file = os.path.join(alignments_dir, f'{base_name}.ali')
            
            if os.path.exists(ali_file):
                try:
                    alignment = np.loadtxt(ali_file, dtype=int)
                    if alignment.ndim == 0:  # Single value
                        alignment = np.array([alignment])
                    alignments.append(alignment)
                except Exception as e:
                    logger.warning(f"Failed to load alignment {ali_file}: {e}")
                    alignments.append(np.array([0]))
            else:
                logger.warning(f"Alignment file not found: {ali_file}")
                alignments.append(np.array([0]))
        return alignments
    
    # Load alignments
    hindi_train_alignments = load_alignments(hindi_train_pairs, config['senone_alignments_dir'])
    hindi_test_alignments = load_alignments(hindi_test_pairs, config['senone_alignments_dir'])
    sanskrit_train_alignments = load_alignments(sanskrit_train_pairs, config['senone_alignments_dir'])
    sanskrit_test_alignments = load_alignments(sanskrit_test_pairs, config['senone_alignments_dir'])
    
    # Create datasets
    logger.info("Creating training datasets...")
    
    # Training datasets
    hindi_train_dataset = ASRDataset(
        [pair[0] for pair in hindi_train_pairs],
        [pair[1] for pair in hindi_train_pairs],
        hindi_train_alignments,
        [0] * len(hindi_train_pairs),  # Domain 0 for Hindi
        feature_extractor
    )
    
    sanskrit_train_dataset = ASRDataset(
        [pair[0] for pair in sanskrit_train_pairs],
        [pair[1] for pair in sanskrit_train_pairs],
        sanskrit_train_alignments,
        [1] * len(sanskrit_train_pairs),  # Domain 1 for Sanskrit
        feature_extractor
    )
    
    # Test datasets
    hindi_test_dataset = ASRDataset(
        [pair[0] for pair in hindi_test_pairs],
        [pair[1] for pair in hindi_test_pairs],
        hindi_test_alignments,
        [0] * len(hindi_test_pairs),
        feature_extractor
    )
    
    sanskrit_test_dataset = ASRDataset(
        [pair[0] for pair in sanskrit_test_pairs],
        [pair[1] for pair in sanskrit_test_pairs],
        sanskrit_test_alignments,
        [1] * len(sanskrit_test_pairs),
        feature_extractor
    )
    
    # Create data loaders
    hindi_train_loader = DataLoader(
        hindi_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    sanskrit_train_loader = DataLoader(
        sanskrit_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Combine test datasets for evaluation
    test_audio_files = [pair[0] for pair in hindi_test_pairs + sanskrit_test_pairs]
    test_transcripts = [pair[1] for pair in hindi_test_pairs + sanskrit_test_pairs]
    test_alignments = hindi_test_alignments + sanskrit_test_alignments
    test_domains = [0] * len(hindi_test_pairs) + [1] * len(sanskrit_test_pairs)
    
    test_dataset = ASRDataset(
        test_audio_files,
        test_transcripts,
        test_alignments,
        test_domains,
        feature_extractor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Hindi train: {len(hindi_train_dataset)} samples")
    logger.info(f"  Sanskrit train: {len(sanskrit_train_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    return hindi_train_loader, sanskrit_train_loader, test_loader, (test_audio_files, test_transcripts)

if __name__ == "__main__":
    from fixed_config import get_config
    config = get_config()
    
    # Test data loader creation
    hindi_loader, sanskrit_loader, test_loader, test_data = create_data_loaders(config)
    
    if hindi_loader:
        logger.info("Data loaders created successfully!")
        
        # Test loading a batch
        try:
            batch = next(iter(hindi_loader))
            logger.info(f"Sample batch shape: {batch['features'].shape}")
            logger.info(f"Sample senones shape: {batch['senones'].shape}")
        except Exception as e:
            logger.error(f"Error loading batch: {e}")
    else:
        logger.error("Failed to create data loaders!")
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import json
import numpy as np
from collections import defaultdict
import editdistance
from models import GRLModel, DSNModel
from feature_extraction import FeatureExtractor
from fixed_config import get_config
import os

logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS (to be added to utils.py)
# ============================================================================

def load_vocabulary(vocab_file: str) -> List[str]:
    """Load vocabulary from file"""
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded vocabulary with {len(vocab)} words")
        return vocab
    except FileNotFoundError:
        logger.warning(f"Vocabulary file {vocab_file} not found, creating default Hindi vocab")
        # Create basic Hindi vocabulary
        hindi_words = [
            '<unk>', '<pad>', '<sos>', '<eos>',
            'यह', 'एक', 'नमूना', 'वाक्य', 'है', 'हैं', 'की', 'के', 'को', 'में', 'से', 'पर',
            'और', 'या', 'तो', 'जो', 'कि', 'अगर', 'तब', 'फिर', 'अब', 'यहाँ', 'वहाँ',
            'कैसे', 'क्यों', 'कब', 'कहाँ', 'क्या', 'कौन', 'कितना', 'कितनी', 'कुछ', 'सब',
            'इदं', 'नमूना', 'वाक्यम्', 'अस्ति'  # Sanskrit words
        ]
        return hindi_words

def load_g2p_mapping(g2p_file: str) -> Dict[str, str]:
    """Load grapheme-to-phoneme mapping for Hindi/Sanskrit"""
    try:
        with open(g2p_file, 'r', encoding='utf-8') as f:
            g2p_map = json.load(f)
        logger.info(f"Loaded G2P mapping with {len(g2p_map)} entries")
        return g2p_map
    except FileNotFoundError:
        logger.warning(f"G2P mapping file {g2p_file} not found, creating default mapping")
        # Create basic Hindi/Sanskrit G2P mapping
        g2p_map = {
            'यह': 'j e h',
            'एक': 'e k',
            'नमूना': 'n e m u u n aa',
            'वाक्य': 'v aa k j',
            'है': 'h ai',
            'हैं': 'h ai n',
            'की': 'k ii',
            'के': 'k e',
            'को': 'k o',
            'में': 'm e n',
            'से': 's e',
            'पर': 'p e r',
            'और': 'au r',
            'या': 'j aa',
            'इदं': 'i d e m',
            'वाक्यम्': 'v aa k j e m',
            'अस्ति': 'e s t i'
        }
        return g2p_map

def create_hindi_senone_mapping(num_senones: int = 3080) -> Dict[int, str]:
    """
    Create Hindi senone to phoneme mapping
    This is a simplified mapping - in practice, this should be derived from
    your acoustic model training data and senone clustering
    """
    # Hindi phonemes (approximation)
    hindi_phonemes = [
        # Vowels
        'a', 'aa', 'i', 'ii', 'u', 'uu', 'e', 'ai', 'o', 'au',
        # Consonants
        'k', 'kh', 'g', 'gh', 'ng',  # Velar
        'ch', 'chh', 'j', 'jh', 'ny',  # Palatal
        't', 'th', 'd', 'dh', 'n',  # Retroflex
        'p', 'ph', 'b', 'bh', 'm',  # Bilabial
        'y', 'r', 'l', 'v', 'sh', 's', 'h',  # Semi-vowels and fricatives
        # Additional sounds
        'sil', 'sp', 'noise'  # Silence, short pause, noise
    ]
    
    # Map senones to phonemes (simplified - each phoneme gets multiple senones)
    senone_map = {}
    phonemes_per_senone = len(hindi_phonemes)
    senones_per_phoneme = num_senones // phonemes_per_senone
    
    for i in range(num_senones):
        phoneme_idx = i // senones_per_phoneme
        if phoneme_idx < len(hindi_phonemes):
            senone_map[i] = hindi_phonemes[phoneme_idx]
        else:
            senone_map[i] = 'sil'  # Default to silence
    
    return senone_map

# ============================================================================
# WFST DECODER
# ============================================================================

class WFSTDecoder:
    """
    Weighted Finite State Transducer decoder for Hindi/Sanskrit ASR
    Integrated with your existing model architecture and config
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        
        # Load vocabulary and mappings
        self.vocab = load_vocabulary(self.config['vocab_file'])
        self.g2p_mapping = load_g2p_mapping(
            os.path.join(self.config['data_dir'], 'g2p_mapping.json')
        )
        
        # Create senone mappings for Hindi (3080 senones)
        self.senone_to_phoneme_map = create_hindi_senone_mapping(
            self.config['num_senones']
        )
        
        # Build phoneme to word lexicon
        self.phoneme_lexicon = self._build_phoneme_lexicon()
        
        # Build reverse vocab mapping
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        
        logger.info(f"Decoder initialized with {len(self.vocab)} words, "
                   f"{len(self.senone_to_phoneme_map)} senones, "
                   f"{len(self.phoneme_lexicon)} lexicon entries")
    
    def _build_phoneme_lexicon(self) -> Dict[str, List[List[str]]]:
        """Build phoneme to word lexicon from G2P mapping"""
        lexicon = defaultdict(list)
        
        for word, phoneme_seq in self.g2p_mapping.items():
            phonemes = phoneme_seq.split()
            lexicon[word].append(phonemes)
        
        # Add words not in G2P mapping with simple character mapping
        for word in self.vocab:
            if word not in lexicon and word not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                # Simple fallback: map each character to a phoneme
                phonemes = []
                for char in word:
                    if char in 'अआइईउऊएऐओऔ':  # Vowels
                        phonemes.append('a')  # Simplified
                    else:  # Consonants
                        phonemes.append('k')  # Simplified
                if phonemes:
                    lexicon[word].append(phonemes)
        
        return dict(lexicon)
    
    def decode(self, senone_posteriors: torch.Tensor, beam_width: int = 10) -> str:
        """
        Decode senone posteriors to text
        
        Args:
            senone_posteriors: Log probabilities [seq_len, num_senones]
            beam_width: Beam width for decoding
            
        Returns:
            decoded_text: Decoded text string
        """
        # Convert to numpy for processing
        if isinstance(senone_posteriors, torch.Tensor):
            posteriors = senone_posteriors.cpu().numpy()
        else:
            posteriors = senone_posteriors
        
        # Get best senone sequence
        best_senones = self._get_best_senone_sequence(posteriors, beam_width)
        
        # Convert senones to phonemes
        phonemes = self._senones_to_phonemes(best_senones)
        
        # Convert phonemes to words
        words = self._phonemes_to_words(phonemes)
        
        return ' '.join(words) if words else '<unk>'
    
    def _get_best_senone_sequence(self, posteriors: np.ndarray, beam_width: int) -> List[int]:
        """Get best senone sequence using simple argmax (can be enhanced with beam search)"""
        # Simple greedy decoding - can be enhanced with proper beam search
        best_senones = np.argmax(posteriors, axis=-1).tolist()
        return best_senones
    
    def _senones_to_phonemes(self, senones: List[int]) -> List[str]:
        """Convert senone sequence to phonemes with CTC-like collapsing"""
        if not senones:
            return []
        
        phonemes = []
        prev_phoneme = None
        
        for senone_id in senones:
            if senone_id in self.senone_to_phoneme_map:
                phoneme = self.senone_to_phoneme_map[senone_id]
                
                # Collapse repeated phonemes (CTC-like behavior)
                if phoneme != prev_phoneme and phoneme not in ['sil', 'sp']:
                    phonemes.append(phoneme)
                    prev_phoneme = phoneme
        
        return phonemes
    
    def _phonemes_to_words(self, phonemes: List[str]) -> List[str]:
        """Convert phoneme sequence to words using lexicon matching"""
        if not phonemes:
            return []
        
        words = []
        i = 0
        
        while i < len(phonemes):
            best_match = None
            best_length = 0
            best_score = 0.0
            
            # Try to match words from lexicon
            for word, phoneme_sequences in self.phoneme_lexicon.items():
                for phoneme_seq in phoneme_sequences:
                    if i + len(phoneme_seq) <= len(phonemes):
                        candidate = phonemes[i:i+len(phoneme_seq)]
                        score = self._phoneme_similarity(candidate, phoneme_seq)
                        
                        if score > 0.6 and len(phoneme_seq) > best_length:
                            best_match = word
                            best_length = len(phoneme_seq)
                            best_score = score
            
            if best_match:
                words.append(best_match)
                i += best_length
            else:
                # Skip unmatched phoneme
                i += 1
        
        return words if words else ['<unk>']
    
    def _phoneme_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two phoneme sequences"""
        if len(seq1) != len(seq2):
            return 0.0
        
        if not seq1 or not seq2:
            return 0.0
        
        matches = sum(1 for p1, p2 in zip(seq1, seq2) if p1 == p2)
        return matches / len(seq1)

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def inference(model, audio_file: str, feature_extractor: FeatureExtractor,
              decoder: WFSTDecoder, device: str = 'cuda') -> str:
    """
    Perform inference on a single audio file
    
    Args:
        model: Trained ASR model (GRLModel or DSNModel)
        audio_file: Path to audio file
        feature_extractor: Feature extraction object
        decoder: WFST decoder
        device: Device for computation
        
    Returns:
        decoded_text: Recognized text
    """
    model.eval()
    
    try:
        # Extract features
        features = feature_extractor.extract_features(audio_file)  # [T, 1320]
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # [1, T, 1320]
        
        with torch.no_grad():
            # Forward pass
            if isinstance(model, GRLModel):
                senone_logits, _ = model(features_tensor, alpha=0.0)
            else:  # DSNModel
                senone_logits, _, _, _, _ = model(features_tensor, domain_id=1)  # Target domain
            
            # Convert to log probabilities
            senone_log_probs = F.log_softmax(senone_logits, dim=-1)  # [1, T, 3080]
            senone_log_probs = senone_log_probs.squeeze(0)  # [T, 3080]
            
            # Decode
            decoded_text = decoder.decode(senone_log_probs)
        
        logger.info(f"Decoded '{audio_file}': {decoded_text}")
        return decoded_text
        
    except Exception as e:
        logger.error(f"Error during inference on {audio_file}: {str(e)}")
        return "<error>"

def batch_inference(model, audio_files: List[str], feature_extractor: FeatureExtractor,
                   decoder: WFSTDecoder, device: str = 'cuda') -> List[str]:
    """
    Perform batch inference on multiple audio files
    
    Args:
        model: Trained ASR model
        audio_files: List of audio file paths
        feature_extractor: Feature extraction object
        decoder: WFST decoder
        device: Device for computation
        
    Returns:
        decoded_texts: List of recognized texts
    """
    results = []
    
    for i, audio_file in enumerate(audio_files):
        result = inference(model, audio_file, feature_extractor, decoder, device)
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(audio_files)} files")
    
    return results

def evaluate_model(model, audio_files: List[str], transcripts: List[str],
                  feature_extractor: FeatureExtractor, decoder: WFSTDecoder, 
                  device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate model performance using Word Error Rate (WER)
    
    Args:
        model: Trained ASR model
        audio_files: List of audio file paths
        transcripts: List of ground truth transcripts
        feature_extractor: Feature extraction object
        decoder: WFST decoder
        device: Device for computation
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    if len(audio_files) != len(transcripts):
        raise ValueError("Number of audio files must match number of transcripts")
    
    logger.info(f"Evaluating on {len(audio_files)} files...")
    predictions = batch_inference(model, audio_files, feature_extractor, decoder, device)
    
    total_words = 0
    total_errors = 0
    exact_matches = 0
    
    for pred, true in zip(predictions, transcripts):
        pred_words = pred.strip().split()
        true_words = true.strip().split()
        
        # Calculate word error rate
        errors = editdistance.eval(pred_words, true_words)
        total_errors += errors
        total_words += len(true_words)
        
        # Count exact matches
        if pred.strip() == true.strip():
            exact_matches += 1
    
    wer = total_errors / total_words if total_words > 0 else 1.0
    accuracy = 1.0 - wer
    exact_match_rate = exact_matches / len(transcripts)
    
    metrics = {
        'word_error_rate': wer,
        'accuracy': accuracy,
        'exact_match_rate': exact_match_rate,
        'total_words': total_words,
        'total_errors': total_errors,
        'total_files': len(audio_files)
    }
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  WER: {wer:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Exact Match Rate: {exact_match_rate:.4f}")
    
    return metrics

# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def create_inference_pipeline(model_path: str, config: Dict = None):
    """
    Create complete inference pipeline
    
    Args:
        model_path: Path to trained model checkpoint
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, feature_extractor, decoder, device)
    """
    config = config or get_config()
    device = config['device']
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        n_mels=config['n_mels'],
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        context_frames=config['context_frames']
    )
    
    # Initialize decoder
    decoder = WFSTDecoder(config)
    
    # Load model
    if config['model_type'] == 'GRL':
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
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Inference pipeline created with {config['model_type']} model")
    
    return model, feature_extractor, decoder, device

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    config = get_config()
    
    # Create inference pipeline
    model_path = "./outputs/best_model.pth"  # Path to your trained model
    
    try:
        model, feature_extractor, decoder, device = create_inference_pipeline(model_path, config)
        
        # Single file inference
        audio_file = "./data/test_audio.wav"
        if os.path.exists(audio_file):
            result = inference(model, audio_file, feature_extractor, decoder, device)
            print(f"Recognition result: {result}")
        
        # Batch inference example
        audio_files = ["./data/test1.wav", "./data/test2.wav"]  # Add your test files
        transcripts = ["यह एक परीक्षा है", "नमूना वाक्य"]  # Ground truth
        
        if all(os.path.exists(f) for f in audio_files):
            results = batch_inference(model, audio_files, feature_extractor, decoder, device)
            print(f"Batch results: {results}")
            
            # Evaluate
            metrics = evaluate_model(model, audio_files, transcripts, 
                                   feature_extractor, decoder, device)
            print(f"Evaluation metrics: {metrics}")
        
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train a model first.")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Pipeline error: {e}")
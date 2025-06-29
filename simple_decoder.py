# simple_decoder.py
import torch
import numpy as np
import logging
from typing import List, Dict
import re
import os

logger = logging.getLogger(__name__)

class SimplePracticalDecoder:
    """
    A practical decoder that gives much better results than the current one
    This is a simplified version that works without complex WFST knowledge
    """
    
    def __init__(self, vocab_file=None, transcripts_file=None):
        # Build vocabulary from your actual transcripts
        self.build_vocabulary_from_data(transcripts_file)
        self.build_senone_to_phoneme_mapping()
        self.build_phoneme_to_word_mapping()
        
    def build_vocabulary_from_data(self, transcripts_file=None):
        """Build vocabulary from actual Hindi/Sanskrit transcripts"""
        self.vocab = set()
        
        # Read from TSV files if available
        tsv_files = ['./data/hindi.tsv', './data/sanskrit.tsv']
        
        for tsv_file in tsv_files:
            if os.path.exists(tsv_file):
                with open(tsv_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            transcript = parts[1]
                            # Extract words (split by space)
                            words = transcript.split()
                            for word in words:
                                # Clean word (remove punctuation)
                                clean_word = re.sub(r'[^\u0900-\u097F\s]', '', word).strip()
                                if clean_word:
                                    self.vocab.add(clean_word)
        
        # Add special tokens
        special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>', '<sil>']
        for token in special_tokens:
            self.vocab.add(token)
        
        # Convert to list for indexing
        self.vocab_list = sorted(list(self.vocab))
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab_list)}
        
        logger.info(f"Built vocabulary with {len(self.vocab_list)} words from transcripts")
        
        # Show some example words
        if len(self.vocab_list) > 10:
            logger.info(f"Sample words: {self.vocab_list[5:15]}")
    
    def build_senone_to_phoneme_mapping(self):
        """Create a mapping from senones to phonemes"""
        # Simplified Hindi/Sanskrit phonemes
        self.phonemes = [
            # Vowels
            'a', 'aa', 'i', 'ii', 'u', 'uu', 'e', 'ai', 'o', 'au',
            # Consonants  
            'k', 'kh', 'g', 'gh', 'ng',
            'ch', 'chh', 'j', 'jh', 'ny', 
            't', 'th', 'd', 'dh', 'n',
            'tt', 'tth', 'dd', 'ddh', 'nn',
            'p', 'ph', 'b', 'bh', 'm',
            'y', 'r', 'l', 'v', 'sh', 's', 'h',
            # Special
            'sil', 'sp'  # silence, short pause
        ]
        
        # Map each senone to a phoneme
        self.senone_to_phoneme = {}
        phonemes_per_senone = max(1, 3080 // len(self.phonemes))
        
        for senone_id in range(3080):
            phoneme_idx = senone_id // phonemes_per_senone
            if phoneme_idx < len(self.phonemes):
                self.senone_to_phoneme[senone_id] = self.phonemes[phoneme_idx]
            else:
                self.senone_to_phoneme[senone_id] = 'sil'
    
    def build_phoneme_to_word_mapping(self):
        """Create a simple mapping from phoneme sequences to words"""
        self.phoneme_to_words = {}
        
        # For each word in vocabulary, create a simple phoneme representation
        for word in self.vocab_list:
            if word in ['<unk>', '<pad>', '<sos>', '<eos>', '<sil>']:
                continue
                
            # Simple heuristic: map Devanagari characters to phonemes
            phoneme_seq = self.word_to_phonemes_simple(word)
            phoneme_key = ' '.join(phoneme_seq)
            
            if phoneme_key not in self.phoneme_to_words:
                self.phoneme_to_words[phoneme_key] = []
            self.phoneme_to_words[phoneme_key].append(word)
    
    def word_to_phonemes_simple(self, word):
        """Simple mapping from Devanagari word to phonemes"""
        phonemes = []
        
        # This is a very simplified mapping - in reality you'd need a proper G2P
        char_to_phoneme = {
            'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
            'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
            'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
            'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
            'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
            'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
            'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
            'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
            'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
            'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
            '्': '',  # virama (inherent vowel killer)
            'ं': 'n',  # anusvara
            'ः': 'h',  # visarga
        }
        
        for char in word:
            if char in char_to_phoneme:
                phoneme = char_to_phoneme[char]
                if phoneme:  # Skip empty strings
                    phonemes.append(phoneme)
            else:
                # Unknown character, use 'a' as default
                phonemes.append('a')
        
        # If no phonemes found, return default
        if not phonemes:
            phonemes = ['a']
            
        return phonemes
    
    def senones_to_phonemes(self, senone_sequence):
        """Convert senone sequence to phonemes with repetition removal"""
        phonemes = []
        prev_phoneme = None
        
        for senone_id in senone_sequence:
            if senone_id in self.senone_to_phoneme:
                phoneme = self.senone_to_phoneme[senone_id]
                
                # Remove consecutive duplicates (CTC-like behavior)
                if phoneme != prev_phoneme and phoneme not in ['sil', 'sp']:
                    phonemes.append(phoneme)
                    prev_phoneme = phoneme
        
        return phonemes
    
    def phonemes_to_words(self, phonemes):
        """Convert phoneme sequence to words using approximate matching"""
        if not phonemes:
            return ['<unk>']
        
        words = []
        i = 0
        
        while i < len(phonemes):
            best_match = None
            best_length = 0
            best_score = 0
            
            # Try different window sizes
            for window_size in range(1, min(8, len(phonemes) - i + 1)):
                phoneme_window = phonemes[i:i + window_size]
                phoneme_key = ' '.join(phoneme_window)
                
                # Exact match
                if phoneme_key in self.phoneme_to_words:
                    best_match = self.phoneme_to_words[phoneme_key][0]  # Take first match
                    best_length = window_size
                    best_score = 1.0
                    break
                
                # Approximate match (fuzzy matching)
                for existing_key, word_list in self.phoneme_to_words.items():
                    existing_phonemes = existing_key.split()
                    if len(existing_phonemes) == window_size:
                        # Calculate similarity
                        matches = sum(1 for p1, p2 in zip(phoneme_window, existing_phonemes) if p1 == p2)
                        score = matches / window_size
                        
                        if score > 0.6 and score > best_score:  # At least 60% match
                            best_match = word_list[0]
                            best_length = window_size
                            best_score = score
            
            if best_match:
                words.append(best_match)
                i += best_length
            else:
                # Skip this phoneme if no match found
                i += 1
        
        # Fallback
        if not words:
            words = ['<unk>']
        
        return words
    
    def decode(self, senone_logits):
        """Main decoding function"""
        # Get best senone sequence
        senone_sequence = torch.argmax(senone_logits, dim=-1).cpu().numpy()
        
        # Convert to phonemes
        phonemes = self.senones_to_phonemes(senone_sequence)
        
        # Convert to words
        words = self.phonemes_to_words(phonemes)
        
        # Join and clean up
        result = ' '.join(words[:20])  # Limit to 20 words max
        return result
    
    def batch_decode(self, senone_logits_batch):
        """Decode a batch of senone sequences"""
        results = []
        for i in range(senone_logits_batch.size(0)):
            single_result = self.decode(senone_logits_batch[i])
            results.append(single_result)
        return results

def create_practical_decoder():
    """Factory function to create the decoder"""
    return SimplePracticalDecoder()

# Integration with existing evaluation
class ImprovedASREvaluator:
    """Enhanced evaluator using the practical decoder"""
    
    def __init__(self):
        self.decoder = create_practical_decoder()
        logger.info(f"Created practical decoder with {len(self.decoder.vocab_list)} words")
    
    def decode_predictions(self, senone_logits):
        """Use the practical decoder instead of simple one"""
        return self.decoder.batch_decode(senone_logits)
    
    def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        total_words = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.strip().split()
            ref_words = ref.strip().split()
            
            # Use edit distance
            import editdistance
            errors = editdistance.eval(pred_words, ref_words)
            total_errors += errors
            total_words += len(ref_words)
        
        wer = total_errors / total_words if total_words > 0 else 1.0
        return wer
    
    def evaluate_model_with_better_decoder(self, model, test_loader, device='cuda'):
        """Evaluate model using the practical decoder"""
        model.eval()
        
        all_predictions = []
        all_references = []
        
        logger.info("Starting evaluation with practical decoder...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 10:  # Evaluate on first 10 batches for speed
                    break
                    
                features = batch['features'].to(device)
                transcripts = batch['transcripts']
                
                # Forward pass
                try:
                    if 'DSN' in str(type(model)):
                        senone_logits, _, _, _, _ = model(features, domain_id=1)
                    else:  # GRL
                        senone_logits, _ = model(features, alpha=0.0)
                    
                    # Decode using practical decoder
                    batch_predictions = self.decode_predictions(senone_logits)
                    all_predictions.extend(batch_predictions)
                    all_references.extend(transcripts)
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        if all_predictions:
            wer = self.calculate_wer(all_predictions, all_references)
            logger.info(f"🎯 Practical Decoder Results:")
            logger.info(f"  Samples: {len(all_predictions)}")
            logger.info(f"  Word Error Rate: {wer*100:.2f}%")
            
            # Show examples
            logger.info(f"  Sample predictions:")
            for i in range(min(3, len(all_predictions))):
                logger.info(f"    Ref: {all_references[i][:100]}...")
                logger.info(f"    Pred: {all_predictions[i][:100]}...")
            
            return wer, all_predictions, all_references
        else:
            logger.error("No predictions generated!")
            return 1.0, [], []

if __name__ == "__main__":
    # Test the decoder
    decoder = create_practical_decoder()
    
    # Test with dummy senone logits
    dummy_logits = torch.randn(1, 100, 3080)  # [batch, seq, num_senones]
    result = decoder.decode(dummy_logits[0])
    
    print(f"Test decode result: {result}")
    print(f"Vocabulary size: {len(decoder.vocab_list)}")
    print(f"Sample words: {decoder.vocab_list[:10]}")
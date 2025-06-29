# improved_test_decoder.py
import torch
from simple_decoder import ImprovedASREvaluator, SimplePracticalDecoder
from fixed_config import get_config
from models import DSNModel
import editdistance

class BetterPracticalDecoder(SimplePracticalDecoder):
    """Improved version with better word limiting and filtering"""
    
    def decode(self, senone_logits):
        """Improved decoding with better length control"""
        # Get best senone sequence
        senone_sequence = torch.argmax(senone_logits, dim=-1).cpu().numpy()
        
        # Limit sequence length to prevent too many words
        max_frames = min(len(senone_sequence), 200)  # Limit to 200 frames max
        senone_sequence = senone_sequence[:max_frames]
        
        # Convert to phonemes with better filtering
        phonemes = self.senones_to_phonemes_filtered(senone_sequence)
        
        # Convert to words with strict limiting
        words = self.phonemes_to_words_limited(phonemes)
        
        # Join with strict word limit
        result = ' '.join(words[:5])  # STRICT: Max 5 words only
        return result if result.strip() else '<unk>'
    
    def senones_to_phonemes_filtered(self, senone_sequence):
        """Better phoneme extraction with filtering"""
        phonemes = []
        prev_phoneme = None
        consecutive_count = 0
        
        for senone_id in senone_sequence:
            if senone_id in self.senone_to_phoneme:
                phoneme = self.senone_to_phoneme[senone_id]
                
                # Skip silence and short pause
                if phoneme in ['sil', 'sp']:
                    consecutive_count = 0
                    continue
                
                # Remove excessive repetition
                if phoneme == prev_phoneme:
                    consecutive_count += 1
                    if consecutive_count > 3:  # Skip if repeated more than 3 times
                        continue
                else:
                    consecutive_count = 0
                
                phonemes.append(phoneme)
                prev_phoneme = phoneme
                
                # Stop if we have enough phonemes for ~3-5 words
                if len(phonemes) > 15:
                    break
        
        return phonemes[:15]  # Strict limit
    
    def phonemes_to_words_limited(self, phonemes):
        """Convert phonemes to words with strict limits"""
        if not phonemes:
            return ['<unk>']
        
        words = []
        i = 0
        
        while i < len(phonemes) and len(words) < 5:  # Max 5 words
            best_match = None
            best_length = 0
            
            # Try different window sizes (smaller windows first)
            for window_size in range(1, min(4, len(phonemes) - i + 1)):  # Max 4 phonemes per word
                phoneme_window = phonemes[i:i + window_size]
                phoneme_key = ' '.join(phoneme_window)
                
                # Exact match
                if phoneme_key in self.phoneme_to_words:
                    best_match = self.phoneme_to_words[phoneme_key][0]
                    best_length = window_size
                    break
                
                # Partial match for longer sequences
                if window_size >= 2:
                    for existing_key, word_list in self.phoneme_to_words.items():
                        existing_phonemes = existing_key.split()
                        if len(existing_phonemes) == window_size:
                            matches = sum(1 for p1, p2 in zip(phoneme_window, existing_phonemes) if p1 == p2)
                            if matches >= window_size - 1:  # Allow 1 mismatch
                                best_match = word_list[0]
                                best_length = window_size
                                break
                
                if best_match:
                    break
            
            if best_match and best_match not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                words.append(best_match)
                i += best_length
            else:
                i += 1  # Skip this phoneme
        
        # Return limited words
        return words[:3] if words else ['<unk>']  # Max 3 words

class ImprovedTestEvaluator:
    """Test evaluator with better metrics"""
    
    def __init__(self):
        self.decoder = BetterPracticalDecoder()
        print(f"Created improved decoder with {len(self.decoder.vocab_list)} words")
    
    def calculate_metrics(self, predictions, references):
        """Calculate multiple metrics"""
        if len(predictions) != len(references):
            return {}
        
        total_ref_words = 0
        total_pred_words = 0
        total_errors = 0
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.strip().split()
            ref_words = ref.strip().split()
            
            # Count words
            total_ref_words += len(ref_words)
            total_pred_words += len(pred_words)
            
            # Calculate edit distance
            errors = editdistance.eval(pred_words, ref_words)
            total_errors += errors
            
            # Check exact match
            if pred.strip() == ref.strip():
                exact_matches += 1
        
        # Calculate metrics
        wer = total_errors / total_ref_words if total_ref_words > 0 else 1.0
        avg_pred_length = total_pred_words / len(predictions) if predictions else 0
        avg_ref_length = total_ref_words / len(references) if references else 0
        exact_match_rate = exact_matches / len(predictions) if predictions else 0
        
        return {
            'wer': wer,
            'avg_pred_length': avg_pred_length,
            'avg_ref_length': avg_ref_length,
            'exact_match_rate': exact_match_rate,
            'total_samples': len(predictions)
        }
    
    def test_model(self, model, test_loader, device='cuda', max_batches=5):
        """Test model with improved decoder"""
        model.eval()
        
        # FIXED: Ensure model is on the correct device
        model = model.to(device)
        print(f"🔍 Model moved to: {device}")
        print(f"🔍 Model parameters device: {next(model.parameters()).device}")
        
        all_predictions = []
        all_references = []
        
        print("Testing with improved decoder...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= max_batches:  # Test on limited batches
                    break
                    
                features = batch['features'].to(device)
                transcripts = batch['transcripts']
                
                print(f"🔍 Batch {batch_idx}: Features device = {features.device}")
                
                try:
                    # Forward pass
                    if 'DSN' in str(type(model)):
                        senone_logits, _, _, _, _ = model(features, domain_id=1)
                    else:  # GRL
                        senone_logits, _ = model(features, alpha=0.0)
                    
                    print(f"🔍 Senone logits device: {senone_logits.device}")
                    
                    # Decode each sample in batch
                    for i in range(senone_logits.size(0)):
                        pred = self.decoder.decode(senone_logits[i])
                        all_predictions.append(pred)
                        all_references.append(transcripts[i])
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    # Try CPU fallback
                    try:
                        print(f"🔧 Trying CPU fallback...")
                        model_cpu = model.cpu()
                        features_cpu = features.cpu()
                        
                        if 'DSN' in str(type(model)):
                            senone_logits, _, _, _, _ = model_cpu(features_cpu, domain_id=1)
                        else:  # GRL
                            senone_logits, _ = model_cpu(features_cpu, alpha=0.0)
                        
                        for i in range(senone_logits.size(0)):
                            pred = self.decoder.decode(senone_logits[i])
                            all_predictions.append(pred)
                            all_references.append(transcripts[i])
                        
                        # Move model back to GPU for next iteration
                        model = model.to(device)
                        print(f"✅ CPU fallback successful")
                        
                    except Exception as e2:
                        print(f"❌ CPU fallback also failed: {e2}")
                        continue
        
        return all_predictions, all_references

def main():
    # Load config and model
    config = get_config()
    model = DSNModel(
        input_dim=config['input_dim'],
        private_hidden=config['private_hidden'],
        shared_hidden=config['hidden_dim'],
        num_senones=config['num_senones'],
        num_domains=config['num_domains']
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load('./outputs/dsn_epoch_20.pth', map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    try:
        from fixed_dataset import create_data_loaders
        _, _, test_loader, _ = create_data_loaders(config)
        print("✅ Test data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Test with improved decoder
    evaluator = ImprovedTestEvaluator()
    predictions, references = evaluator.test_model(model, test_loader, config['device'])
    
    if predictions:
        metrics = evaluator.calculate_metrics(predictions, references)
        
        print(f"\n🎯 IMPROVED DECODER RESULTS:")
        print(f"{'='*50}")
        print(f"Word Error Rate:        {metrics['wer']*100:.2f}%")
        print(f"Exact Match Rate:       {metrics['exact_match_rate']*100:.2f}%")
        print(f"Avg Prediction Length:  {metrics['avg_pred_length']:.1f} words")
        print(f"Avg Reference Length:   {metrics['avg_ref_length']:.1f} words")
        print(f"Total Samples:          {metrics['total_samples']}")
        
        print(f"\n📝 Sample Results:")
        for i in range(min(5, len(predictions))):
            print(f"  {i+1}. Reference: {references[i][:80]}...")
            print(f"     Predicted:  {predictions[i]}")
            print()
        
        # Performance assessment
        wer_percent = metrics['wer'] * 100
        if wer_percent < 30:
            performance = "🎉 Excellent!"
        elif wer_percent < 60:
            performance = "✅ Good progress!"
        elif wer_percent < 100:
            performance = "📈 Improving!"
        else:
            performance = "🔧 Needs more training"
        
        print(f"Assessment: {performance}")
        
        if wer_percent > 80:
            print(f"\n💡 Tips to improve:")
            print(f"   - Train for more epochs (15-25)")
            print(f"   - Use better alignments (Kaldi)")
            print(f"   - Increase vocabulary size")
    else:
        print("❌ No predictions generated!")

if __name__ == "__main__":
    main()
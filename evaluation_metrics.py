# # # evaluation_metrics.py
# # import torch
# # import torch.nn.functional as F
# # import numpy as np
# # import logging
# # from typing import List, Dict, Tuple
# # import editdistance
# # import re
# # from collections import defaultdict
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import os
# # import nltk

# # logger = logging.getLogger(__name__)

# # class ASREvaluator:
# #     """Comprehensive ASR evaluation with multiple metrics"""
    
# #     def __init__(self, vocab_file=None):
# #         self.vocab_file = vocab_file
# #         self.load_vocabulary()
        
# #     def load_vocabulary(self):
# #         """Load vocabulary for decoding"""
# #         if self.vocab_file and os.path.exists(self.vocab_file):
# #             with open(self.vocab_file, 'r', encoding='utf-8') as f:
# #                 self.vocab = [line.strip() for line in f if line.strip()]
# #         else:
# #             # Default Hindi/Sanskrit vocabulary
# #             self.vocab = [
# #                 '<unk>', '<pad>', '<sos>', '<eos>',
# #                 # Hindi words
# #                 'यह', 'एक', 'नमूना', 'वाक्य', 'है', 'हैं', 'की', 'के', 'को', 'में', 'से', 'पर',
# #                 'और', 'या', 'तो', 'जो', 'कि', 'अगर', 'तब', 'फिर', 'अब', 'यहाँ', 'वहाँ',
# #                 'कैसे', 'क्यों', 'कब', 'कहाँ', 'क्या', 'कौन', 'कितना', 'कितनी', 'कुछ', 'सब',
# #                 # Sanskrit words
# #                 'इदं', 'तत्', 'एतत्', 'वाक्यम्', 'अस्ति', 'भवति', 'गच्छति', 'आगच्छति',
# #                 'श्री', 'ॐ', 'नमः', 'स्वाहा', 'मन्त्र', 'यज्ञ', 'होम', 'पूजा'
# #             ]
        
# #         logger.info(f"Loaded vocabulary with {len(self.vocab)} words")
    
# #     def simple_decode_senones(self, senone_logits: torch.Tensor) -> List[str]:
# #         """
# #         Simple senone to text decoder for evaluation
# #         This is a basic implementation for demonstration
# #         """
# #         # Get most likely senones
# #         senone_ids = torch.argmax(senone_logits, dim=-1)  # [batch_size, seq_len]
        
# #         decoded_texts = []
# #         for batch_idx in range(senone_ids.shape[0]):
# #             sequence = senone_ids[batch_idx].cpu().numpy()
            
# #             # Simple mapping: group senones into words
# #             words = []
# #             current_word_senones = []
            
# #             for senone_id in sequence:
# #                 if senone_id < len(self.vocab):
# #                     # Map senone to vocabulary word (simplified)
# #                     vocab_idx = senone_id % len(self.vocab)
# #                     if vocab_idx > 3:  # Skip special tokens
# #                         words.append(self.vocab[vocab_idx])
                
# #                 # Add word every few senones (simplified segmentation)
# #                 if len(current_word_senones) > 10:
# #                     current_word_senones = []
            
# #             # Fallback if no words generated
# #             if not words:
# #                 words = ['<unk>']
            
# #             decoded_texts.append(' '.join(words[:10]))  # Limit to 10 words
        
# #         return decoded_texts
    
# #     def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
# #         """Calculate Word Error Rate"""
# #         if len(predictions) != len(references):
# #             raise ValueError("Predictions and references must have same length")
        
# #         total_words = 0
# #         total_errors = 0
        
# #         for pred, ref in zip(predictions, references):
# #             pred_words = pred.strip().split()
# #             ref_words = ref.strip().split()
            
# #             errors = editdistance.eval(pred_words, ref_words)
# #             total_errors += errors
# #             total_words += len(ref_words)
        
# #         wer = total_errors / total_words if total_words > 0 else 1.0
# #         return wer
    
# #     def calculate_cer(self, predictions: List[str], references: List[str]) -> float:
# #         """Calculate Character Error Rate"""
# #         if len(predictions) != len(references):
# #             raise ValueError("Predictions and references must have same length")
        
# #         total_chars = 0
# #         total_errors = 0
        
# #         for pred, ref in zip(predictions, references):
# #             # Remove spaces for character-level comparison
# #             pred_chars = list(pred.replace(' ', ''))
# #             ref_chars = list(ref.replace(' ', ''))
            
# #             errors = editdistance.eval(pred_chars, ref_chars)
# #             total_errors += errors
# #             total_chars += len(ref_chars)
        
# #         cer = total_errors / total_chars if total_chars > 0 else 1.0
# #         return cer
    
# #     def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
# #         """Calculate BLEU score (simplified version)"""
# #         try:
# #             from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
# #             total_bleu = 0
# #             smoothing = SmoothingFunction().method1
            
# #             for pred, ref in zip(predictions, references):
# #                 pred_tokens = pred.split()
# #                 ref_tokens = [ref.split()]  # BLEU expects list of reference lists
                
# #                 bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
# #                 total_bleu += bleu
            
# #             return total_bleu / len(predictions) if predictions else 0.0
            
# #         except ImportError:
# #             logger.warning("NLTK not available, BLEU score set to 0")
# #             return 0.0
    
# #     def calculate_accuracy_metrics(self, senone_logits: torch.Tensor, 
# #                                  senone_targets: torch.Tensor) -> Dict[str, float]:
# #         """Calculate senone-level accuracy metrics"""
# #         # Get predictions
# #         senone_preds = torch.argmax(senone_logits, dim=-1)
        
# #         # Create mask for valid targets (not -1)
# #         mask = senone_targets != -1
        
# #         if mask.sum() == 0:
# #             return {'senone_accuracy': 0.0, 'top5_accuracy': 0.0}
        
# #         # Senone accuracy
# #         correct = (senone_preds == senone_targets) & mask
# #         senone_accuracy = correct.sum().float() / mask.sum().float()
        
# #         # Top-5 accuracy
# #         _, top5_preds = torch.topk(senone_logits, 5, dim=-1)
# #         top5_correct = (top5_preds == senone_targets.unsqueeze(-1)).any(dim=-1) & mask
# #         top5_accuracy = top5_correct.sum().float() / mask.sum().float()
        
# #         return {
# #             'senone_accuracy': senone_accuracy.item(),
# #             'top5_accuracy': top5_accuracy.item()
# #         }
    
# #     def evaluate_model(self, model, test_loader, device='cuda') -> Dict[str, float]:
# #         """Comprehensive model evaluation"""
# #         model.eval()
        
# #         all_predictions = []
# #         all_references = []
# #         total_senone_accuracy = 0
# #         total_top5_accuracy = 0
# #         total_samples = 0
# #         domain_accuracies = defaultdict(list)
        
# #         logger.info("Starting model evaluation...")
        
# #         with torch.no_grad():
# #             for batch_idx, batch in enumerate(test_loader):
# #                 features = batch['features'].to(device)
# #                 senones = batch['senones'].to(device)
# #                 domains = batch['domains'].to(device)
# #                 transcripts = batch['transcripts']
                
# #                 # Forward pass
# #                 if hasattr(model, 'forward'):
# #                     # Check model type
# #                     if 'GRL' in str(type(model)):
# #                         senone_logits, domain_logits = model(features, alpha=0.0)
# #                     else:  # DSN
# #                         senone_logits, domain_logits, _, _, _ = model(features, domain_id=1)
# #                 else:
# #                     continue
                
# #                 # Calculate senone accuracy
# #                 accuracy_metrics = self.calculate_accuracy_metrics(senone_logits, senones)
# #                 total_senone_accuracy += accuracy_metrics['senone_accuracy']
# #                 total_top5_accuracy += accuracy_metrics['top5_accuracy']
                
# #                 # Store domain-specific accuracies
# #                 for i, domain in enumerate(domains):
# #                     domain_accuracies[domain.item()].append(accuracy_metrics['senone_accuracy'])
                
# #                 # Decode predictions
# #                 batch_predictions = self.simple_decode_senones(senone_logits)
# #                 all_predictions.extend(batch_predictions)
# #                 all_references.extend(transcripts)
                
# #                 total_samples += 1
                
# #                 if (batch_idx + 1) % 10 == 0:
# #                     logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
# #         # Calculate final metrics
# #         if total_samples == 0:
# #             logger.error("No samples processed!")
# #             return {}
        
# #         avg_senone_accuracy = total_senone_accuracy / total_samples
# #         avg_top5_accuracy = total_top5_accuracy / total_samples
        
# #         # Calculate text-level metrics
# #         wer = self.calculate_wer(all_predictions, all_references)
# #         cer = self.calculate_cer(all_predictions, all_references)
# #         bleu = self.calculate_bleu(all_predictions, all_references)
        
# #         # Domain-specific accuracies
# #         hindi_accuracy = np.mean(domain_accuracies[0]) if 0 in domain_accuracies else 0.0
# #         sanskrit_accuracy = np.mean(domain_accuracies[1]) if 1 in domain_accuracies else 0.0
        
# #         metrics = {
# #             'word_error_rate': wer,
# #             'character_error_rate': cer,
# #             'bleu_score': bleu,
# #             'senone_accuracy': avg_senone_accuracy,
# #             'top5_senone_accuracy': avg_top5_accuracy,
# #             'hindi_senone_accuracy': hindi_accuracy,
# #             'sanskrit_senone_accuracy': sanskrit_accuracy,
# #             'total_samples': len(all_predictions)
# #         }
        
# #         return metrics
    
# #     def print_evaluation_report(self, metrics: Dict[str, float], model_name: str = "Model"):
# #         """Print a comprehensive evaluation report"""
# #         print(f"\n{'='*60}")
# #         print(f"EVALUATION REPORT - {model_name}")
# #         print(f"{'='*60}")
        
# #         print(f"\n📊 Text-Level Metrics:")
# #         print(f"  Word Error Rate (WER):     {metrics['word_error_rate']*100:.2f}%")
# #         print(f"  Character Error Rate (CER): {metrics['character_error_rate']*100:.2f}%")
# #         print(f"  BLEU Score:                {metrics['bleu_score']:.4f}")
        
# #         print(f"\n🎯 Senone-Level Metrics:")
# #         print(f"  Senone Accuracy:           {metrics['senone_accuracy']*100:.2f}%")
# #         print(f"  Top-5 Senone Accuracy:     {metrics['top5_senone_accuracy']*100:.2f}%")
        
# #         print(f"\n🌍 Domain-Specific Metrics:")
# #         print(f"  Hindi Senone Accuracy:     {metrics['hindi_senone_accuracy']*100:.2f}%")
# #         print(f"  Sanskrit Senone Accuracy:  {metrics['sanskrit_senone_accuracy']*100:.2f}%")
        
# #         print(f"\n📈 Evaluation Summary:")
# #         print(f"  Total Test Samples:        {metrics['total_samples']}")
        
# #         # Performance interpretation
# #         wer = metrics['word_error_rate'] * 100
# #         if wer < 15:
# #             performance = "Excellent"
# #         elif wer < 25:
# #             performance = "Good"
# #         elif wer < 40:
# #             performance = "Fair"
# #         else:
# #             performance = "Needs Improvement"
        
# #         print(f"  Overall Performance:       {performance}")
        
# #         print(f"\n🎯 Expected vs Actual (from paper):")
# #         print(f"  Baseline (Hindi only):     24.58% WER")
# #         print(f"  Expected GRL:              17.87% WER")
# #         print(f"  Expected DSN:              17.26% WER")
# #         print(f"  Your Model:                {wer:.2f}% WER")
        
# #         improvement = 24.58 - wer
# #         if improvement > 0:
# #             print(f"  Improvement over baseline: -{improvement:.2f}% WER ✅")
# #         else:
# #             print(f"  Difference from baseline:  +{abs(improvement):.2f}% WER ⚠️")
        
# #         print(f"{'='*60}\n")
    
# #     def save_evaluation_results(self, metrics: Dict[str, float], 
# #                               predictions: List[str], references: List[str],
# #                               output_dir: str, model_name: str):
# #         """Save detailed evaluation results"""
# #         import os
# #         import json
        
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         # Save metrics
# #         metrics_file = os.path.join(output_dir, f'{model_name}_metrics.json')
# #         with open(metrics_file, 'w') as f:
# #             json.dump(metrics, f, indent=2)
        
# #         # Save predictions vs references
# #         results_file = os.path.join(output_dir, f'{model_name}_results.txt')
# #         with open(results_file, 'w', encoding='utf-8') as f:
# #             f.write(f"Evaluation Results for {model_name}\n")
# #             f.write("="*50 + "\n\n")
            
# #             for i, (pred, ref) in enumerate(zip(predictions, references)):
# #                 f.write(f"Sample {i+1}:\n")
# #                 f.write(f"  Reference: {ref}\n")
# #                 f.write(f"  Predicted: {pred}\n")
# #                 f.write(f"  WER: {self.calculate_wer([pred], [ref])*100:.2f}%\n")
# #                 f.write("-" * 30 + "\n")
        
# #         logger.info(f"Evaluation results saved to {output_dir}")

# # def quick_evaluation_demo(model, test_loader, device='cuda'):
# #     """Quick evaluation demo for testing"""
# #     evaluator = ASREvaluator()
    
# #     logger.info("Running quick evaluation demo...")
    
# #     # Evaluate on first few batches only
# #     demo_predictions = []
# #     demo_references = []
    
# #     model.eval()
# #     with torch.no_grad():
# #         for batch_idx, batch in enumerate(test_loader):
# #             if batch_idx >= 3:  # Only first 3 batches for demo
# #                 break
                
# #             features = batch['features'].to(device)
# #             transcripts = batch['transcripts']
            
# #             # Forward pass
# #             try:
# #                 if 'GRL' in str(type(model)):
# #                     senone_logits, _ = model(features, alpha=0.0)
# #                 else:  # DSN
# #                     senone_logits, _, _, _, _ = model(features, domain_id=1)
                
# #                 # Decode
# #                 batch_predictions = evaluator.simple_decode_senones(senone_logits)
# #                 demo_predictions.extend(batch_predictions)
# #                 demo_references.extend(transcripts)
                
# #             except Exception as e:
# #                 logger.error(f"Error in demo evaluation: {e}")
# #                 break
    
# #     if demo_predictions:
# #         # Calculate metrics
# #         wer = evaluator.calculate_wer(demo_predictions, demo_references)
# #         cer = evaluator.calculate_cer(demo_predictions, demo_references)
        
# #         print(f"\n🚀 QUICK DEMO RESULTS:")
# #         print(f"  Samples evaluated: {len(demo_predictions)}")
# #         print(f"  Word Error Rate: {wer*100:.2f}%")
# #         print(f"  Character Error Rate: {cer*100:.2f}%")
        
# #         # Show some examples
# #         print(f"\n📝 Sample Predictions:")
# #         for i in range(min(3, len(demo_predictions))):
# #             print(f"  {i+1}. Reference: {demo_references[i]}")
# #             print(f"     Predicted:  {demo_predictions[i]}")
# #             print()
# #     else:
# #         logger.error("No predictions generated in demo!")

# # if __name__ == "__main__":
# #     # Test the evaluator
# #     evaluator = ASREvaluator()
    
# #     # Test with dummy data
# #     dummy_predictions = ["यह एक परीक्षा है", "नमूना वाक्य"]
# #     dummy_references = ["यह एक परीक्षण है", "नमूना वाक्य"]
    
# #     wer = evaluator.calculate_wer(dummy_predictions, dummy_references)
# #     cer = evaluator.calculate_cer(dummy_predictions, dummy_references)
    
# #     print(f"Test WER: {wer*100:.2f}%")
# #     print(f"Test CER: {cer*100:.2f}%")











# # evaluation_metrics.py
# import torch
# import torch.nn.functional as F
# import numpy as np
# import logging
# from typing import List, Dict, Tuple
# import editdistance
# import re
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# logger = logging.getLogger(__name__)

# class ASREvaluator:
#     """Comprehensive ASR evaluation with multiple metrics"""
    
#     def __init__(self, vocab_file):
#         self.vocab_file = vocab_file
#         self.load_vocabulary()
        
#     def load_vocabulary(self):
#         """Load vocabulary for decoding"""
#         if self.vocab_file and os.path.exists(self.vocab_file):
#             with open(self.vocab_file, 'r', encoding='utf-8') as f:
#                 self.vocab = [line.strip() for line in f if line.strip()]
#         else:
#             # Default Hindi/Sanskrit vocabulary
#             self.vocab = [
#                 '<unk>', '<pad>', '<sos>', '<eos>',
#                 # Common Hindi words
#                 'यह', 'एक', 'नमूना', 'वाक्य', 'है', 'हैं', 'की', 'के', 'को', 'में', 'से', 'पर',
#                 'और', 'या', 'तो', 'जो', 'कि', 'अगर', 'तब', 'फिर', 'अब', 'यहाँ', 'वहाँ',
#                 'कैसे', 'क्यों', 'कब', 'कहाँ', 'क्या', 'कौन', 'कितना', 'कितनी', 'कुछ', 'सब',
#                 # Common Sanskrit words
#                 'इदं', 'तत्', 'एतत्', 'वाक्यम्', 'अस्ति', 'भवति', 'गच्छति', 'आगच्छति',
#                 'श्री', 'ॐ', 'नमः', 'स्वाहा', 'मन्त्र', 'यज्ञ', 'होम', 'पूजा'
#             ]
        
#         logger.info(f"Loaded vocabulary with {len(self.vocab)} words")
    
#     def build_senone_to_word_mapping(self, num_senones=3080):
#         """Build a simple mapping from senones to words for initial testing"""
#         # This is a simplified approach for initial testing
#         # In practice, you need proper senone-to-phoneme mappings from Kaldi
#         self.senone_to_word = {}
#         words_per_senone_group = max(1, num_senones // len(self.vocab))
        
#         for senone_id in range(num_senones):
#             vocab_idx = min(senone_id // words_per_senone_group, len(self.vocab) - 1)
#             self.senone_to_word[senone_id] = self.vocab[vocab_idx]
    
#     def simple_decode_senones(self, senone_logits: torch.Tensor) -> List[str]:
#         """
#         Improved senone to text decoder for evaluation
#         """
#         # Build mapping if not exists
#         if not hasattr(self, 'senone_to_word'):
#             self.build_senone_to_word_mapping()
        
#         # Get most likely senones
#         senone_ids = torch.argmax(senone_logits, dim=-1)  # [batch_size, seq_len]
        
#         decoded_texts = []
#         for batch_idx in range(senone_ids.shape[0]):
#             sequence = senone_ids[batch_idx].cpu().numpy()
            
#             # Decode senones to words with repetition removal
#             words = []
#             prev_word = None
#             word_count = 0
            
#             for senone_id in sequence:
#                 if senone_id < len(self.senone_to_word):
#                     word = self.senone_to_word[senone_id]
                    
#                     # Skip special tokens
#                     if word in ['<pad>', '<unk>', '<sos>', '<eos>']:
#                         continue
                    
#                     # Remove consecutive duplicates
#                     if word != prev_word:
#                         words.append(word)
#                         prev_word = word
#                         word_count += 1
                        
#                         # Limit output length
#                         if word_count >= 10:
#                             break
            
#             # Join words
#             if words:
#                 decoded_text = ' '.join(words[:5])  # Limit to 5 words for now
#             else:
#                 decoded_text = '<unk>'
            
#             decoded_texts.append(decoded_text)
        
#         return decoded_texts
    
#     def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
#         """Calculate Word Error Rate"""
#         if len(predictions) != len(references):
#             raise ValueError("Predictions and references must have same length")
        
#         total_words = 0
#         total_errors = 0
        
#         for pred, ref in zip(predictions, references):
#             pred_words = pred.strip().split()
#             ref_words = ref.strip().split()
            
#             errors = editdistance.eval(pred_words, ref_words)
#             total_errors += errors
#             total_words += len(ref_words)
        
#         wer = total_errors / total_words if total_words > 0 else 1.0
#         return wer
    
#     def calculate_cer(self, predictions: List[str], references: List[str]) -> float:
#         """Calculate Character Error Rate"""
#         if len(predictions) != len(references):
#             raise ValueError("Predictions and references must have same length")
        
#         total_chars = 0
#         total_errors = 0
        
#         for pred, ref in zip(predictions, references):
#             # Remove spaces for character-level comparison
#             pred_chars = list(pred.replace(' ', ''))
#             ref_chars = list(ref.replace(' ', ''))
            
#             errors = editdistance.eval(pred_chars, ref_chars)
#             total_errors += errors
#             total_chars += len(ref_chars)
        
#         cer = total_errors / total_chars if total_chars > 0 else 1.0
#         return cer
    
#     def calculate_accuracy_metrics(self, senone_logits: torch.Tensor, 
#                                  senone_targets: torch.Tensor) -> Dict[str, float]:
#         """Calculate senone-level accuracy metrics"""
#         # Get predictions
#         senone_preds = torch.argmax(senone_logits, dim=-1)
        
#         # Create mask for valid targets (not -1)
#         mask = senone_targets != -1
        
#         if mask.sum() == 0:
#             return {'senone_accuracy': 0.0, 'top5_accuracy': 0.0}
        
#         # Senone accuracy
#         correct = (senone_preds == senone_targets) & mask
#         senone_accuracy = correct.sum().float() / mask.sum().float()
        
#         # Top-5 accuracy
#         _, top5_preds = torch.topk(senone_logits, 5, dim=-1)
#         top5_correct = (top5_preds == senone_targets.unsqueeze(-1)).any(dim=-1) & mask
#         top5_accuracy = top5_correct.sum().float() / mask.sum().float()
        
#         return {
#             'senone_accuracy': senone_accuracy.item(),
#             'top5_accuracy': top5_accuracy.item()
#         }
    
#     def evaluate_model(self, model, test_loader, device='cuda') -> Dict[str, float]:
#         """Comprehensive model evaluation"""
#         model.eval()
        
#         all_predictions = []
#         all_references = []
#         total_senone_accuracy = 0
#         total_top5_accuracy = 0
#         total_samples = 0
#         domain_accuracies = defaultdict(list)
        
#         logger.info("Starting model evaluation...")
        
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(test_loader):
#                 features = batch['features'].to(device)
#                 senones = batch['senones'].to(device)
#                 domains = batch['domains'].to(device)
#                 transcripts = batch['transcripts']
                
#                 # Forward pass
#                 if hasattr(model, 'forward'):
#                     # Check model type
#                     if 'GRL' in str(type(model)):
#                         senone_logits, domain_logits = model(features, alpha=0.0)
#                     else:  # DSN
#                         senone_logits, domain_logits, _, _, _ = model(features, domain_id=1, alpha=0.0)
#                 else:
#                     continue
                
#                 # Calculate senone accuracy
#                 accuracy_metrics = self.calculate_accuracy_metrics(senone_logits, senones)
#                 total_senone_accuracy += accuracy_metrics['senone_accuracy']
#                 total_top5_accuracy += accuracy_metrics['top5_accuracy']
                
#                 # Store domain-specific accuracies
#                 for i, domain in enumerate(domains):
#                     domain_accuracies[domain.item()].append(accuracy_metrics['senone_accuracy'])
                
#                 # Decode predictions
#                 batch_predictions = self.simple_decode_senones(senone_logits)
#                 all_predictions.extend(batch_predictions)
#                 all_references.extend(transcripts)
                
#                 total_samples += 1
                
#                 if (batch_idx + 1) % 10 == 0:
#                     logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
#         # Calculate final metrics
#         if total_samples == 0:
#             logger.error("No samples processed!")
#             return {}
        
#         avg_senone_accuracy = total_senone_accuracy / total_samples
#         avg_top5_accuracy = total_top5_accuracy / total_samples
        
#         # Calculate text-level metrics
#         wer = self.calculate_wer(all_predictions, all_references)
#         cer = self.calculate_cer(all_predictions, all_references)
        
#         # Domain-specific accuracies
#         hindi_accuracy = np.mean(domain_accuracies[0]) if 0 in domain_accuracies else 0.0
#         sanskrit_accuracy = np.mean(domain_accuracies[1]) if 1 in domain_accuracies else 0.0
        
#         metrics = {
#             'word_error_rate': wer,
#             'character_error_rate': cer,
#             'senone_accuracy': avg_senone_accuracy,
#             'top5_senone_accuracy': avg_top5_accuracy,
#             'hindi_senone_accuracy': hindi_accuracy,
#             'sanskrit_senone_accuracy': sanskrit_accuracy,
#             'total_samples': len(all_predictions)
#         }
        
#         return metrics
    
#     def print_evaluation_report(self, metrics: Dict[str, float], model_name: str = "Model"):
#         """Print a comprehensive evaluation report"""
#         print(f"\n{'='*60}")
#         print(f"EVALUATION REPORT - {model_name}")
#         print(f"{'='*60}")
        
#         print(f"\n📊 Text-Level Metrics:")
#         print(f"  Word Error Rate (WER):     {metrics['word_error_rate']*100:.2f}%")
#         print(f"  Character Error Rate (CER): {metrics['character_error_rate']*100:.2f}%")
        
#         print(f"\n🎯 Senone-Level Metrics:")
#         print(f"  Senone Accuracy:           {metrics['senone_accuracy']*100:.2f}%")
#         print(f"  Top-5 Senone Accuracy:     {metrics['top5_senone_accuracy']*100:.2f}%")
        
#         print(f"\n🌍 Domain-Specific Metrics:")
#         print(f"  Hindi Senone Accuracy:     {metrics['hindi_senone_accuracy']*100:.2f}%")
#         print(f"  Sanskrit Senone Accuracy:  {metrics['sanskrit_senone_accuracy']*100:.2f}%")
        
#         print(f"\n📈 Evaluation Summary:")
#         print(f"  Total Test Samples:        {metrics['total_samples']}")
        
#         # Performance interpretation
#         wer = metrics['word_error_rate'] * 100
#         if wer < 15:
#             performance = "Excellent"
#         elif wer < 25:
#             performance = "Good"
#         elif wer < 40:
#             performance = "Fair"
#         else:
#             performance = "Needs Improvement"
        
#         print(f"  Overall Performance:       {performance}")
        
#         print(f"\n🎯 Expected vs Actual (from paper):")
#         print(f"  Baseline (Hindi only):     24.58% WER")
#         print(f"  Expected GRL:              17.87% WER")
#         print(f"  Expected DSN:              17.26% WER")
#         print(f"  Your Model:                {wer:.2f}% WER")
        
#         improvement = 24.58 - wer
#         if improvement > 0:
#             print(f"  Improvement over baseline: -{improvement:.2f}% WER ✅")
#         else:
#             print(f"  Difference from baseline:  +{abs(improvement):.2f}% WER ⚠️")
        
#         print(f"{'='*60}\n")

# def quick_evaluation_demo(model, test_loader, device='cuda'):
#     """Quick evaluation demo for testing"""
#     evaluator = ASREvaluator("/data/vocab.txt")
    
#     logger.info("Running quick evaluation demo...")
    
#     # Evaluate on first few batches only
#     demo_predictions = []
#     demo_references = []
    
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             if batch_idx >= 3:  # Only first 3 batches for demo
#                 break
                
#             features = batch['features'].to(device)
#             transcripts = batch['transcripts']
            
#             # Forward pass
#             try:
#                 if 'GRL' in str(type(model)):
#                     senone_logits, _ = model(features, alpha=0.0)
#                 else:  # DSN
#                     senone_logits, _, _, _, _ = model(features, domain_id=1, alpha=0.0)
                
#                 # Decode
#                 batch_predictions = evaluator.simple_decode_senones(senone_logits)
#                 demo_predictions.extend(batch_predictions)
#                 demo_references.extend(transcripts)
                
#             except Exception as e:
#                 logger.error(f"Error in demo evaluation: {e}")
#                 break
    
#     if demo_predictions:
#         # Calculate metrics
#         wer = evaluator.calculate_wer(demo_predictions, demo_references)
#         cer = evaluator.calculate_cer(demo_predictions, demo_references)
        
#         print(f"\n🚀 QUICK DEMO RESULTS:")
#         print(f"  Samples evaluated: {len(demo_predictions)}")
#         print(f"  Word Error Rate: {wer*100:.2f}%")
#         print(f"  Character Error Rate: {cer*100:.2f}%")
        
#         # Show some examples
#         print(f"\n📝 Sample Predictions:")
#         for i in range(min(3, len(demo_predictions))):
#             print(f"  {i+1}. Reference: {demo_references[i][:80]}...")
#             print(f"     Predicted:  {demo_predictions[i]}")
#             print()
#     else:
#         logger.error("No predictions generated in demo!")






# evaluation_metrics.py
# evaluation_metrics.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple
import editdistance
from collections import defaultdict
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ASREvaluator:
    """
    Comprehensive ASR evaluation with multiple metrics.
    Includes qualitative examples in the final report.
    """
    
    def __init__(self, vocab_file=None):
        self.vocab = []
        if vocab_file and os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    self.vocab = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded vocabulary with {len(self.vocab)} words from {vocab_file}")
            except Exception as e:
                logger.error(f"Could not load vocab from {vocab_file}: {e}")
        
        if not self.vocab:
            logger.warning("Vocabulary file not found or empty. Using a small default vocabulary.")
            self.vocab = [
                '<unk>', '<pad>', '<sos>', '<eos>',
                'यह', 'एक', 'नमूना', 'वाक्य', 'है', 'हैं', 'की', 'के', 'को', 'में',
                'इदं', 'तत्', 'एतत्', 'वाक्यम्', 'अस्ति', 'भवति', 'गच्छति', 'आगच्छति'
            ]
        
        self.build_senone_to_word_mapping()

    def build_senone_to_word_mapping(self, num_senones=3080):
        """Builds a simple, deterministic mapping from senone IDs to vocabulary words."""
        if not self.vocab:
            self.senone_to_word = {}
            return
            
        self.senone_to_word = {}
        valid_vocab_size = max(1, len(self.vocab) - 4)
        words_per_group = max(1, num_senones // valid_vocab_size)
        
        for i in range(num_senones):
            vocab_idx = (i // words_per_group) % valid_vocab_size
            self.senone_to_word[i] = self.vocab[vocab_idx + 4]

    def simple_decode_senones(self, senone_ids: torch.Tensor) -> List[str]:
        """Decodes a batch of senone ID sequences into text."""
        decoded_texts = []
        for seq in senone_ids:
            words = []
            last_word = None
            for senone_id in seq.cpu().numpy():
                word = self.senone_to_word.get(senone_id, '<unk>')
                if word != last_word and word not in ['<pad>', '<sos>', '<eos>']:
                    words.append(word)
                    last_word = word
            
            decoded_texts.append(" ".join(words) if words else "<unk>")
        return decoded_texts

    def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """Calculates Word Error Rate (WER)."""
        if not predictions or not references: return 1.0
        
        total_errors, total_words = 0, 0
        for pred, ref in zip(predictions, references):
            pred_words, ref_words = pred.strip().split(), ref.strip().split()
            total_errors += editdistance.eval(pred_words, ref_words)
            total_words += len(ref_words)
            
        return total_errors / total_words if total_words > 0 else 1.0

    def calculate_cer(self, predictions: List[str], references: List[str]) -> float:
        """Calculates Character Error Rate (CER)."""
        if not predictions or not references: return 1.0
        
        total_errors, total_chars = 0, 0
        for pred, ref in zip(predictions, references):
            pred_chars, ref_chars = list(pred.replace(" ", "")), list(ref.replace(" ", ""))
            total_errors += editdistance.eval(pred_chars, ref_chars)
            total_chars += len(ref_chars)

        return total_errors / total_chars if total_chars > 0 else 1.0

    def calculate_accuracy_metrics(self, senone_logits: torch.Tensor, 
                                 senone_targets: torch.Tensor) -> Dict[str, float]:
        """Calculates senone-level accuracy metrics."""
        senone_targets_flat = senone_targets.view(-1)
        
        if senone_logits.shape[0] != senone_targets_flat.shape[0]:
            min_len = min(senone_logits.shape[0], senone_targets_flat.shape[0])
            senone_logits, senone_targets_flat = senone_logits[:min_len], senone_targets_flat[:min_len]

        senone_preds_flat = torch.argmax(senone_logits, dim=-1)
        mask = (senone_targets_flat != -1)
        
        if mask.sum() == 0: return {'senone_accuracy': 0.0, 'top5_accuracy': 0.0}

        correct = (senone_preds_flat == senone_targets_flat) & mask
        senone_accuracy = correct.sum().float() / mask.sum().float()

        _, top5_preds = torch.topk(senone_logits, 5, dim=-1)
        top5_correct = (top5_preds == senone_targets_flat.unsqueeze(-1)).any(dim=-1) & mask
        top5_accuracy = top5_correct.sum().float() / mask.sum().float()

        return {'senone_accuracy': senone_accuracy.item(), 'top5_accuracy': top5_accuracy.item()}

    def evaluate_model(self, model, test_loader, device='cuda') -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
        """
        Comprehensive model evaluation.
        NOW RETURNS: (metrics_dict, examples_list)
        """
        model.eval()
        all_predictions_text, all_references_text = [], []
        all_senone_logits, all_senone_targets = [], []

        logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                features = batch['features'].to(device)
                senones = batch['senones'].to(device)
                transcripts = batch['transcripts']
                
                if "GRL" in model.__class__.__name__:
                    senone_logits, _ = model(features, alpha=0.0)
                elif "DSN" in model.__class__.__name__:
                    senone_logits, _, _, _, _ = model(features, domain_id=1, alpha=0.0)
                else: continue
                
                all_senone_logits.append(senone_logits.cpu())
                all_senone_targets.append(senones.cpu())
                
                pred_ids_2d = torch.argmax(senone_logits, dim=-1).view(features.shape[0], -1)
                batch_predictions = self.simple_decode_senones(pred_ids_2d)
                all_predictions_text.extend(batch_predictions)
                all_references_text.extend(transcripts)

        if not all_senone_logits:
            logger.error("Evaluation failed: No data processed.")
            return {}, []

            
        #yaha evaluation me dikkat aarhi
        # final_logits = torch.cat(all_senone_logits, dim=0)
        # final_targets = torch.cat([t.view(-1) for t in all_senone_targets], dim=0)
        flattened_logits = []
        flattened_targets = []

        for logits, targets in zip(all_senone_logits, all_senone_targets):
            B, T, C = logits.shape
            logits = logits.view(-1, C)         # (B*T, C)
            targets = targets.view(-1)          # (B*T,)

            mask = targets != -1                # Ignore padding (common in ASR)
            valid_logits = logits[mask]
            valid_targets = targets[mask]

            flattened_logits.append(valid_logits)
            flattened_targets.append(valid_targets)

        final_logits = torch.cat(flattened_logits, dim=0)    # shape: [N, C]
        final_targets = torch.cat(flattened_targets, dim=0)  # shape: [N]
        
        accuracy_metrics = self.calculate_accuracy_metrics(final_logits, final_targets.view(1, -1))
        
        metrics = {
            'word_error_rate': self.calculate_wer(all_predictions_text, all_references_text),
            'character_error_rate': self.calculate_cer(all_predictions_text, all_references_text),
            'senone_accuracy': accuracy_metrics['senone_accuracy'],
            'top5_senone_accuracy': accuracy_metrics['top5_accuracy'],
        }
        
        # --- NEW: Prepare examples for printing ---
        num_examples = min(5, len(all_predictions_text)) # Show up to 5 examples
        examples = list(zip(all_predictions_text[:num_examples], all_references_text[:num_examples]))
        
        return metrics, examples

    def print_evaluation_report(self, metrics: Dict[str, float], examples: List[Tuple[str, str]], model_name: str = "Model"):
        """
        Prints a formatted evaluation report, now including qualitative examples.
        """
        if not metrics:
            print(f"No metrics to report for {model_name}.")
            return
            
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT - {model_name}")
        print(f"{'='*60}")
        
        print(f"📊 Text-Level Metrics (Approximate):")
        print(f"  Word Error Rate (WER):     {metrics.get('word_error_rate', 0)*100:.2f}%")
        print(f"  Character Error Rate (CER): {metrics.get('character_error_rate', 0)*100:.2f}%")
        
        print(f"\n🎯 Senone-Level Metrics (Frame Accuracy):")
        print(f"  Senone Accuracy:           {metrics.get('senone_accuracy', 0)*100:.2f}%")
        print(f"  Top-5 Senone Accuracy:     {metrics.get('top5_senone_accuracy', 0)*100:.2f}%")

        # --- NEW: Print the qualitative examples ---
        if examples:
            print(f"\n📝 Qualitative Examples (Prediction vs. Reference):")
            for i, (prediction, reference) in enumerate(examples):
                print(f"  --- Example {i+1} ---")
                print(f"  [Reference]: {reference}")
                print(f"  [Predicted]: {prediction}")
        
        print(f"\n🎯 Expected WER vs Actual (from paper):")
        wer = metrics.get('word_error_rate', 0) * 100
        print(f"  Baseline (Hindi only):     24.58% WER")
        print(f"  Expected GRL:              17.87% WER")
        print(f"  Expected DSN:              17.26% WER")
        print(f"  Your Model's Approx. WER:  {wer:.2f}% WER")
        
        print(f"{'='*60}\n")
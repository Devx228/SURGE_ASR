# complete_evaluation.py
#!/usr/bin/env python3
"""
Complete evaluation script to test your trained model
"""

import torch
import os
import logging
from fixed_config import get_config
from models import DSNModel, GRLModel
from fixed_dataset import create_data_loaders
from evaluation_metrics import ASREvaluator, quick_evaluation_demo
import editdistance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Complete evaluator to test your model thoroughly"""
    
    def __init__(self, model_path, model_type='DSN', device='cpu'):
        self.config = get_config()
        self.device = device
        self.model_type = model_type
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Load data
        self.load_test_data()
        
        # Initialize evaluators
        self.simple_evaluator = ASREvaluator()
        
    def load_model(self, model_path):
        """Load the trained model"""
        print(f"Loading model from: {model_path}")
        
        # Create model
        if self.model_type == 'DSN':
            model = DSNModel(
                input_dim=self.config['input_dim'],
                private_hidden=self.config['private_hidden'],
                shared_hidden=self.config['hidden_dim'],
                num_senones=self.config['num_senones'],
                num_domains=self.config['num_domains']
            )
        else:  # GRL
            model = GRLModel(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_senones=self.config['num_senones'],
                num_domains=self.config['num_domains']
            )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded successfully")
            print(f"✅ Model parameters on: {next(model.parameters()).device}")
            
            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model has {total_params:,} parameters")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def load_test_data(self):
        """Load test data"""
        print("Loading test data...")
        try:
            _, _, self.test_loader, self.test_data = create_data_loaders(self.config)
            if self.test_loader:
                print(f"✅ Test data loaded: {len(self.test_data[0])} samples")
            else:
                print("❌ Failed to load test data")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.test_loader = None
    
    def test_model_forward_pass(self):
        """Test basic model forward pass"""
        print(f"\n{'='*60}")
        print("TESTING MODEL FORWARD PASS")
        print(f"{'='*60}")
        
        if not self.test_loader:
            print("❌ No test data available")
            return
        
        try:
            # Get one batch
            batch = next(iter(self.test_loader))
            features = batch['features'].to(self.device)
            transcripts = batch['transcripts']
            
            print(f"📊 Batch info:")
            print(f"  Features shape: {features.shape}")
            print(f"  Features device: {features.device}")
            print(f"  Transcripts: {len(transcripts)}")
            
            # Forward pass
            with torch.no_grad():
                if self.model_type == 'DSN':
                    # Test both domains
                    print(f"\n🧪 Testing DSN model:")
                    
                    # Source domain (Hindi)
                    source_outputs = self.model(features, domain_id=0)
                    senone_logits_src, domain_logits_src, reconstructed_src, shared_src, private_src = source_outputs
                    print(f"  Source domain (Hindi) - Senone logits shape: {senone_logits_src.shape}")
                    
                    # Target domain (Sanskrit)
                    target_outputs = self.model(features, domain_id=1)
                    senone_logits_tgt, domain_logits_tgt, reconstructed_tgt, shared_tgt, private_tgt = target_outputs
                    print(f"  Target domain (Sanskrit) - Senone logits shape: {senone_logits_tgt.shape}")
                    
                    return {
                        'source_senone_logits': senone_logits_src,
                        'target_senone_logits': senone_logits_tgt,
                        'transcripts': transcripts
                    }
                    
                else:  # GRL
                    print(f"\n🧪 Testing GRL model:")
                    senone_logits, domain_logits = self.model(features, alpha=0.0)
                    print(f"  Senone logits shape: {senone_logits.shape}")
                    print(f"  Domain logits shape: {domain_logits.shape}")
                    
                    return {
                        'senone_logits': senone_logits,
                        'transcripts': transcripts
                    }
                    
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            return None
    
    def test_simple_decoder(self, outputs):
        """Test the simple decoder (same as training)"""
        print(f"\n{'='*60}")
        print("TESTING SIMPLE DECODER (Same as Training)")
        print(f"{'='*60}")
        
        if not outputs:
            print("❌ No model outputs to decode")
            return
        
        try:
            # Use the same decoder as training
            if self.model_type == 'DSN':
                # Test both domains
                print("🎯 Testing Source Domain (Hindi):")
                src_predictions = self.simple_evaluator.simple_decode_senones(outputs['source_senone_logits'])
                
                print("🎯 Testing Target Domain (Sanskrit):")
                tgt_predictions = self.simple_evaluator.simple_decode_senones(outputs['target_senone_logits'])
                
                transcripts = outputs['transcripts']
                
                print(f"\n📝 Results Comparison:")
                for i in range(min(5, len(transcripts))):
                    print(f"\n  Sample {i+1}:")
                    print(f"    Reference:    {transcripts[i][:80]}...")
                    print(f"    Source (Hi):  {src_predictions[i]}")
                    print(f"    Target (Sa):  {tgt_predictions[i]}")
                
                # Calculate WER for both
                src_wer = self.calculate_wer(src_predictions, transcripts)
                tgt_wer = self.calculate_wer(tgt_predictions, transcripts)
                
                print(f"\n📊 Word Error Rates:")
                print(f"  Source Domain (Hindi):    {src_wer*100:.2f}%")
                print(f"  Target Domain (Sanskrit): {tgt_wer*100:.2f}%")
                
            else:  # GRL
                predictions = self.simple_evaluator.simple_decode_senones(outputs['senone_logits'])
                transcripts = outputs['transcripts']
                
                print(f"\n📝 Results:")
                for i in range(min(5, len(transcripts))):
                    print(f"  {i+1}. Reference: {transcripts[i][:80]}...")
                    print(f"     Predicted:  {predictions[i]}")
                
                wer = self.calculate_wer(predictions, transcripts)
                print(f"\n📊 Word Error Rate: {wer*100:.2f}%")
                
        except Exception as e:
            print(f"❌ Error in simple decoder test: {e}")
    
    def calculate_wer(self, predictions, references):
        """Calculate Word Error Rate"""
        total_words = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.strip().split()
            ref_words = ref.strip().split()
            
            errors = editdistance.eval(pred_words, ref_words)
            total_errors += errors
            total_words += len(ref_words)
        
        return total_errors / total_words if total_words > 0 else 1.0
    
    def run_quick_evaluation_demo(self):
        """Run the same quick evaluation as during training"""
        print(f"\n{'='*60}")
        print("RUNNING QUICK EVALUATION (Same as Training)")
        print(f"{'='*60}")
        
        if not self.test_loader:
            print("❌ No test data available")
            return
        
        try:
            quick_evaluation_demo(self.model, self.test_loader, self.device)
        except Exception as e:
            print(f"❌ Error in quick evaluation: {e}")
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation"""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}")
        
        if not self.test_loader:
            print("❌ No test data available")
            return
        
        try:
            metrics = self.simple_evaluator.evaluate_model(self.model, self.test_loader, self.device)
            self.simple_evaluator.print_evaluation_report(metrics, f"{self.model_type} Model")
            return metrics
        except Exception as e:
            print(f"❌ Error in comprehensive evaluation: {e}")
            return None
    
    def run_all_tests(self):
        """Run all evaluation tests"""
        print(f"\n🚀 STARTING COMPREHENSIVE EVALUATION")
        print(f"Model: {self.model_type}")
        print(f"Device: {self.device}")
        
        if not self.model:
            print("❌ No model loaded, exiting")
            return
        
        # Test 1: Forward pass
        outputs = self.test_model_forward_pass()
        
        # Test 2: Simple decoder (same as training)
        self.test_simple_decoder(outputs)
        
        # Test 3: Quick evaluation (same as training)
        self.run_quick_evaluation_demo()
        
        # Test 4: Comprehensive evaluation
        metrics = self.run_comprehensive_evaluation()
        
        print(f"\n🎉 EVALUATION COMPLETE!")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='DSN',
                       choices=['DSN', 'GRL'],
                       help='Model type')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print(f"Available models:")
        for f in os.listdir('./outputs'):
            if f.endswith('.pth'):
                print(f"  ./outputs/{f}")
        return
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    evaluator.run_all_tests()

if __name__ == "__main__":
    main()

# Alternative: Quick test without command line arguments
def quick_test():
    """Quick test function"""
    # List available models
    model_files = [f for f in os.listdir('./outputs') if f.endswith('.pth')]
    if not model_files:
        print("❌ No model files found in ./outputs/")
        return
    
    print("📁 Available models:")
    for i, f in enumerate(model_files):
        print(f"  {i+1}. {f}")
    
    # Use the latest model
    latest_model = max(model_files)
    model_path = f'./outputs/{latest_model}'
    
    print(f"\n🚀 Testing latest model: {latest_model}")
    
    evaluator = ComprehensiveEvaluator(
        model_path=model_path,
        model_type='DSN',
        device='cpu'
    )
    
    evaluator.run_all_tests()

# Uncomment this to run quick test
# quick_test()
# import torch
# import torch.nn as nn
# from gradient_reversal import GradientReversalLayer
# import torch.nn.functional as F

# # ============================================================================
# # GRL MODEL IMPLEMENTATION
# # ============================================================================

# class GRLModel(nn.Module):
#     """
#     Gradient Reversal Layer based model for domain adversarial training.
    
#     Architecture:
#     - Feature Extractor (Gf): 6 hidden layers with 1024 nodes each
#     - Senone Classifier (Gy): 2 hidden layers + output (3080 senones for Hindi)
#     - Domain Classifier (Gd): 1 hidden layer (256 nodes) + output (2 domains)
#     """
    
#     def __init__(self, input_dim=1320, hidden_dim=1024, num_senones=3080, num_domains=2):
#         super(GRLModel, self).__init__()
        
#         # Feature Extractor (Gf) - 6 hidden layers
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU()
#         )
        
#         # Senone Classifier (Gy) - 2 hidden layers + output
#         self.senone_classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_senones)
#         )
        
#         # Domain Classifier (Gd) - 1 hidden layer + output
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, num_domains)
#         )
        
#         # Gradient Reversal Layer
#         self.grl = GradientReversalLayer()
        
#     def forward(self, x, alpha=1.0):
#         """
#         Forward pass through GRL model
        
#         Args:
#             x: Input features [batch_size, seq_len, input_dim]
#             alpha: Domain adaptation factor for gradient reversal
        
#         Returns:
#             senone_logits: Senone classification outputs
#             domain_logits: Domain classification outputs
#         """
#         batch_size, seq_len, input_dim = x.shape
        
#         # Reshape for processing: [batch_size * seq_len, input_dim]
#         x = x.reshape(-1, input_dim)
        
#         # Extract features
#         features = self.feature_extractor(x)  # [batch_size * seq_len, hidden_dim]
        
#         # Senone classification
#         senone_logits = self.senone_classifier(features)
        
#         # Domain classification with gradient reversal
#         self.grl.set_lambda(alpha)
#         reversed_features = self.grl(features)
#         domain_logits = self.domain_classifier(reversed_features)
        
#         # Reshape back: [batch_size, seq_len, num_classes]
#         senone_logits = senone_logits.reshape(batch_size, seq_len, -1)
#         domain_logits = domain_logits.reshape(batch_size, seq_len, -1)
        
#         return senone_logits, domain_logits

# # ============================================================================
# # DSN MODEL IMPLEMENTATION
# # ============================================================================

# class DSNModel(nn.Module):
#     """
#     Domain Separation Network model for cross-lingual ASR.
    
#     Architecture:
#     - Private Encoders: 4 hidden layers with 512 nodes each (source & target)
#     - Shared Encoder: 6 hidden layers with 1024 nodes each
#     - Shared Decoder: 3 hidden layers + output (1320 dimensions)
#     - Senone Classifier: 2 hidden layers + output
#     - Domain Classifier: 1 hidden layer + output (with GRL)
#     """
    
#     def __init__(self, input_dim=1320, private_hidden=512, shared_hidden=1024, 
#                  num_senones=3080, num_domains=2):
#         super(DSNModel, self).__init__()
        
#         # Private Encoders
#         self.source_private_encoder = self._build_private_encoder(input_dim, private_hidden)
#         self.target_private_encoder = self._build_private_encoder(input_dim, private_hidden)
        
#         # Shared Encoder
#         self.shared_encoder = self._build_shared_encoder(input_dim, shared_hidden)
        
#         # Shared Decoder
#         self.shared_decoder = nn.Sequential(
#             nn.Linear(shared_hidden + private_hidden, shared_hidden), nn.BatchNorm1d(shared_hidden), nn.ReLU(),
#             nn.Linear(shared_hidden, shared_hidden), nn.BatchNorm1d(shared_hidden), nn.ReLU(),
#             nn.Linear(shared_hidden, shared_hidden), nn.BatchNorm1d(shared_hidden), nn.ReLU(),
#             nn.Linear(shared_hidden, input_dim)
#         )
        
#         # Senone Classifier (from shared features)
#         self.senone_classifier = nn.Sequential(
#             nn.Linear(shared_hidden, shared_hidden), nn.BatchNorm1d(shared_hidden), nn.ReLU(),
#             nn.Linear(shared_hidden, shared_hidden), nn.BatchNorm1d(shared_hidden), nn.ReLU(),
#             nn.Linear(shared_hidden, num_senones)
#         )
        
#         # Domain Classifier (with GRL, as per the paper)
#         self.grl = GradientReversalLayer()
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(shared_hidden, 256), nn.BatchNorm1d(256), nn.ReLU(),
#             nn.Linear(256, num_domains)
#         )
        
#     def _build_private_encoder(self, input_dim, hidden_dim):
#         return nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
#         )
    
#     def _build_shared_encoder(self, input_dim, hidden_dim):
#         return nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
#         )
    
#     def forward(self, x, domain_id, alpha=1.0):
#         # Flatten the input: [batch_size, seq_len, input_dim] -> [batch_size * seq_len, input_dim]
#         # This is crucial for Batch Normalization to work correctly.
#         x_flat = x.reshape(-1, x.shape[-1])
        
#         # Private encoding depends on the domain
#         if domain_id == 0:  # Source domain
#             private_features = self.source_private_encoder(x_flat)
#         else:  # Target domain
#             private_features = self.target_private_encoder(x_flat)
        
#         # Shared encoding is common to both
#         shared_features = self.shared_encoder(x_flat)
        
#         # --- Outputs for Loss Calculation ---
        
#         # 1. Reconstruction
#         combined_for_recon = torch.cat([shared_features, private_features], dim=-1)
#         reconstructed = self.shared_decoder(combined_for_recon)
        
#         # 2. Senone classification (from shared features)
#         senone_logits = self.senone_classifier(shared_features)
        
#         # 3. Domain classification (from shared features with GRL)
#         self.grl.set_lambda(alpha)
#         reversed_shared = self.grl(shared_features)
#         domain_logits = self.domain_classifier(reversed_shared)
        
#         # Return all components needed for the loss function
#         # Note: We return the flat tensors, which is much easier for loss calculation.
#         return senone_logits, domain_logits, reconstructed, shared_features, private_features

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer from DANN paper.
    Forward pass is identity, backward pass multiplies gradients by -lambda.
    """
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val
    
    def set_lambda(self, lambda_val):
        self.lambda_val = lambda_val
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

# ============================================================================
# GRL MODEL IMPLEMENTATION
# ============================================================================

class GRLModel(nn.Module):
    """
    Gradient Reversal Layer based model for domain adversarial training.
    
    Architecture (as per paper):
    - Feature Extractor (Gf): 6 hidden layers with 1024 nodes each
    - Senone Classifier (Gy): 2 hidden layers + output (3080 senones for Hindi)
    - Domain Classifier (Gd): 1 hidden layer (256 nodes) + output (2 domains)
    """
    
    def __init__(self, input_dim=1320, hidden_dim=1024, num_senones=3080, num_domains=2):
        super(GRLModel, self).__init__()
        
        # Feature Extractor (Gf) - 6 hidden layers
        layers = []
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ])
        # Remaining 5 layers
        for _ in range(5):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
        self.feature_extractor = nn.Sequential(*layers)
        
        # Senone Classifier (Gy) - 2 hidden layers + output
        self.senone_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_senones)
        )
        
        # Domain Classifier (Gd) - 1 hidden layer + output
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer()
        
    def forward(self, x, alpha=1.0):
        """
        Forward pass through GRL model
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            alpha: Domain adaptation factor for gradient reversal
        
        Returns:
            senone_logits: Senone classification outputs [batch_size, seq_len, num_senones]
            domain_logits: Domain classification outputs [batch_size, seq_len, num_domains]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape for processing: [batch_size * seq_len, input_dim]
        x_flat = x.reshape(-1, input_dim)
        
        # Extract features
        features = self.feature_extractor(x_flat)  # [batch_size * seq_len, hidden_dim]
        
        # Senone classification
        senone_logits = self.senone_classifier(features)
        
        # Domain classification with gradient reversal
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)
        
        # Reshape back: [batch_size, seq_len, num_classes]
        senone_logits = senone_logits.reshape(batch_size, seq_len, -1)
        domain_logits = domain_logits.reshape(batch_size, seq_len, -1)
        
        return senone_logits, domain_logits

# ============================================================================
# DSN MODEL IMPLEMENTATION
# ============================================================================

class DSNModel(nn.Module):
    """
    Domain Separation Network model for cross-lingual ASR.
    
    Architecture (as per paper):
    - Private Encoders: 4 hidden layers with 512 nodes each (source & target)
    - Shared Encoder: 6 hidden layers with 1024 nodes each
    - Shared Decoder: 3 hidden layers + output (1320 dimensions)
    - Senone Classifier: 2 hidden layers + output
    - Domain Classifier: 1 hidden layer + output (with GRL)
    """
    
    def __init__(self, input_dim=1320, private_hidden=512, shared_hidden=1024, 
                 num_senones=3080, num_domains=2):
        super(DSNModel, self).__init__()
        
        # Private Encoders
        self.source_private_encoder = self._build_private_encoder(input_dim, private_hidden)
        self.target_private_encoder = self._build_private_encoder(input_dim, private_hidden)
        
        # Shared Encoder
        self.shared_encoder = self._build_shared_encoder(input_dim, shared_hidden)
        
        # Shared Decoder (takes concatenated shared + private features)
        self.shared_decoder = nn.Sequential(
            nn.Linear(shared_hidden + private_hidden, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, input_dim)
        )
        
        # Senone Classifier (from shared features only)
        self.senone_classifier = nn.Sequential(
            nn.Linear(shared_hidden, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, num_senones)
        )
        
        # Domain Classifier (with GRL, from shared features)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(shared_hidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )
        
    def _build_private_encoder(self, input_dim, hidden_dim):
        """Build private encoder with 4 hidden layers"""
        layers = []
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ])
        # Remaining 3 layers
        for _ in range(3):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
        return nn.Sequential(*layers)
    
    def _build_shared_encoder(self, input_dim, hidden_dim):
        """Build shared encoder with 6 hidden layers"""
        layers = []
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ])
        # Remaining 5 layers
        for _ in range(5):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x, domain_id, alpha=1.0):
        """
        Forward pass through DSN model
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            domain_id: 0 for source (Hindi), 1 for target (Sanskrit)
            alpha: Domain adaptation factor for gradient reversal
        
        Returns:
            senone_logits: [batch_size, seq_len, num_senones]
            domain_logits: [batch_size, seq_len, num_domains]
            reconstructed: [batch_size, seq_len, input_dim]
            shared_features: [batch_size, seq_len, shared_hidden]
            private_features: [batch_size, seq_len, private_hidden]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Flatten for batch norm: [batch_size * seq_len, input_dim]
        x_flat = x.reshape(-1, input_dim)
        
        # Private encoding based on domain
        if domain_id == 0:  # Source domain (Hindi)
            private_features_flat = self.source_private_encoder(x_flat)
        else:  # Target domain (Sanskrit)
            private_features_flat = self.target_private_encoder(x_flat)
        
        # Shared encoding (common for both domains)
        shared_features_flat = self.shared_encoder(x_flat)
        
        # Reconstruction
        combined_flat = torch.cat([shared_features_flat, private_features_flat], dim=-1)
        reconstructed_flat = self.shared_decoder(combined_flat)
        
        # Senone classification (from shared features only)
        senone_logits_flat = self.senone_classifier(shared_features_flat)
        
        # Domain classification (from shared features with GRL)
        self.grl.set_lambda(alpha)
        reversed_shared_flat = self.grl(shared_features_flat)
        domain_logits_flat = self.domain_classifier(reversed_shared_flat)
        
        # Reshape back to sequence format
        senone_logits = senone_logits_flat.reshape(batch_size, seq_len, -1)
        domain_logits = domain_logits_flat.reshape(batch_size, seq_len, -1)
        reconstructed = reconstructed_flat.reshape(batch_size, seq_len, input_dim)
        shared_features = shared_features_flat.reshape(batch_size, seq_len, -1)
        private_features = private_features_flat.reshape(batch_size, seq_len, -1)
        
        return senone_logits, domain_logits, reconstructed, shared_features, private_features

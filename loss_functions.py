import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# LOSS FUNCTIONS FOR GRL AND DSN MODELS
# ============================================================================

class GRLLoss(nn.Module):
    """
    Loss function for Gradient Reversal Layer model.
    Combines senone classification loss and domain classification loss.
    """
    def __init__(self):
        super(GRLLoss, self).__init__()
        self.senone_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def forward(self, source_senone_logits, source_domain_logits, target_domain_logits,
                source_senone_labels, source_domain_labels, target_domain_labels):
        """
        Compute GRL loss.
        
        Args:
            source_senone_logits: [batch_size, seq_len, num_senones]
            source_domain_logits: [batch_size, seq_len, num_domains]
            target_domain_logits: [batch_size, seq_len, num_domains]
            source_senone_labels: [batch_size, seq_len]
            source_domain_labels: [batch_size]
            target_domain_labels: [batch_size]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Senone classification loss (only on source/labeled data)
        senone_loss = self.senone_criterion(
            source_senone_logits.reshape(-1, source_senone_logits.size(-1)),
            source_senone_labels.reshape(-1)
        )
        
        # Domain classification loss
        # Average domain logits over sequence length
        source_domain_avg = source_domain_logits.mean(dim=1)
        target_domain_avg = target_domain_logits.mean(dim=1)
        
        domain_loss_source = self.domain_criterion(source_domain_avg, source_domain_labels)
        domain_loss_target = self.domain_criterion(target_domain_avg, target_domain_labels)
        domain_loss = domain_loss_source + domain_loss_target
        
        total_loss = senone_loss + domain_loss
        
        return total_loss, {
            'total_loss': total_loss,
            'senone_loss': senone_loss,
            'domain_loss': domain_loss
        }


class DSNLoss(nn.Module):
    """
    Complete DSN Loss function implementing all components from the paper:
    L = L_class + β*L_sim + γ*L_diff + δ*L_recon
    
    Where:
    - L_class: Senone classification loss (source only)
    - L_sim: Domain adversarial similarity loss
    - L_diff: Orthogonality constraint between private and shared features
    - L_recon: Reconstruction loss
    """
    
    def __init__(self, beta=0.25, gamma=0.075, delta=0.1, use_simse=False):
        super(DSNLoss, self).__init__()
        self.beta = beta    # Weight for similarity loss
        self.gamma = gamma  # Weight for difference loss
        self.delta = delta  # Weight for reconstruction loss
        self.use_simse = use_simse  # Whether to use SIMSE instead of MSE
        
        # Loss functions
        self.senone_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def compute_classification_loss(self, senone_logits, senone_labels):
        """
        L_class: Senone classification loss (negative log-likelihood).
        Applied only to source domain data.
        """
        # Reshape for loss computation
        logits_flat = senone_logits.reshape(-1, senone_logits.size(-1))
        labels_flat = senone_labels.reshape(-1)
        
        return self.senone_criterion(logits_flat, labels_flat)
    
    def compute_similarity_loss(self, source_domain_logits, target_domain_logits,
                               source_domain_labels, target_domain_labels):
        """
        L_sim: Domain adversarial similarity loss.
        Ensures shared features are domain-invariant.
        Uses GRL internally via the model.
        """
        # Average over sequence length
        source_domain_avg = source_domain_logits.mean(dim=1)
        target_domain_avg = target_domain_logits.mean(dim=1)
        
        # Domain classification losses
        source_loss = self.domain_criterion(source_domain_avg, source_domain_labels)
        target_loss = self.domain_criterion(target_domain_avg, target_domain_labels)
        
        return source_loss + target_loss
    
    def compute_difference_loss(self, source_shared, source_private, 
                               target_shared, target_private):
        """
        L_diff: Orthogonality constraint between private and shared features.
        As per equation (2) in the paper:
        L_diff = ||F_c^T * F_p||_F^2
        """
        # Reshape to 2D: [batch_size * seq_len, feature_dim]
        source_shared_2d = source_shared.reshape(-1, source_shared.size(-1))
        source_private_2d = source_private.reshape(-1, source_private.size(-1))
        target_shared_2d = target_shared.reshape(-1, target_shared.size(-1))
        target_private_2d = target_private.reshape(-1, target_private.size(-1))
        
        # Compute F_c^T * F_p for source
        source_correlation = torch.matmul(source_shared_2d.T, source_private_2d)
        source_diff_loss = torch.norm(source_correlation, p='fro') ** 2
        
        # Compute F_c^T * F_p for target
        target_correlation = torch.matmul(target_shared_2d.T, target_private_2d)
        target_diff_loss = torch.norm(target_correlation, p='fro') ** 2
        
        # Normalize by number of samples
        num_samples = source_shared_2d.size(0) + target_shared_2d.size(0)
        
        return (source_diff_loss + target_diff_loss) / num_samples
    
    def compute_reconstruction_loss(self, source_input, source_recon,
                                   target_input, target_recon):
        """
        L_recon: Reconstruction loss.
        Can use either MSE or SIMSE based on configuration.
        """
        if self.use_simse:
            source_loss = self._simse_loss(source_input, source_recon)
            target_loss = self._simse_loss(target_input, target_recon)
        else:
            source_loss = F.mse_loss(source_recon, source_input)
            target_loss = F.mse_loss(target_recon, target_input)
        
        return source_loss + target_loss
    
    def _simse_loss(self, x, x_hat):
        """
        Scale-Invariant Mean Squared Error (SIMSE) as per equation (4).
        SIMSE = (1/k)||x - x̂||²₂ - (1/k²)([x - x̂]·1_k)²
        """
        diff = x - x_hat
        k = diff.size(-1)  # Feature dimension
        
        # First term: (1/k)||x - x̂||²₂
        mse_term = torch.mean(diff ** 2)
        
        # Second term: (1/k²)([x - x̂]·1_k)²
        # This is essentially (mean(diff))²
        mean_diff = torch.mean(diff, dim=-1, keepdim=True)
        bias_term = torch.mean(mean_diff ** 2)
        
        return mse_term - bias_term
    
    def forward(self, source_outputs, target_outputs, 
                source_input, target_input,
                source_senone_labels,
                activate_adversarial=True):
        """
        Compute complete DSN loss.
        
        Args:
            source_outputs: tuple (senone_logits, domain_logits, reconstructed, shared, private)
            target_outputs: tuple (senone_logits, domain_logits, reconstructed, shared, private)
            source_input: Original source features [batch_size, seq_len, input_dim]
            target_input: Original target features [batch_size, seq_len, input_dim]
            source_senone_labels: Senone labels for source [batch_size, seq_len]
            activate_adversarial: Whether to activate adversarial losses (after 10k steps)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Unpack outputs
        (source_senone_logits, source_domain_logits, source_recon,
         source_shared, source_private) = source_outputs
        
        (_, target_domain_logits, target_recon,
         target_shared, target_private) = target_outputs
        
        batch_size = source_input.size(0)
        device = source_input.device
        
        # 1. Classification loss (L_class) - source only
        L_class = self.compute_classification_loss(source_senone_logits, source_senone_labels)
        
        # 2. Similarity loss (L_sim) - domain adversarial
        if activate_adversarial:
            source_domain_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            target_domain_labels = torch.ones(batch_size, dtype=torch.long, device=device)
            
            L_sim = self.compute_similarity_loss(
                source_domain_logits, target_domain_logits,
                source_domain_labels, target_domain_labels
            )
        else:
            L_sim = torch.tensor(0.0, device=device)
        
        # 3. Difference loss (L_diff) - orthogonality constraint
        L_diff = self.compute_difference_loss(
            source_shared, source_private,
            target_shared, target_private
        )
        
        # 4. Reconstruction loss (L_recon)
        L_recon = self.compute_reconstruction_loss(
            source_input, source_recon,
            target_input, target_recon
        )
        
        # Total loss as per equation (1)
        total_loss = L_class + self.beta * L_sim + self.gamma * L_diff + self.delta * L_recon
        
        loss_dict = {
            'total_loss': total_loss,
            'L_class': L_class,
            'L_sim': L_sim,
            'L_diff': L_diff,
            'L_recon': L_recon
        }
        
        return total_loss, loss_dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_loss_function(model_type, **kwargs):
    """
    Factory function to get appropriate loss function.
    
    Args:
        model_type: 'GRL' or 'DSN'
        **kwargs: Additional arguments for DSN loss
    
    Returns:
        Loss function instance
    """
    if model_type.upper() == 'GRL':
        return GRLLoss()
    elif model_type.upper() == 'DSN':
        return DSNLoss(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
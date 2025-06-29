# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import logging
# from models import GRLModel, DSNModel
# from loss_functions import DSNLoss

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ============================================================================
# # TRAINING FUNCTIONS
# # ============================================================================

# class Trainer:
#     """Trainer class for GRL and DSN models"""
    
#     def __init__(self, model, device='cuda', learning_rate=0.01, batch_size=32):
#         self.model = model.to(device)
#         self.device = device
#         self.batch_size = batch_size
        
#         # Optimizer with momentum as specified in paper
#         self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
#         # Learning rate scheduler - scale by 0.95 every 20000 steps
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.95)
        
#         # Loss functions
#         self.senone_criterion = nn.CrossEntropyLoss(ignore_index=-1)
#         self.domain_criterion = nn.CrossEntropyLoss()
#         self.dsn_criterion = DSNLoss()
        
#         # Training statistics
#         self.step = 0
#         self.epoch = 0
        
#     def get_lambda_p(self):
#         """Get domain adaptation factor that gradually changes from 0 to 1"""
#         # Gradual change as described in DANN paper
#         p = float(self.step) / 100000  # Adjust based on total steps
#         lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
#         return lambda_p
    
#     def train_grl_epoch(self, source_loader, target_loader):
#         """Train GRL model for one epoch"""
#         self.model.train()
        
#         source_iter = iter(source_loader)
#         target_iter = iter(target_loader)
        
#         total_senone_loss = 0
#         total_domain_loss = 0
#         num_batches = 0
        
#         while True:
#             try:
#                 # Get source batch (Hindi with labels)
#                 source_batch = next(source_iter)
#                 source_features = source_batch['features'].to(self.device)
#                 source_senones = source_batch['senones'].to(self.device)
#                 source_domains = torch.zeros(source_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get target batch (Sanskrit without senone labels)
#                 try:
#                     target_batch = next(target_iter)
#                 except StopIteration:
#                     # Reset target iterator if it runs out
#                     target_iter = iter(target_loader)
#                     target_batch = next(target_iter)
                
#                 target_features = target_batch['features'].to(self.device)
#                 target_domains = torch.ones(target_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get domain adaptation factor
#                 lambda_p = self.get_lambda_p()
                
#                 # Forward pass on source data
#                 source_senone_logits, source_domain_logits = self.model(source_features, lambda_p)
                
#                 # Forward pass on target data
#                 _, target_domain_logits = self.model(target_features, lambda_p)
                
#                 # Debug shapes
#                 logger.debug(f"Source features shape: {source_features.shape}")
#                 logger.debug(f"Source senones shape: {source_senones.shape}")
#                 logger.debug(f"Source senone logits shape: {source_senone_logits.shape}")
                
#                 # Handle variable sequence lengths
#                 # Get the minimum sequence length between predictions and targets
#                 batch_size, seq_len_pred, num_classes = source_senone_logits.shape
#                 _, seq_len_target = source_senones.shape
                
#                 min_seq_len = min(seq_len_pred, seq_len_target)
                
#                 # Truncate to minimum length
#                 source_senone_logits_truncated = source_senone_logits[:, :min_seq_len, :]
#                 source_senones_truncated = source_senones[:, :min_seq_len]
                
#                 # Create mask for valid positions (not -1)
#                 mask = source_senones_truncated != -1
                
#                 # Compute senone loss only on valid positions
#                 if mask.any():
#                     # Reshape for cross entropy loss
#                     senone_logits_flat = source_senone_logits_truncated[mask]
#                     senone_targets_flat = source_senones_truncated[mask]
                    
#                     senone_loss = self.senone_criterion(senone_logits_flat, senone_targets_flat)
#                 else:
#                     # No valid targets in this batch
#                     senone_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
#                 # Domain loss - average over sequence dimension first
#                 domain_loss_source = self.domain_criterion(
#                     source_domain_logits.mean(dim=1), source_domains
#                 )
#                 domain_loss_target = self.domain_criterion(
#                     target_domain_logits.mean(dim=1), target_domains
#                 )
#                 domain_loss = domain_loss_source + domain_loss_target
                
#                 total_loss = senone_loss + domain_loss
                
#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
                
#                 # Gradient clipping to prevent explosion
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
#                 self.optimizer.step()
#                 self.scheduler.step()
                
#                 # Statistics
#                 total_senone_loss += senone_loss.item()
#                 total_domain_loss += domain_loss.item()
#                 num_batches += 1
#                 self.step += 1
                
#                 if num_batches % 100 == 0:
#                     logger.info(f"Step {self.step}, Senone Loss: {senone_loss.item():.4f}, "
#                               f"Domain Loss: {domain_loss.item():.4f}, Lambda: {lambda_p:.4f}")
                
#             except StopIteration:
#                 break
#             except Exception as e:
#                 logger.error(f"Error in batch {num_batches}: {e}")
#                 logger.error(f"Shapes - features: {source_features.shape}, senones: {source_senones.shape}")
#                 continue
        
#         if num_batches > 0:
#             avg_senone_loss = total_senone_loss / num_batches
#             avg_domain_loss = total_domain_loss / num_batches
            
#             logger.info(f"Epoch {self.epoch} - Avg Senone Loss: {avg_senone_loss:.4f}, "
#                        f"Avg Domain Loss: {avg_domain_loss:.4f}")
#         else:
#             avg_senone_loss = avg_domain_loss = 0
#             logger.warning(f"Epoch {self.epoch} - No batches processed!")
        
#         self.epoch += 1
#         return avg_senone_loss, avg_domain_loss
    
#     def train_dsn_epoch(self, source_loader, target_loader):
#         """Train DSN model for one epoch"""
#         self.model.train()
        
#         source_iter = iter(source_loader)
#         target_iter = iter(target_loader)
        
#         total_loss = 0
#         num_batches = 0
#         loss_components = {
#             'senone_loss': 0,
#             'domain_loss': 0,
#             'reconstruction_loss': 0,
#             'similarity_loss': 0,
#             'dissimilarity_loss': 0
#         }
        
#         while True:
#             try:
#                 # Get source batch
#                 source_batch = next(source_iter)
#                 source_features = source_batch['features'].to(self.device)
#                 source_senones = source_batch['senones'].to(self.device)
#                 source_domains = torch.zeros(source_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get target batch
#                 try:
#                     target_batch = next(target_iter)
#                 except StopIteration:
#                     target_iter = iter(target_loader)
#                     target_batch = next(target_iter)
                    
#                 target_features = target_batch['features'].to(self.device)
#                 target_domains = torch.ones(target_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Ensure same sequence length for source and target
#                 min_seq_len = min(source_features.size(1), target_features.size(1))
#                 source_features = source_features[:, :min_seq_len, :]
#                 target_features = target_features[:, :min_seq_len, :]
#                 source_senones = source_senones[:, :min_seq_len]
                
#                 # Forward pass on source
#                 (source_senone_logits, source_domain_logits, source_reconstructed, 
#                  source_shared, source_private) = self.model(source_features, domain_id=0)
                
#                 # Forward pass on target
#                 (target_senone_logits, target_domain_logits, target_reconstructed, 
#                  target_shared, target_private) = self.model(target_features, domain_id=1)
                
#                 # Create dummy senone targets for target domain (all -1)
#                 target_senones = torch.full_like(source_senones, -1)
                
#                 # Compute losses with shared features for similarity
#                 source_losses = self.dsn_criterion(
#                     source_senone_logits, source_domain_logits, source_reconstructed,
#                     source_shared, source_private, source_features,
#                     source_senones, source_domains,
#                     shared_source=source_shared, shared_target=target_shared
#                 )
                
#                 target_losses = self.dsn_criterion(
#                     target_senone_logits, target_domain_logits, target_reconstructed,
#                     target_shared, target_private, target_features,
#                     target_senones, target_domains,
#                     shared_source=source_shared, shared_target=target_shared
#                 )
                
#                 # Combine losses based on training stage
#                 if self.step > 10000:
#                     # After 10k steps, use all losses
#                     total_batch_loss = (
#                         source_losses['total_loss'] + 
#                         target_losses['reconstruction_loss'] +
#                         target_losses['domain_loss']
#                     )
#                 else:
#                     # Initially, focus on senone and domain classification
#                     total_batch_loss = (
#                         source_losses['senone_loss'] + 
#                         source_losses['domain_loss'] + 
#                         target_losses['domain_loss']
#                     )
                
#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 total_batch_loss.backward()
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
#                 self.optimizer.step()
#                 self.scheduler.step()
                
#                 # Statistics
#                 total_loss += total_batch_loss.item()
#                 for key in loss_components:
#                     if key in source_losses:
#                         loss_components[key] += source_losses[key].item()
                
#                 num_batches += 1
#                 self.step += 1
                
#                 if num_batches % 100 == 0:
#                     logger.info(f"Step {self.step}, Total Loss: {total_batch_loss.item():.4f}")
#                     logger.info(f"  Components - Senone: {source_losses['senone_loss'].item():.4f}, "
#                               f"Domain: {source_losses['domain_loss'].item():.4f}, "
#                               f"Recon: {source_losses['reconstruction_loss'].item():.4f}")
                
#             except StopIteration:
#                 break
#             except Exception as e:
#                 logger.error(f"Error in batch {num_batches}: {e}")
#                 continue
        
#         if num_batches > 0:
#             avg_loss = total_loss / num_batches
#             logger.info(f"Epoch {self.epoch} - Avg Loss: {avg_loss:.4f}")
            
#             # Log component averages
#             logger.info("Loss components:")
#             for key, value in loss_components.items():
#                 logger.info(f"  {key}: {value/num_batches:.4f}")
#         else:
#             avg_loss = 0
#             logger.warning(f"Epoch {self.epoch} - No batches processed!")
        
#         self.epoch += 1
#         return avg_loss



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import logging
# from models import GRLModel, DSNModel
# from loss_functions import DSNLoss
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ============================================================================
# # TRAINING FUNCTIONS
# # ============================================================================

# class Trainer:
#     """Trainer class for GRL and DSN models"""
    
#     def __init__(self, model, device='cuda', learning_rate=0.001, batch_size=32):
#         self.model = model.to(device)
#         self.device = device
#         self.batch_size = batch_size
        
#         # Optimizer with momentum as specified in paper
#         self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
#         # Learning rate scheduler - scale by 0.95 every 20000 steps
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.95)
        
#         # Loss functions
#         self.senone_criterion = nn.CrossEntropyLoss(ignore_index=-1)
#         self.domain_criterion = nn.CrossEntropyLoss()
#         self.dsn_criterion = DSNLoss()
        
#         # Training statistics
#         self.step = 0
#         self.epoch = 0
        
#     def get_lambda_p(self):
#         """Get domain adaptation factor that gradually changes from 0 to 1"""
#         # Gradual change as described in DANN paper
#         p = float(self.step) / 100000  # Adjust based on total steps
#         lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
#         return max(0.0, lambda_p)  # Ensure non-negative
    
#     def train_grl_epoch(self, source_loader, target_loader):
#         """Train GRL model for one epoch"""
#         self.model.train()
        
#         source_iter = iter(source_loader)
#         target_iter = iter(target_loader)
        
#         total_senone_loss = 0
#         total_domain_loss = 0
#         num_batches = 0
        
#         # Calculate number of batches (use smaller dataset size)
#         num_batches_total = min(len(source_loader), len(target_loader))
        
#         pbar = tqdm(range(num_batches_total), desc=f"Epoch {self.epoch + 1}")
        
#         for _ in pbar:
#             try:
#                 # Get source batch (Hindi with labels)
#                 try:
#                     source_batch = next(source_iter)
#                 except StopIteration:
#                     source_iter = iter(source_loader)
#                     source_batch = next(source_iter)
                
#                 source_features = source_batch['features'].to(self.device)
#                 source_senones = source_batch['senones'].to(self.device)
#                 source_domains = torch.zeros(source_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get target batch (Sanskrit without senone labels)
#                 try:
#                     target_batch = next(target_iter)
#                 except StopIteration:
#                     target_iter = iter(target_loader)
#                     target_batch = next(target_iter)
                
#                 target_features = target_batch['features'].to(self.device)
#                 target_domains = torch.ones(target_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get domain adaptation factor
#                 lambda_p = self.get_lambda_p()
                
#                 # Forward pass on source data
#                 source_senone_logits, source_domain_logits = self.model(source_features, lambda_p)
                
#                 # Forward pass on target data
#                 target_senone_logits, target_domain_logits = self.model(target_features, lambda_p)
                
#                 # Senone loss (only on source domain)
#                 senone_loss = self.senone_criterion(
#                     source_senone_logits.reshape(-1, source_senone_logits.size(-1)),
#                     source_senones.reshape(-1)
#                 )
                
#                 # Domain loss on both domains
#                 source_domain_loss = self.domain_criterion(
#                     source_domain_logits.mean(dim=1), source_domains
#                 )
#                 target_domain_loss = self.domain_criterion(
#                     target_domain_logits.mean(dim=1), target_domains
#                 )
#                 domain_loss = (source_domain_loss + target_domain_loss) / 2
                
#                 # Total loss
#                 total_loss = senone_loss + domain_loss
                
#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
#                 self.optimizer.step()
#                 self.scheduler.step()
                
#                 # Statistics
#                 total_senone_loss += senone_loss.item()
#                 total_domain_loss += domain_loss.item()
#                 num_batches += 1
#                 self.step += 1
                
#                 # Update progress bar
#                 pbar.set_postfix({
#                     'S_Loss': f'{senone_loss.item():.4f}',
#                     'D_Loss': f'{domain_loss.item():.4f}',
#                     'λ': f'{lambda_p:.3f}'
#                 })
                
#             except Exception as e:
#                 logger.error(f"Error in batch: {e}")
#                 continue
        
#         avg_senone_loss = total_senone_loss / num_batches if num_batches > 0 else 0
#         avg_domain_loss = total_domain_loss / num_batches if num_batches > 0 else 0
        
#         logger.info(f"Epoch {self.epoch + 1} - Avg Senone Loss: {avg_senone_loss:.4f}, "
#                    f"Avg Domain Loss: {avg_domain_loss:.4f}")
        
#         self.epoch += 1
#         return avg_senone_loss, avg_domain_loss
    
#     def train_dsn_epoch(self, source_loader, target_loader):
#         """Train DSN model for one epoch"""
#         self.model.train()
        
#         source_iter = iter(source_loader)
#         target_iter = iter(target_loader)
        
#         total_loss = 0
#         num_batches = 0
#         loss_components = {
#             'senone_loss': 0,
#             'domain_loss': 0,
#             'reconstruction_loss': 0,
#             'similarity_loss': 0,
#             'dissimilarity_loss': 0
#         }
        
#         # Calculate number of batches
#         num_batches_total = min(len(source_loader), len(target_loader))
        
#         pbar = tqdm(range(num_batches_total), desc=f"Epoch {self.epoch + 1}")
        
#         for batch_idx in pbar:
#             try:
#                 # Get source batch
#                 try:
#                     source_batch = next(source_iter)
#                 except StopIteration:
#                     source_iter = iter(source_loader)
#                     source_batch = next(source_iter)
                
#                 source_features = source_batch['features'].to(self.device)
#                 source_senones = source_batch['senones'].to(self.device)
#                 source_domains = torch.zeros(source_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get target batch
#                 try:
#                     target_batch = next(target_iter)
#                 except StopIteration:
#                     target_iter = iter(target_loader)
#                     target_batch = next(target_iter)
                    
#                 target_features = target_batch['features'].to(self.device)
#                 target_domains = torch.ones(target_features.size(0), dtype=torch.long).to(self.device)
                
#                 # Get domain adaptation factor
#                 lambda_p = self.get_lambda_p()
                
#                 # Forward pass on source
#                 (source_senone_logits, source_domain_logits, source_reconstructed, 
#                  source_shared, source_private) = self.model(source_features, domain_id=0, alpha=lambda_p)
                
#                 # Forward pass on target
#                 (target_senone_logits, target_domain_logits, target_reconstructed, 
#                  target_shared, target_private) = self.model(target_features, domain_id=1, alpha=lambda_p)
                
#                 # Accumulate shared features for similarity loss
#                 self.dsn_criterion.accumulate_shared_features(source_shared, domain_id=0)
#                 self.dsn_criterion.accumulate_shared_features(target_shared, domain_id=1)
                
#                 # Compute similarity loss every N batches
#                 compute_similarity = (batch_idx + 1) % 10 == 0 and self.step > 10000
                
#                 # Compute source losses
#                 source_losses = self.dsn_criterion(
#                     source_senone_logits, source_domain_logits, source_reconstructed,
#                     source_shared, source_private, source_features,
#                     source_senones, source_domains,
#                     compute_similarity=compute_similarity
#                 )
                
#                 # Target domain - no senone labels
#                 target_senones = torch.full_like(source_senones[:target_features.size(0)], -1)
                
#                 # Compute target losses (no senone loss)
#                 target_losses = self.dsn_criterion(
#                     target_senone_logits, target_domain_logits, target_reconstructed,
#                     target_shared, target_private, target_features,
#                     target_senones, target_domains,
#                     compute_similarity=False  # Already computed with source
#                 )
                
#                 # Combine losses
#                 if self.step > 10000:
#                     # After warmup, use all losses
#                     total_batch_loss = (
#                         source_losses['senone_loss'] + 
#                         0.5 * (source_losses['domain_loss'] + target_losses['domain_loss']) +
#                         self.dsn_criterion.beta * (source_losses['reconstruction_loss'] + target_losses['reconstruction_loss']) +
#                         self.dsn_criterion.gamma * source_losses['similarity_loss'] +
#                         self.dsn_criterion.delta * (source_losses['dissimilarity_loss'] + target_losses['dissimilarity_loss'])
#                     )
#                 else:
#                     # During warmup, focus on senone and domain classification
#                     total_batch_loss = (
#                         source_losses['senone_loss'] + 
#                         0.5 * (source_losses['domain_loss'] + target_losses['domain_loss'])
#                     )
                
#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 total_batch_loss.backward()
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
#                 self.optimizer.step()
#                 self.scheduler.step()
                
#                 # Statistics
#                 total_loss += total_batch_loss.item()
#                 for key in loss_components:
#                     if key in source_losses:
#                         loss_components[key] += source_losses[key].item()
                
#                 num_batches += 1
#                 self.step += 1
                
#                 # Update progress bar
#                 pbar.set_postfix({
#                     'Loss': f'{total_batch_loss.item():.4f}',
#                     'S': f'{source_losses["senone_loss"].item():.3f}',
#                     'D': f'{source_losses["domain_loss"].item():.3f}',
#                     'R': f'{source_losses["reconstruction_loss"].item():.3f}'
#                 })
                
#             except Exception as e:
#                 logger.error(f"Error in batch {batch_idx}: {e}")
#                 continue
        
#         avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
#         logger.info(f"Epoch {self.epoch + 1} - Avg Loss: {avg_loss:.4f}")
#         logger.info("Loss components:")
#         for key, value in loss_components.items():
#             logger.info(f"  {key}: {value/num_batches:.4f}")
        
#         self.epoch += 1
#         return avg_loss



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm

# Import the models and loss functions
from models import GRLModel, DSNModel
from loss_functions import GRLLoss, DSNLoss, get_loss_function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """
    Trainer class for GRL and DSN models following the paper specifications.
    Now properly integrated with loss_functions.py
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.01, batch_size=32):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Determine model type
        if isinstance(model, GRLModel):
            self.model_type = 'GRL'
        elif isinstance(model, DSNModel):
            self.model_type = 'DSN'
        else:
            raise ValueError("Unknown model type. Expected GRLModel or DSNModel")
        
        # Optimizer with momentum=0.9 as specified in paper
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        # Learning rate scheduler - scale by 0.95 every 20000 steps (as per paper)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.95)
        
        # Initialize loss function based on model type
        self.loss_fn = get_loss_function(
            self.model_type,
            beta=0.25,      # DSN reconstruction loss weight
            gamma=0.075,    # DSN similarity loss weight  
            delta=0.1,      # DSN dissimilarity loss weight
            use_simse=False # Use MSE instead of SIMSE for reconstruction
        )
        
        self.step = 0
        self.epoch = 0
        
    def get_alpha(self):
        """
        Get domain adaptation factor alpha that gradually changes from 0 to 1.
        Following the DANN paper: α = 2 / (1 + exp(-10 * p)) - 1
        where p is the training progress.
        """
        p = float(self.step) / 100000  # Assuming ~100k total steps for 20 epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        return alpha
    
    def train_grl_epoch(self, source_loader, target_loader):
        """Train GRL model for one epoch using GRLLoss."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0,
            'senone_loss': 0,
            'domain_loss': 0
        }
        
        # Use the same number of frames from source and target (as per paper)
        num_batches = min(len(source_loader), len(target_loader))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        pbar = tqdm(range(num_batches), desc=f"GRL Epoch {self.epoch + 1}")
        
        for i in pbar:
            try:
                # Get source batch (Hindi - labeled)
                source_batch = next(source_iter)
                source_features = source_batch['features'].to(self.device)
                source_senones = source_batch['senones'].to(self.device)
                
                # Get target batch (Sanskrit - unlabeled)
                target_batch = next(target_iter)
                target_features = target_batch['features'].to(self.device)
                
                # Get current alpha for gradient reversal
                alpha = self.get_alpha()
                
                # Forward pass for source
                source_senone_logits, source_domain_logits = self.model(source_features, alpha=alpha)
                
                # Forward pass for target
                _, target_domain_logits = self.model(target_features, alpha=alpha)
                
                # Create domain labels
                batch_size = source_features.size(0)
                source_domain_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                target_domain_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
                
                # Compute loss using GRLLoss
                total_loss, loss_dict = self.loss_fn(
                    source_senone_logits=source_senone_logits,
                    source_domain_logits=source_domain_logits,
                    target_domain_logits=target_domain_logits,
                    source_senone_labels=source_senones,
                    source_domain_labels=source_domain_labels,
                    target_domain_labels=target_domain_labels
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                
                # Update statistics
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Senone': f'{loss_dict["senone_loss"].item():.4f}',
                    'Domain': f'{loss_dict["domain_loss"].item():.4f}',
                    'α': f'{alpha:.3f}',
                    'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
                
            except StopIteration:
                # Restart iterators if one runs out
                source_iter = iter(source_loader)
                target_iter = iter(target_loader)
                continue
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate averages
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        logger.info(f"GRL Epoch {self.epoch + 1} - "
                   f"Avg Loss: {epoch_losses['total_loss']:.4f}, "
                   f"Senone: {epoch_losses['senone_loss']:.4f}, "
                   f"Domain: {epoch_losses['domain_loss']:.4f}")
        
        self.epoch += 1
        return epoch_losses['senone_loss'], epoch_losses['domain_loss']
    
    def train_dsn_epoch(self, source_loader, target_loader):
        """Train DSN model for one epoch using DSNLoss."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0,
            'L_class': 0,
            'L_sim': 0,
            'L_diff': 0,
            'L_recon': 0
        }
        
        # Use the same number of frames from source and target
        num_batches = min(len(source_loader), len(target_loader))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        pbar = tqdm(range(num_batches), desc=f"DSN Epoch {self.epoch + 1}")
        
        for i in pbar:
            try:
                # Get source batch (Hindi - labeled)
                source_batch = next(source_iter)
                source_features = source_batch['features'].to(self.device)
                source_senones = source_batch['senones'].to(self.device)
                
                # Get target batch (Sanskrit - unlabeled)
                target_batch = next(target_iter)
                target_features = target_batch['features'].to(self.device)
                
                # Determine if adversarial losses should be active
                # Paper: "domain adversarial similarity losses are activated only after 10000 steps"
                activate_adversarial = self.step >= 10000
                alpha = self.get_alpha() if activate_adversarial else 0.0
                
                # Forward pass for source (domain_id=0)
                source_outputs = self.model(source_features, domain_id=0, alpha=alpha)
                
                # Forward pass for target (domain_id=1)
                target_outputs = self.model(target_features, domain_id=1, alpha=alpha)
                
                # Compute loss using DSNLoss
                total_loss, loss_dict = self.loss_fn(
                    source_outputs=source_outputs,
                    target_outputs=target_outputs,
                    source_input=source_features,
                    target_input=target_features,
                    source_senone_labels=source_senones,
                    activate_adversarial=activate_adversarial
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                
                # Update statistics
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Total': f'{total_loss.item():.4f}',
                    'Class': f'{loss_dict["L_class"].item():.3f}',
                    'Sim': f'{loss_dict["L_sim"].item():.3f}',
                    'Recon': f'{loss_dict["L_recon"].item():.3f}',
                    'α': f'{alpha:.3f}' if activate_adversarial else 'OFF',
                    'Step': self.step
                })
                
            except StopIteration:
                # Restart iterators if one runs out
                source_iter = iter(source_loader)
                target_iter = iter(target_loader)
                continue
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate averages
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        logger.info(f"DSN Epoch {self.epoch + 1} - "
                   f"Avg Loss: {epoch_losses['total_loss']:.4f}, "
                   f"Class: {epoch_losses['L_class']:.4f}, "
                   f"Sim: {epoch_losses['L_sim']:.4f}, "
                   f"Recon: {epoch_losses['L_recon']:.4f}, "
                   f"Diff: {epoch_losses['L_diff']:.4f}")
        
        self.epoch += 1
        return epoch_losses['total_loss']
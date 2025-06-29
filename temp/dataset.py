# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# import numpy as np
# from typing import List
# from feature_extraction import FeatureExtractor

# # ============================================================================
# # DATASET CLASSES
# # ============================================================================

# class ASRDataset(Dataset):
#     """Dataset class for ASR training with domain labels"""
    
#     def __init__(self, audio_files: List[str], transcripts: List[str], 
#                  senones: List[np.ndarray], domain_labels: List[int], 
#                  feature_extractor: FeatureExtractor):
#         self.audio_files = audio_files
#         self.transcripts = transcripts
#         self.senones = senones
#         self.domain_labels = domain_labels
#         self.feature_extractor = feature_extractor
        
#     def __len__(self):
#         return len(self.audio_files)
    
#     def __getitem__(self, idx):
#         # Extract features
#         features = self.feature_extractor.extract_features(self.audio_files[idx])
        
#         return {
#             'features': torch.FloatTensor(features),
#             'senones': torch.LongTensor(self.senones[idx]),
#             'domain': torch.LongTensor([self.domain_labels[idx]]),
#             'transcript': self.transcripts[idx]
#         }

# def collate_fn(batch):
#     """Custom collate function for variable length sequences"""
#     features = [item['features'] for item in batch]
#     senones = [item['senones'] for item in batch]
#     domains = torch.cat([item['domain'] for item in batch])
#     transcripts = [item['transcript'] for item in batch]
    
#     # Pad sequences
#     features_padded = pad_sequence(features, batch_first=True)
#     senones_padded = pad_sequence(senones, batch_first=True, padding_value=-1)
    
#     # Create length tensors
#     feature_lengths = torch.LongTensor([len(f) for f in features])
#     senone_lengths = torch.LongTensor([len(s) for s in senones])
    
#     return {
#         'features': features_padded,
#         'senones': senones_padded,
#         'domains': domains,
#         'feature_lengths': feature_lengths,
#         'senone_lengths': senone_lengths,
#         'transcripts': transcripts
#     }
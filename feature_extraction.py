import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pydub import AudioSegment
import io

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Extract 40-dimensional filterbank features with delta and acceleration coefficients.
    Spliced with context of 5 frames on each side (total: 40 x 3 x 11 = 1320 dimensions)
    """
    def __init__(self, n_mels=40, sample_rate=8000, n_fft=512, hop_length=160, context_frames=5):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.context_frames = context_frames
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract filterbank features from audio file"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 8kHz if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Extract mel-filterbank features
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        mel_features = mel_spectrogram(waveform).squeeze(0).T  # [T, n_mels]
        
        # Convert to log scale
        log_mel = torch.log(mel_features + 1e-8)
        
        # Compute delta and acceleration coefficients
        delta = self._compute_delta(log_mel)
        delta_delta = self._compute_delta(delta)
        
        # Concatenate features
        features = torch.cat([log_mel, delta, delta_delta], dim=1)  # [T, 120]
        
        # Apply cepstral mean and variance normalization
        features = self._cmvn(features)
        
        # Add context frames (splicing)
        features = self._add_context(features)  # [T, 1320]
        
        return features.numpy()
    
    def _compute_delta(self, features: torch.Tensor) -> torch.Tensor:
        """Compute delta coefficients using manual padding"""
        T, D = features.shape
        
        # Manual replicate padding for 2D tensor
        padded_features = torch.zeros(T + 4, D)
        padded_features[2:T+2] = features
        # Replicate first frame for left padding
        padded_features[:2] = features[0:1].repeat(2, 1)
        # Replicate last frame for right padding
        padded_features[T+2:] = features[-1:].repeat(2, 1)
        
        # Compute delta: (f[t+2] - f[t-2]) / 10
        delta = (padded_features[4:] - padded_features[:-4]) / 10.0
        
        return delta
    
    def _cmvn(self, features: torch.Tensor) -> torch.Tensor:
        """Cepstral Mean and Variance Normalization"""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)
    
    def _add_context(self, features: torch.Tensor) -> torch.Tensor:
        """Add left and right context frames using manual padding"""
        T, D = features.shape
        context_size = 2 * self.context_frames + 1  # 11 frames total
        
        # Manual replicate padding for context
        padded_features = torch.zeros(T + 2 * self.context_frames, D)
        padded_features[self.context_frames:T + self.context_frames] = features
        
        # Replicate boundary frames
        if T > 0:
            # Left padding
            padded_features[:self.context_frames] = features[0:1].repeat(self.context_frames, 1)
            # Right padding
            padded_features[T + self.context_frames:] = features[-1:].repeat(self.context_frames, 1)
        
        # Create spliced features
        spliced = []
        for i in range(T):
            context_frames = padded_features[i:i + context_size]  # [11, D]
            spliced.append(context_frames.flatten())  # [11*D]
        
        return torch.stack(spliced)  # [T, 11*D]
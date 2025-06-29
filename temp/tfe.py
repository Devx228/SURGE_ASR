# import sys
# import numpy as np
# from feature_extraction import FeatureExtractor  # Make sure your class is in this file or change import
# import matplotlib.pyplot as plt

# def main(audio_path):
#     extractor = FeatureExtractor()
#     features = extractor.extract_features(audio_path)

#     print(f"Extracted features shape: {features.shape}")  # Should be [T, 1320]
#     print(f"Feature mean (approx 0): {np.mean(features):.4f}")
#     print(f"Feature std (approx 1): {np.std(features):.4f}")

#     # Optional: visualize a small part of the features
#     plt.imshow(features[:100].T, aspect='auto', origin='lower')
#     plt.title("Extracted Features (first 100 frames)")
#     plt.xlabel("Time frames")
#     plt.ylabel("Feature dimensions")
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python test_feature_extractor.py <path_to_audio.wav>")
#         sys.exit(1)
    
#     audio_path = sys.argv[1]
#     main(audio_path)

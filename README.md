# Zero-Knowledge Faked Speech Detection

This project implements a neural network-based digital forensics tool for detecting artificially generated speech without requiring any reference or source audio — a concept known as **zero-knowledge detection**. The system is designed to support real-time applications such as speech-to-text, voice authentication, and secure voice-based interactions.

## Overview

Traditional fake speech detection systems often rely on comparing suspect audio with reference or original speech. In contrast, this project introduces a zero-knowledge detection method that identifies synthetic or manipulated audio using only intrinsic acoustic features.

Key features include:
- No dependence on reference audio
- Robust feature extraction (spectral and cepstral)
- Real-time inference
- Standalone operation suitable for deployment

## Key Technologies

- Python, NumPy
- PyTorch (neural network training and inference)
- Librosa / torchaudio (audio processing)
- Spectral and cepstral analysis (e.g., MFCCs)
- CNNs / RNNs (classification model)

## Applications
- Speech-to-text (S2T) verification
- Voice biometric systems
- Audio-based content moderation
- Real-time conference monitoring

## Reference
@article{alajmi2024faked,
  title={Faked Speech Detection with Zero Prior Knowledge},
  author={Al Ajmi, Sahar and Hayat, Khizar and Al Obaidi, Alaa M. and Kumar, Naresh and Najmuldeen, Munaf and Magnier, Baptiste},
  journal={Discover Applied Sciences},
  volume={6},
  number={288},
  year={2024},
  doi={10.1007/s42452-024-05893-3},
  archivePrefix={arXiv},
  eprint={2209.12573},
  primaryClass={cs.SD}
}

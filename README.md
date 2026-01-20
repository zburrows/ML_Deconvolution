# Reverb Deconvolution with Machine Learning

**Author**: Zachariah Burrows

**Date**: May 2025

## Overview

This project explores using machine learning to remove reverb from audio signals without access to the impulse response. Traditional deconvolution methods require both the reverberated (“wet”) signal and the impulse response of the room, which is often unavailable in real-world recordings.

Here, a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) is trained to predict a dry (anechoic) signal directly from a reverberated input, operating purely in the time domain.

The result is an experimental system that partially reduces reverberation characteristics, demonstrating the feasibility of impulse-response-free audio deconvolution using deep learning.

## Key Goals

- Remove reverb from audio without a known impulse response

- Train on raw audio in the time domain

- Provide a simple GUI for users to upload and process audio files

- Lay groundwork for future impulse response estimation or room-size prediction

## Approach

For this project, I implemented a PyTorch LSTM with 3 hidden layers, 256 neurons per layer, and a learning rate of 0.0001. LSTMs are well-suited for this task because reverberation manifests as temporal smearing, which can be learned by analyzing how audio evolves over time.

## Training Pipeline

Training data was taken from anechoic (dry) recordings from an anechoic chamber and artificially reverberated versions created via convolution in Ableton Live. 

Sample rate: 44.1 kHz

Audio format: Mono (for computational efficiency)

Segment length: 44,100 samples (1 second)

Epochs: 100

Loss: Computed between predicted dry signal and ground truth dry signal

Hardware: NVIDIA GPUs with CUDA acceleration

Each training step randomly selects aligned wet/dry segments and updates model weights via backpropagation.

## Testing & Inference

Segment length: 5,000 samples

Execution: CPU-based, parallelized across available cores

Post-processing: Segments are stitched together to form the final output

## GUI

A basic graphical interface allows users to upload audio files, run the deconvolution model, preview audio output, and save processed files locally.

To launch, clone the repository and run
```
python gui.py
```
## Results

The output audio shows reduced high-frequency content, slightly smoother waveform peaks, and less perceived “fullness” consistent with reverb reduction:

Issues observed:

- Audible clicks due to segment stitching

- Gain inconsistencies

- Limited effectiveness on recordings with heavy or natural reverb

The model performs best on reverb similar to what it was trained on, indicating dataset limitations.

## Limitations

- Clicking artifacts from non-overlapping segments

- Training data uses uniform artificial reverb, not diverse real-world acoustics

- No gain normalization during prediction

- Hyperparameters not fully optimized

## Future Work

- Implement overlapping windows with crossfades to eliminate clicks

- Expand training data with real-world room recordings

- Add gain normalization

- Perform hyperparameter tuning (e.g., grid search)

- Explore impulse response estimation or room-size prediction as downstream tasks

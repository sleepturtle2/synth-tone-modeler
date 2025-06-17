# Neural Synth Modeler

## Overview

Neural Synth Modeler is an end-to-end machine learning system that learns to model analog synthesizer behavior by predicting parameter settings from audio features. This project bridges digital signal processing and deep learning to create intelligent tools for sound designers and music producers.

Key capabilities:
- **Preset Parameter Prediction**: Reverse-engineers synthesizer settings from audio
- **Tone Matching**: Finds similar existing presets for any input sound
- **Intelligent Sound Design**: Suggests parameter adjustments to achieve desired timbres

## Technical Architecture

![System Architecture Diagram](docs/architecture.png)


## Model 
WTSv2(
  (mfcc_encoder): Sequential(
    (0): LayerNorm(normalized_shape=30)
    (1): GRU(input_size=30, hidden_size=512, batch_first=True)
    (2): Linear(in_features=512, out_features=16, bias=True)
  )
  (feature_combiner): ModuleList(
    (0): MLP(input_size=1, hidden_size=hidden_size, num_layers=3)  # For pitch
    (1): MLP(input_size=1, hidden_size=hidden_size, num_layers=3)  # For loudness
    (2): MLP(input_size=16, hidden_size=hidden_size, num_layers=3) # For MFCCs
  )
  (gru_combiner): GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
  (out_mlp): MLP(input_size=hidden_size*4, hidden_size=hidden_size, num_layers=3)
  (loudness_mlp): Sequential(
    (0): Linear(in_features=1, out_features=1, bias=True)
    (1): Sigmoid()
  )
  (proj_matrices): ModuleList(
    (0): Linear(in_features=hidden_size, out_features=n_harmonic+1, bias=True)  # Harmonic amplitudes
    (1): Linear(in_features=hidden_size, out_features=n_bands, bias=True)      # Noise bands
  )
  (wavetable_generator): Sequential(
    (0): Conv1d(in_channels=1, out_channels=num_wavetables, kernel_size=16, stride=16)
    (1): Tanh()
    (2): Conv1d(in_channels=num_wavetables, out_channels=num_wavetables, kernel_size=8, stride=8)
    (3): Tanh()
    (4): Linear(in_features=500, out_features=512, bias=True)  # Wavetable length
    (5): Tanh()
  )
  (attention_wt): Linear(in_features=512, out_features=1, bias=True)
  (smoothing_control): Sequential(
    (0): Linear(in_features=512, out_features=1, bias=True)
    (1): Sigmoid()
  )
  (adsr_extractors): ModuleList(
    (0): GRU(input_size=1, hidden_size=8, batch_first=True, bidirectional=True)  # Attack
    (1): GRU(input_size=1, hidden_size=8, batch_first=True, bidirectional=True)  # Decay
    (2): GRU(input_size=1, hidden_size=8, batch_first=True, bidirectional=True)  # Sustain
  )
  (adsr_heads): ModuleList(
    (0): Sequential(  # Attack
      (0): Linear(in_features=16, out_features=1, bias=True)
      (1): Sigmoid()
    )
    (1): Sequential(  # Decay
      (0): Linear(in_features=16, out_features=1, bias=True)
      (1): Sigmoid()
    )
    (2): Sequential(  # Sustain
      (0): Linear(in_features=16, out_features=1, bias=True)
      (1): Sigmoid()
    )
  )
  (adsr_conv): Conv1d(in_channels=1, out_channels=1, kernel_size=block_size, stride=block_size)
  (reverb): Reverb(length=sampling_rate, sampling_rate=sampling_rate)
  (wavetable_synth): WavetableSynthV2(sr=sampling_rate, duration_secs=duration_secs, block_size=block_size, enable_amplitude=True)
)
### Core Components

1. **Data Pipeline**
   - `.fxp` preset file parser
   - Audio renderer (4s mono @ 44.1kHz)
   - Feature extractor (Mel spectrograms, F0, loudness)
   - Data validation system (checksums, bounds checking)

2. **Machine Learning Models**
   - **Parameter Predictor**: CNN-BiLSTM hybrid (128-unit latent space)
     - Input: 128-band Mel spectrogram + pitch/loudness
     - Output: Normalized parameter values (oscillators, filters, FX)
   - **Tone Matcher**: Siamese network with triplet loss
   - **Parameter Recommender**: Variational Autoencoder

3. **Training Infrastructure**
   - Custom `ParameterLoss` with component weighting
   - Mixed precision training (FP16/FP32)
   - Apple Metal GPU acceleration
   - Experiment tracking (TensorBoard, MLflow)

## Key Results

| Metric               | Value       |
|----------------------|-------------|
| Parameter MAE        | 0.043       | 
| Tone Match Accuracy  | 89.2%       |
| Inference Speed      | 23ms/preset |

## Installation

```bash
conda create -n synthml python=3.9
conda activate synthml
pip install -r requirements.txt

# Install audio tools
brew install libsndfile  # macOS
sudo apt-get install libsndfile1  # Linux
```

## Usage

### 1. Data Preparation

```python
from neural_synth_modeler import DataGenerator

generator = DataGenerator(config_path="configs/data_config.toml")
generator.process_batch()  # Processes all .fxp files
```

### 2. Training

```bash
python train.py --config configs/training.toml
```

### 3. Inference

```python
from neural_synth_modeler import SynthModeler

model = SynthModeler.load("models/production/v3")
params = model.predict_parameters("guitar_sample.wav")
```

## Dataset

The included dataset contains:
- 1,560 professionally designed presets
- 390 parameter-audio pairs across 12 synth categories
- Balanced representation of:
  - Bass (23%)
  - Lead (19%)
  - Pad (17%)
  - FX (14%)
  - Other (27%)

## Development Roadmap

- [ ] Real-time parameter prediction VST plugin
- [ ] Few-shot learning for rare synth types
- [ ] User feedback integration (active learning)

## Contributing

We welcome contributions! Please see:
- [Style Guide](docs/STYLE.md)
- [Testing Protocol](docs/TESTING.md)
- [Roadmap](docs/ROADMAP.md)

## Citation

If you use this work in research, please cite:

```bibtex
@software{NeuralSynthModeler,
  author = {Your Name},
  title = {Neural Synth Modeler},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/neural-synth-modeler}}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
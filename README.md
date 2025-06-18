# Neural Synth Modeler

## Overview

Neural Synth Modeler is a synth preset inferencer system that learns to model one-shot audio behavior by predicting synth parameter settings from audio features. This project bridges digital signal processing and deep learning to create intelligent tools for sound designers and music producers.

Key capabilities:
- **Preset Parameter Prediction**: Reverse-engineers synthesizer settings from audio
- **Intelligent Sound Design**: Suggests parameter adjustments to achieve desired timbres

## Technical Architecture

![System Architecture Diagram](docs/architecture.png)


## Model 
```
Model architecture reference: 
https://github.com/magenta/ddsp
https://github.com/gudgud96/syntheon

model name - Wave Table Synthesizer v2 

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
```



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

TODO : 

Implementation notes: 
```
# Pseudocode for WTSv2 Model
# Inputs:
#   y: Tensor(batch_size, 64000) - Raw audio signal (e.g., 4s at 16000 Hz)
#   mfcc: Tensor(batch_size, num_frames, 30) - Mel-frequency cepstral coefficients
#   pitch: Tensor(batch_size, num_frames, 1) - Pitch estimates (e.g., 400 frames for 4s)
#   loudness: Tensor(batch_size, num_frames, 1) - Loudness estimates
#   times: Tensor(num_frames) - Timestamps for frames
#   onset_frames: Tensor(num_onsets) - Indices of note onsets
# Outputs:
#   signal: Tensor(batch_size, duration_secs*sampling_rate) - Intermediate signal (harmonic + noise)
#   adsr_params: Tuple(attack_secs, decay_secs, sustain_level) - ADSR parameters
#   final_signal: Tensor(batch_size, duration_secs*sampling_rate) - ADSR-shaped signal
#   attention_output: Tensor(batch_size, num_wavetables) - Wavetable attention scores
#   wavetables: Tensor(batch_size, num_wavetables, 512) - Generated wavetables
#   wavetables_old: Tensor or None - Unsmoothed wavetables (if smoothing applied)
#   smoothing_coeff: Tensor or None - Smoothing coefficients (if smoothing applied)

# Initialization
initialize WTSv2 with:
    hidden_size: Size of hidden layers
    n_harmonic: Number of harmonic components
    n_bands: Number of noise bands
    sampling_rate: Audio sampling rate (e.g., 16000 Hz)
    block_size: Synthesis block size
    mode: "wavetable" (default synthesis mode)
    duration_secs: Output duration (e.g., 3s)
    num_wavetables: Number of wavetables (e.g., 3)
    wavetable_smoothing: Boolean to enable Gaussian smoothing
    min_smoothing_sigma, max_smoothing_sigma: Smoothing range
    preload_wt: Boolean to infer wavetables from input audio
    is_round_secs: Boolean for ADSR timing
    enable_amplitude: Boolean for amplitude modulation
    device: "cuda" or "cpu"
    register buffers: sampling_rate, block_size
    initialize components:
        mfcc_encoder: LayerNorm(30), GRU(30->512), Linear(512->16)
        feature_combiner: MLPs for pitch(1->hidden_size), loudness(1->hidden_size), MFCCs(16->hidden_size)
        gru_combiner: GRU(3->hidden_size)
        out_mlp: MLP(hidden_size*4->hidden_size)
        loudness_mlp: Linear(1->1), Sigmoid
        proj_matrices: Linear(hidden_size->n_harmonic+1), Linear(hidden_size->n_bands)
        wavetable_generator: Conv1d(1->num_wavetables), Tanh, Conv1d, Tanh, Linear(500->512), Tanh
        attention_wt: Linear(512->1)
        smoothing_control: Linear(512->1), Sigmoid
        adsr_extractors: Bidirectional GRUs(1->8) for attack, decay, sustain
        adsr_heads: Linear(16->1), Sigmoid for attack, decay, sustain
        adsr_conv: Conv1d(1->1, kernel=block_size, stride=block_size)
        reverb: Reverb(length=sampling_rate, sampling_rate)
        wavetable_synth: WavetableSynthV2(sr, duration_secs, block_size, enable_amplitude)
        adsr_params: Parameters for attack_sec, decay_sec, sustain_level (initialized to 1)
        max_attack_secs, max_decay_secs: 2.0 seconds

# Forward Pass
function forward(y, mfcc, pitch, loudness, times, onset_frames):
    batch_size = y.shape[0]

    # Step 1: Process MFCCs
    # Input: mfcc (batch_size, num_frames, 30)
    mfcc = transpose(mfcc, dims=(1, 2))  # (batch_size, 30, num_frames)
    mfcc = layer_norm(mfcc)  # Normalize across 30 features
    mfcc, _ = gru_mfcc(mfcc)  # GRU: (batch_size, num_frames, 512)
    mfcc = mlp_mfcc(mfcc)  # Linear: (batch_size, num_frames, 16)
    mfcc = resize(mfcc, size=(duration_secs*100, 16))  # (batch_size, duration_secs*100, 16)

    # Step 2: Combine Features
    # Inputs: pitch (batch_size, num_frames, 1), loudness (batch_size, num_frames, 1), mfcc (batch_size, num_frames, 16)
    pitch_hidden = in_mlps[0](pitch)  # (batch_size, num_frames, hidden_size)
    loudness_hidden = in_mlps[1](loudness)  # (batch_size, num_frames, hidden_size)
    mfcc_hidden = in_mlps[2](mfcc)  # (batch_size, num_frames, hidden_size)
    hidden = concatenate([pitch_hidden, loudness_hidden, mfcc_hidden], dim=-1)  # (batch_size, num_frames, hidden_size*3)
    gru_output, _ = gru(hidden)  # (batch_size, num_frames, hidden_size)
    hidden = concatenate([gru_output, hidden], dim=-1)  # (batch_size, num_frames, hidden_size*4)
    hidden = out_mlp(hidden)  # (batch_size, num_frames, hidden_size)

    # Step 3: Compute Amplitude
    # Input: loudness (batch_size, num_frames, 1)
    total_amp = loudness_mlp(loudness)  # (batch_size, num_frames, 1), values in [0, 1]
    pitch_prev = pitch  # Save for ADSR
    pitch = upsample(pitch, block_size)  # (batch_size, num_samples, 1)
    total_amp = upsample(total_amp, block_size)  # (batch_size, num_samples, 1)

    # Step 4: Generate Wavetables
    if preload_wt:
        wavetables = []
        for i in range(batch_size):
            wt = infer_wavetables(y[i], pitch_prev[i])  # (512,)
            wavetables.append(wt)
        wavetables = stack(wavetables, dim=0).unsqueeze(1)  # (batch_size, 1, 512)
        check for NaN/infinite values in wavetables
    else:
        wavetables = wt1_conv1d(y.unsqueeze(1))  # (batch_size, num_wavetables, 512)

    # Step 5: Smooth Wavetables (Optional)
    if wavetable_smoothing:
        smoothing_coeff = smoothing_linear(wavetables)  # (batch_size, num_wavetables, 1)
        smoothing_coeff = smoothing_coeff.squeeze(1)  # (batch_size, num_wavetables)
        smoothing_coeff = smoothing_sigmoid(smoothing_coeff)  # (batch_size, num_wavetables), [0, 1]
        wavetables_old = wavetables
        wavetables = smoothing(wavetables, smoothing_coeff)  # Apply Gaussian smoothing
    else:
        wavetables_old = None
        smoothing_coeff = None

    # Step 6: Wavetable Attention
    attention_output = attention_wt1(wavetables).squeeze(-1)  # (batch_size, num_wavetables)
    attention_output = softmax(attention_output, dim=-1)  # (batch_size, num_wavetables)

    # Step 7: Harmonic Synthesis
    harmonic, attention_output = wavetable_synth(pitch, total_amp, wavetables, attention_output)
    # harmonic: (batch_size, duration_secs*sampling_rate, 1)

    # Step 8: Noise Synthesis
    noise_param = scale_function(proj_matrices[1](hidden) - 5)  # (batch_size, num_frames, n_bands)
    impulse = amp_to_impulse_response(noise_param, block_size)  # (batch_size, num_frames, block_size)
    noise = random_tensor(impulse.shape, range=[-1, 1]).to(device)  # (batch_size, num_frames, block_size)
    noise = fft_convolve(noise, impulse)  # (batch_size, num_frames, block_size)
    noise = reshape(noise, (batch_size, -1, 1))  # (batch_size, duration_secs*sampling_rate, 1)

    # Step 9: Combine Harmonic and Noise
    signal = harmonic + noise  # (batch_size, duration_secs*sampling_rate, 1)

    # Step 10: ADSR Envelope Extraction
    attack_output, hn_attack = attack_gru(loudness)  # (batch_size, num_frames, 8)
    hn_attack = concatenate([hn_attack[0], hn_attack[1]], dim=-1)  # (batch_size, num_frames, 16)
    decay_output, hn_decay = decay_gru(loudness)  # (batch_size, num_frames, 8)
    hn_decay = concatenate([hn_decay[0], hn_decay[1]], dim=-1)  # (batch_size, num_frames, 16)
    sustain_output, hn_sustain = sustain_gru(loudness)  # (batch_size, num_frames, 8)
    hn_sustain = concatenate([hn_sustain[0], hn_sustain[1]], dim=-1)  # (batch_size, num_frames, 16)

    attack_level = attack_sec_head(hn_attack).squeeze()  # (batch_size, num_frames)
    decay_level = decay_sec_head(hn_decay).squeeze()  # (batch_size, num_frames)
    sustain_level = sustain_level_head(hn_sustain).squeeze()  # (batch_size, num_frames)

    attack_secs = attack_level * max_attack_secs  # (batch_size, num_frames)
    decay_secs = decay_level * max_decay_secs  # (batch_size, num_frames)

    # Step 11: ADSR Envelope Generation
    amp_onsets = append(times[onset_frames], times[-1])  # (num_onsets + 1,)
    adsr = get_amp_shaper(shaper, amp_onsets, attack_secs, decay_secs, sustain_level)
    # adsr: (batch_size, num_frames)
    if adsr.shape[1] < pitch_prev.shape[1]:
        adsr = concatenate([adsr, adsr[:, -1].unsqueeze(-1)], dim=-1)
    else:
        adsr = adsr[:, :pitch_prev.shape[1]]
    store adsr in self.adsr
    adsr = adsr.unsqueeze(-1)  # (batch_size, num_frames, 1)
    adsr = upsample(adsr, block_size).squeeze(-1)  # (batch_size, duration_secs*sampling_rate)
    adsr = adsr[:, :signal.shape[1]]  # Truncate to signal length

    # Step 12: Apply ADSR Envelope
    final_signal = signal.squeeze() * adsr  # (batch_size, duration_secs*sampling_rate)

    # Step 13: Optional Reverb (Commented Out)
    # signal = reverb(signal)

    # Step 14: Return Outputs
    return signal, (attack_secs, decay_secs, sustain_level), final_signal, attention_output, wavetables, wavetables_old, smoothing_coeff

# Smoothing Function
function smoothing(wavetables, p):
    # Input: wavetables (batch_size, num_wavetables, wavetable_length), p (batch_size, num_wavetables)
    batch_size, wavetable_length = wavetables.shape[0], wavetables.shape[2]
    smoothed_wavetables = zeros(batch_size, wavetable_length).to(device)
    sigma = p * (max_smoothing_sigma - min_smoothing_sigma) + min_smoothing_sigma  # (batch_size, num_wavetables)
    sigma = sigma.unsqueeze(-1)  # (batch_size, num_wavetables, 1)
    kernel = arange(wavetable_length).to(device)  # (wavetable_length,)
    kernel = kernel.unsqueeze(0) - kernel.unsqueeze(-1)  # (wavetable_length, wavetable_length)
    kernel = exp(-kernel^2 / (2 * sigma^2))  # (batch_size, wavetable_length, wavetable_length)
    kernel = kernel / sum(kernel, dim=-1).unsqueeze(-1)  # Normalize
    smoothed_wavetables = batch_matrix_multiply(wavetables, kernel)  # (batch_size, num_wavetables, wavetable_length)
    return smoothed_wavetables

```
## Functionalities Explained

- MFCC Processing: Extracts spectral features from MFCCs, normalized and processed through GRU and linear layers to capture temporal patterns, resized to match output duration.
- Feature Combination: Integrates pitch, loudness, and MFCCs via MLPs and GRU, producing a unified hidden representation for synthesis control.
- Amplitude and Pitch Upsampling: Converts low-resolution pitch and loudness to high-resolution signals for synthesis.
- Wavetable Generation: Either infers wavetables from raw audio or learns them via Conv1d layers, (with optional Gaussian smoothing to reduce artifacts).
- Attention Mechanism: Weights wavetables based on their relevance, enhancing synthesis flexibility.
- Harmonic Synthesis: Uses WavetableSynthV2 to generate harmonic signals from pitch, amplitude, and wavetables.
- Noise Synthesis: Adds stochastic noise shaped by learned impulse responses, improving realism.
- ADSR Envelope: Extracts attack, decay, and sustain parameters from loudness, generates an envelope, and shapes the signal’s amplitude dynamics.
- Reverb (Optional): Enhances spatial realism (currently disabled).
- Output: Produces intermediate and final signals, ADSR parameters, and wavetable metadata for analysis or further processing.

## Key Results

| Metric               | Value       |
|----------------------|-------------|
| Parameter MAE        | 0.043       | 
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
import numpy as np
import soundfile as sf
from surgepy import createSurge

# Initialize synth
synth = createSurge(44100)
synth.loadPatch("Bowed Plucked Pipe.fxp")

# Get actual block size from Surge
block_size = synth.getBlockSize()  # Typically 32 or 64
sample_rate = 44100
note_duration = 2.0  # seconds

# Calculate total samples and blocks
total_samples = int(note_duration * sample_rate)
total_blocks = int(np.ceil(total_samples / block_size))

# Initialize audio buffer (shape: [total_samples, 2])
audio = np.zeros((total_blocks * block_size, 2), dtype=np.float32)

# Trigger note
synth.playNote(0, 60, 100)  # channel, midi_note, velocity

# Process audio
for i in range(total_blocks):
    synth.process()
    # Get output and transpose to [block_size, 2] shape
    block = synth.getOutput().T  # Transpose from [2, block_size] to [block_size, 2]
    start_idx = i * block_size
    end_idx = start_idx + block_size
    audio[start_idx:end_idx] = block

# Release note
synth.releaseNote(0, 60)

# Trim to exact duration and save
audio = audio[:int(note_duration * sample_rate)]
sf.write("output.wav", audio, sample_rate)
print(f"Rendered output.wav with {block_size}-sample blocks")
import warnings
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import crepe
from neural_synth_modeler.utils.logger import LoggingUtil
from typing import Dict, Optional

logger = LoggingUtil.setup_logger(__name__)

class AudioPreprocessor:
    def __init__(self, target_sr=16000, duration=4.0):
        self.target_sr = target_sr
        self.target_duration = duration
        self.target_samples = int(target_sr * duration)
        self.min_samples = 512  # Minimum for meaningful processing

    def _prepare_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Prepare audio with proper length and sample rate"""
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        if original_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        
        return self._ensure_length(audio)
    
    def _ensure_length(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio has correct length"""
        if len(audio) < self.min_samples:
            return np.zeros(self.target_samples)
        
        #Pad or truncate to ensure correct length of audio
        return audio[:self.target_samples] if len(audio) > self.target_samples else np.pad(audio, (0, max(0, self.target_samples - len(audio))))
    
    
    def process_audio(self, audio: np.ndarray, original_sr: int) -> dict:
        """Process raw audio into training-ready features"""
        try:
            # Convert to mono and resample if needed
            audio = self._prepare_audio(audio, original_sr)

            # Extract features
            return {
                'waveform': audio.astype(np.float32),
                'mel_spectrogram': self._extract_mel_spectrogram(audio),
                'f0': self._extract_pitch(audio),
                'loudness': self._extract_loudness(audio)
            }
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return self._create_fallback_output(audio)    

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram with safe settings"""
        n_fft = min(2048, len(audio))
        hop_length = max(1, n_fft // 4)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=128
            )
            return librosa.power_to_db(mel, ref=np.max)

    def _extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Pitch extraction with multiple fallbacks"""
        try:
            import crepe
            _, f0, _, _ = crepe.predict(
                audio,
                self.target_sr,
                model_capacity='tiny',
                viterbi=True
            )
            return f0
        except ImportError:
            return np.zeros(len(audio) // 512)  # Fallback array
        except Exception:
            return np.zeros(len(audio) // 512)

    def _extract_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Safe loudness calculation"""
        try:
            rms = librosa.feature.rms(
                y=audio,
                frame_length=min(2048, len(audio)),
                hop_length=max(1, len(audio) // 4)
            )
            return librosa.amplitude_to_db(rms, ref=np.max).squeeze()
        except Exception:
            return np.zeros(len(audio) // 512)
        
    def _create_fallback_output(self, audio: np.ndarray) -> Dict:
        """Create output when processing fails"""
        fallback_len = max(1, len(audio) // 512)
        return {
            'waveform': np.zeros_like(audio),
            'mel_spectrogram': np.zeros((128, fallback_len)),
            'f0': np.zeros(fallback_len),
            'loudness': np.zeros(fallback_len)
        }
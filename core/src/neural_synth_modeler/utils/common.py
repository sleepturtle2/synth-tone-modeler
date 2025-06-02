from pathlib import Path
import numpy as np
from typing import Dict, Literal

from pathlib import Path

def setup_dirs_for_training(output_root: Path) -> None:
    """Create directory structure for training pipeline using unified storage."""
    dirs = [
        output_root / "all",
        output_root / "train",
        output_root / "val",
        output_root / "test"
    ]
    for dir_path in dirs:
        for subdir in ["fxp", "audio", "params", "meta"]:
            (dir_path / subdir).mkdir(parents=True, exist_ok=True)



def normalize_audio(waveform: np.ndarray, config: Dict) -> np.ndarray:
    """Normalize audio using configuration parameters"""
    if not config.get("audio", {}).get("normalize", False):
        return waveform
        
    peak_level = float(config["audio"].get("peak_level", -3.0))
    max_sample = np.max(np.abs(waveform))
    if max_sample == 0:
        return waveform
        
    target_linear = 10 ** (peak_level / 20)
    gain = target_linear / max_sample
    return np.clip(waveform * gain, -1.0, 1.0)

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import traceback
import soundfile as sf
from tqdm import tqdm
import hashlib

from neural_synth_modeler.utils.logger import LoggingUtil
from neural_synth_modeler.utils.config_loader import get_config

logger = LoggingUtil.setup_logger(__name__)

class DataValidator:
    """Validation and quality assurance for training data"""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.config = get_config("data_config.toml")
        self.validation_report = []
        self.stats = {
            'total_presets': 0,
            'valid_presets': 0,
            'errors': {}
        }

    def validate_entire_dataset(self) -> Tuple[bool, Dict]:
        """Validate all presets in the unified 'all/' folder"""
        logger.info("Starting dataset validation in 'all/' directory...")

        try:
            all_dir = self.data_root / "all"
            if not all_dir.exists():
                raise FileNotFoundError("'all/' directory not found.")

            preset_files = list((all_dir / "fxp").glob("*.fxp"))
            self.stats['total_presets'] = len(preset_files)

            for fxp_path in tqdm(preset_files, desc="Validating presets"):
                base_name = fxp_path.stem
                try:
                    self._validate_single_preset(base_name)
                    self.stats['valid_presets'] += 1
                except Exception as e:
                    self._record_error(base_name, str(e))

            self._generate_reports()
            success = self.stats['valid_presets'] == self.stats['total_presets']
            logger.info(f"Validation complete. Success rate: "
                        f"{self.stats['valid_presets']/self.stats['total_presets']:.1%}")
            return success, self.stats

        except Exception as e:
            logger.error(f"Critical validation failure: {str(e)}")
            raise

    def _validate_single_preset(self, base_name: str) -> None:
        """Run validation checks for a single preset in 'all/' directory"""
        all_dir = self.data_root / "all"
        required_files = {
            'fxp': all_dir / "fxp" / f"{base_name}.fxp",
            'params': all_dir / "params" / f"{base_name}.json",
            'waveform': all_dir / "audio" / f"{base_name}_waveform.npy",
            'metadata': all_dir / "meta" / f"{base_name}.json"
        }

        for file_type, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {file_type} file: {path.name}")

        self._validate_parameters(required_files['params'])
        self._validate_waveform(required_files['waveform'])
        self._validate_metadata(required_files['metadata'])
        self._check_cross_file_consistency(
            required_files['params'],
            required_files['metadata'],
            required_files['fxp']
        )

    def _validate_parameters(self, param_path: Path) -> None:
        with open(param_path) as f:
            params = json.load(f)

        for param, value in params.items():
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"Parameter {param} out of bounds: {value}")

        required_params = self.config.get("required_parameters", [])
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

    def _validate_waveform(self, waveform_path: Path) -> None:
        waveform = np.load(waveform_path)
        sr = self.config["audio"]["sample_rate"]
        expected_samples = int(sr * self.config["audio"]["duration"])
        if len(waveform) != expected_samples:
            raise ValueError(f"Invalid waveform length: {len(waveform)} samples")

        if self.config["audio"].get("normalize", False):
            peak = np.max(np.abs(waveform))
            target_peak = 10 ** (self.config["audio"]["peak_level"] / 20)
            if not np.isclose(peak, target_peak, atol=0.01):
                raise ValueError(f"Normalization failed. Peak: {peak:.4f}")

    def _validate_metadata(self, metadata_path: Path) -> None:
        with open(metadata_path) as f:
            metadata = json.load(f)

        required_fields = ["render_date", "plugin_version"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing metadata field: {field}")

    def _check_cross_file_consistency(self, param_path: Path,
                                      metadata_path: Path,
                                      fxp_path: Path) -> None:
        with open(param_path) as f:
            params = json.load(f)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Optional checksum check
        stored_checksum = metadata.get("fxp_checksum")
        if stored_checksum:
            current_checksum = self._calculate_file_checksum(fxp_path)
            if current_checksum != stored_checksum:
                raise ValueError("FXP file checksum mismatch")

    def _calculate_file_checksum(self, file_path: Path) -> str:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _record_error(self, preset_name: str, error: str) -> None:
        error_type = error.split(":")[0]
        self.stats['errors'].setdefault(error_type, 0)
        self.stats['errors'][error_type] += 1

        self.validation_report.append({
            'preset': preset_name,
            'error': error,
            'error_type': error_type
        })
        logger.warning(f"Validation failed for {preset_name}: {error}")

    def _generate_reports(self) -> None:
        json_report = self.data_root / "validation_report.json"
        with open(json_report, 'w') as f:
            json.dump(self.validation_report, f, indent=2)

        csv_report = self.data_root / "validation_report.csv"
        with open(csv_report, 'w') as f:
            f.write("preset,error_type,error\n")
            for entry in self.validation_report:
                f.write(f"{entry['preset']},{entry['error_type']},{entry['error']}\n")

        logger.info(f"Validation reports saved to {json_report} and {csv_report}")

    def get_clean_dataset(self) -> List[Dict]:
        return [entry for entry in self.validation_report if not entry['error']]

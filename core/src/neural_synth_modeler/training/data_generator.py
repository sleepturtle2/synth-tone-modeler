import json
import concurrent.futures
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import traceback
import os
from shutil import copyfile
import re


from neural_synth_modeler.preset.preset_processor import PresetProcessor
from neural_synth_modeler.utils.config_loader import get_config
from neural_synth_modeler.utils.logger import LoggingUtil
from neural_synth_modeler.utils.common import (
    setup_dirs_for_training, 
    normalize_audio
)

logger = LoggingUtil.setup_logger(__name__)

class TrainingDataGenerator:
    """Batch processor for generating (FXP, parameters, audio) training triplets."""
    
    def __init__(self):
        self.main_config = get_config("config.toml")
        self.data_config = get_config("data_config.toml")
        self.failures = []  

        # Get absolute paths from config
        self.raw_presets_dir = Path(self.main_config["paths"]["preset_dir"]).resolve()
        self.output_root = Path(self.main_config["paths"]["output_dir"]).resolve()

        # Debug paths
        logger.info(f"Presets source: {self.raw_presets_dir}")
        logger.info(f"Output directory: {self.output_root}")

        if not self.raw_presets_dir.exists():
            raise FileNotFoundError(f"Preset directory not found: {self.raw_presets_dir}")

        
        setup_dirs_for_training(self.output_root)
        
        self.processor = PresetProcessor(
            preset_dir=str(self.raw_presets_dir),
            output_dir=str(self.output_root / "uncategorized")
        )



    def generate_batches(self) -> None:
        preset_paths = list(self.raw_presets_dir.glob("*.fxp"))
        if not preset_paths:
            logger.error(f"No presets found in directory {self.raw_presets_dir}!")
            return
            
        logger.info(f"Found {len(preset_paths)} presets to process")
        batch_size = self.data_config["batch"]["chunk_size"]
        logger.info(f"Processing {len(preset_paths)} presets in batches of {batch_size}")
        
        for batch_start in range(0, len(preset_paths), batch_size):
            batch = preset_paths[batch_start:batch_start + batch_size]
            
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.data_config["batch"]["workers"]
            ) as executor:
                futures = {
                    executor.submit(self._process_single_preset, path): path
                    for path in batch
                }
                
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {batch_start//batch_size + 1}"
                ):
                    result = future.result()
                    if result:
                        params, base_name = result
                        self._consolidate_preset_outputs(base_name)
            
            logger.info(f"Completed batch {batch_start//batch_size + 1}")
        self._clean_empty_dirs()
        logger.info("All batches processed successfully.")



    def _process_single_preset(self, fxp_path: Path) -> Optional[Tuple[Dict, str]]:
        try:
            result = self.processor.render_preset(
                str(fxp_path),
                note=60,
                duration=4.0
            )
            if result is None:
                raise ValueError("Render returned None")
            
            base_name = fxp_path.stem
            self._save_preset_data_after_norm(fxp_path, result)
            logger.info(f"Render result keys: {list(result.keys())}")
            if 'audio' not in result:
                logger.info("No audio data in render result")
            return result['parameters'], base_name
            
        except Exception as e:
            logger.error(f"Preset {fxp_path.name} failed: {str(e)}\n{traceback.format_exc()}")
            return None


    # def _calculate_complexity(self, params: Dict) -> str:
    #     """Wrapper for the common complexity calculator"""
    #     return calculate_preset_complexity(params, self.data_config["complexity_thresholds"])

    
        
    def _save_preset_data_after_norm(self, fxp_path: Path, result: Dict) -> None:
        """Save all preset components with robust error handling and no redundancy.
        
        Args:
            fxp_path: Path to source FXP file
            result: Dictionary containing:
                - audio: dict with waveform and optional features
                - parameters: dict of preset parameters
                - metadata: dict of preset metadata
                
        Raises:
            ValueError: If required audio waveform is missing
            IOError: If file operations fail
        """
        raw_name = fxp_path.stem
        base_name = re.sub(r'[^\w\-\.]', '_', raw_name)
        logger.debug(f"Sanitized '{raw_name}' → '{base_name}'")

        save_dir = self.output_root / "uncategorized"
        
        try:
            logger.debug(f"Processing preset: {base_name}")
            
            # Validate required data exists
            if 'audio' not in result or 'waveform' not in result['audio']:
                raise ValueError("Missing required audio waveform in render result")
            
            # Normalize just the waveform
            result['audio']['waveform'] = self._normalize_audio(result['audio']['waveform'])
            
            # Create all required directories
            dirs_created = []
            for subdir in ["params", "audio", "meta", "fxp"]:
                (save_dir / subdir).mkdir(parents=True, exist_ok=True)
            dirs_created.append(str(save_dir/subdir))
            logger.info(f"Created directories: {', '.join(dirs_created)}")
            
            # Define all possible audio features (required + optional)
            audio_features_to_save = {
                'waveform': f"{base_name}_waveform.npy",  
                'mel': f"{base_name}_mel.npy",            
                'f0': f"{base_name}_f0.npy",              
                'loudness': f"{base_name}_loudness.npy"   
            }

            # Save each feature exactly once
            saved_features = []
            for feature, filename in audio_features_to_save.items():
                if feature in result['audio']:
                    np.save(save_dir / "audio" / filename, result['audio'][feature])
                    saved_features.append(feature)
                elif feature == 'waveform':
                    logger.error("Missing required waveform data")
                    raise ValueError("Waveform data is required")
                else:
                    logger.debug(f"Optional feature {feature} not available")

            # Log summary of saved features
            if saved_features:
                logger.info(f"Saved audio features: {', '.join(saved_features)}")
            
            # Save parameters
            raw_params = {k: v['value'] for k, v in result['parameters'].items() if isinstance(v, dict) and 'value' in v}
            param_path = save_dir / "params" / f"{base_name}.json"
            with open(param_path, 'w') as f:
                json.dump(raw_params, f, indent=2)
            logger.info(f"Saved parameters to {param_path.relative_to(self.output_root)}")
            
            # Save metadata
            meta_path = save_dir / "meta" / f"{base_name}.json"
            with open(meta_path, 'w') as f:
                json.dump(result['metadata'], f, indent=2)
            logger.info(f"Saved metadata to {meta_path}")
            
            # Copy original FXP
            fxp_dest = save_dir / "fxp" / fxp_path.name
            fxp_dest.write_bytes(fxp_path.read_bytes())
            logger.debug(f"Preserved original FXP at {fxp_dest}")
            
        except Exception as e:
            logger.error(f"Failed to save preset {base_name}: {str(e)}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            raise


    def _consolidate_preset_outputs(self, base_name: str) -> None:
        """Consolidate all output files (audio, params, meta, fxp) into the 'all' directory."""
        try:
            moved_files = []
            for file_type in ["fxp", "audio", "params", "meta"]:
                src_file = self.output_root / "uncategorized" / file_type / f"{base_name}.{ 'json' if file_type != 'audio' and file_type != 'fxp' else 'npy' if file_type == 'audio' else 'fxp' }"
                dest_dir = self.output_root / "all" / file_type
                dest_dir.mkdir(parents=True, exist_ok=True)

                # If multiple files per type (e.g., _f0.npy, _mel.npy), move all that match
                for matching in src_file.parent.glob(f"{base_name}*"):
                    dest_path = dest_dir / matching.name
                    matching.replace(dest_path)
                    moved_files.append(str(dest_path.relative_to(self.output_root)))

            logger.debug(f"Moved {len(moved_files)} files to {self.output_root / 'all'}")
        except Exception as e:
            logger.error(f"Failed to consolidate files for {base_name}: {str(e)}")
            self.failures.append((base_name, f"Consolidation failed: {str(e)}"))

    
    def _clean_empty_dirs(self):
        """Summarized directory cleaning that ignores hidden files"""
        removed = []
        for root, dirs, _ in os.walk(self.output_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                # Check if directory is truly empty (ignoring hidden files)
                is_empty = not any(
                    item.name.startswith(".") or item.name.startswith("_")
                    for item in dir_path.iterdir()
                )
                if is_empty:
                    try:
                        dir_path.rmdir()
                        removed.append(str(dir_path.relative_to(self.output_root)))
                    except OSError as e:
                        logger.debug(f"Couldn't remove {dir_path}: {str(e)}")
        
        if removed:
            logger.info(f"Removed {len(removed)} empty directories")
        else:
            logger.debug("No empty directories to remove")
    
    
    def _normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Wrapper for the common normalize_audio function"""
        return normalize_audio(waveform, self.data_config)

    

    def create_splits(self, ratios: tuple = (0.7, 0.15, 0.15)):
        """Create dataset splits from 'all' directory based on preset files."""
        logger.info("Creating dataset splits from unified 'all' directory...")
        np.random.seed(self.data_config.get("random_seed", 42))
        
        fxp_dir = self.output_root / "all" / "fxp"
        if not fxp_dir.exists():
            logger.error(f"FXP directory not found: {fxp_dir}")
            return
        
        presets = list(fxp_dir.glob("*.fxp"))
        if not presets:
            logger.warning("No presets found for splitting.")
            return

        np.random.shuffle(presets)
        total = len(presets)
        
        train_end = int(total * ratios[0])
        val_end = train_end + int(total * ratios[1])
        
        splits = {
            "train": presets[:train_end],
            "val": presets[train_end:val_end],
            "test": presets[val_end:]
        }

        for split_name, split_presets in splits.items():
            self._copy_split_files(split_name, split_presets)


    def _copy_split_files(self, split_name: str, presets: list):
        """Copy all associated preset files into split folders."""
        for preset_path in presets:
            base_name = preset_path.stem
            for file_type in ["fxp", "audio", "params", "meta"]:
                src_dir = self.output_root / "all" / file_type
                dest_dir = self.output_root / split_name / file_type
                dest_dir.mkdir(parents=True, exist_ok=True)

                for src_file in src_dir.glob(f"{base_name}*"):
                    dest_file = dest_dir / src_file.name
                    copyfile(src_file, dest_file)


    def analyze_failures(self):
        """Generate detailed failure analysis report"""
        if not self.failures:
            logger.info("No processing failures detected")
            return
            
        report_path = self.output_root / "processing_failures.csv"
        with open(report_path, 'w') as f:
            f.write("preset_name,error_type,traceback\n")
            for preset, error in self.failures:
                f.write(f"{preset},{error}\n")
        logger.info(f"Failure analysis report saved to {report_path}")

if __name__ == "__main__":
    # generator = TrainingDataGenerator()
    # generator.generate_batches()
    generator = TrainingDataGenerator()

    # 🔧 Path to a single .fxp file for testing
    test_fxp_file = next(generator.raw_presets_dir.glob("*.fxp"), None)
    
    if not test_fxp_file:
        logger.error("No FXP file found to test.")
    else:
        logger.info(f"Testing single preset processing for: {test_fxp_file.name}")
        result = generator._process_single_preset(test_fxp_file)
        
        if result:
            params, base_name = result
            logger.info(f"✅ Test preset processed: {base_name}")

            # Check if key output files exist
            out_dir = generator.output_root / "uncategorized"
            param_file = out_dir / "params" / f"{base_name}.json"
            meta_file = out_dir / "meta" / f"{base_name}.json"
            fxp_file = out_dir / "fxp" / f"{test_fxp_file.name}"
            audio_dir = out_dir / "audio"

            issues = []
            if not param_file.exists():
                issues.append("❌ Missing parameter JSON")
            if not meta_file.exists():
                issues.append("❌ Missing metadata JSON")
            if not fxp_file.exists():
                issues.append("❌ Missing copied FXP file")

            waveform_exists = any((audio_dir / f"{base_name}_waveform.npy").exists() for _ in range(1))
            if not waveform_exists:
                issues.append("❌ Missing waveform .npy file")

            if not issues:
                logger.info("✅ All expected output files were generated successfully.")
            else:
                for issue in issues:
                    logger.error(issue)
        else:
            logger.error("❌ Preset processing failed.")
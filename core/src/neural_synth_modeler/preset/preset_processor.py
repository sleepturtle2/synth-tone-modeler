from pathlib import Path
import soundfile as sf
from neural_synth_modeler.preset.fxp_modifier import export_modified_patch
from neural_synth_modeler.audio.audio_renderer import SurgeRenderer
from neural_synth_modeler.audio.audio_preprocessor import AudioPreprocessor
import traceback
from typing import Dict, Optional
from neural_synth_modeler.utils.logger import LoggingUtil

logger = LoggingUtil.setup_logger(__name__)

class PresetProcessor:
    def __init__(self, preset_dir, output_dir):
        self.preset_dir = Path(preset_dir)
        self.output_dir = Path(output_dir)
        self.renderer = SurgeRenderer()
        self.preprocessor = AudioPreprocessor()
        
        # Create directory structure
        (self.output_dir/"modified").mkdir(parents=True, exist_ok=True)
        (self.output_dir/"audio").mkdir(parents=True, exist_ok=True)

    def process_all_presets(self):
        """Process all presets in directory"""
        for fxp_path in self.preset_dir.glob("*.fxp"):
            try:
                # Modify preset
                modified_fxp = self.modify_preset(fxp_path)
                
                # Render audio
                self.render_preset(modified_fxp)
                
            except Exception as e:
                print(f"❌ Failed to process {fxp_path.name}: {e}")

    def modify_preset(self, preset_path):
        """Apply modifications to a single preset"""
        modified_path = self.output_dir/"modified"/f"MOD_{preset_path.name}"
        
        # Define your parameter modifications here
        test_values = {
            'A Volume': 0.7,
            'Global Volume': 0.8,
            # Add more parameter changes as needed
        }
        
        return export_modified_patch(
            base_fxp=preset_path,
            new_fxp=modified_path,
            param_values=test_values
        )

    def render_preset(self, fxp_path: str, output_path: str = None, note: int = 60, duration: float = 4.0) -> dict:
        """Render a preset to audio file with synchronized parameter extraction.
        
        Args:
            fxp_path: Path to the FXP preset file
            output_path: Optional custom output path for WAV file
            note: MIDI note number to render (default: 60 - middle C)
            duration: Duration in seconds (default: 4.0)
            
        Returns:
            Dictionary with parameters and audio features
        """
        try:
            # Set default output path if not provided
            if not output_path:
                output_path = str(self.output_dir/"audio"/f"{Path(fxp_path).stem}.wav")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
            # Load the preset fresh for each render to ensure clean state
            self.renderer.load_preset(fxp_path)
            
            # Get parameters immediately after loading
            params = self.renderer.get_parameters()
            
            # Render the audio
            audio = self.renderer.render_note(note=note, duration=duration)
            
            # Process audio features
            audio_features = self.preprocessor.process_audio(
                audio, 
                self.renderer.sample_rate
            )
            # Write the audio file
            sf.write(output_path, audio, self.renderer.sample_rate)
            
            print(f"✅ Rendered {Path(fxp_path).name} to {output_path}")
            # Return complete dataset
            return {
                'parameters': params,
                'audio': {
                    'raw_path': output_path,
                    'waveform': audio_features['waveform'],
                    'mel': audio_features['mel_spectrogram'],
                    'f0': audio_features['f0'],
                    'loudness': audio_features['loudness']
                },
                'metadata': {
                    'source': str(fxp_path),
                    'preset_name': Path(fxp_path).stem,
                    'output_name': Path(output_path).stem,
                    'note': note,
                    'duration': duration,
                    'sample_rate': self.preprocessor.target_sr,
                    'render_sample_rate': self.renderer.sample_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to render {Path(fxp_path).name}: {str(e)}\n{traceback.format_exc()}")
            return None

    def list_preset_parameters(self, preset_path):
        """Debugging function to list parameters"""
        self.renderer.load_preset(preset_path)
        params = self.renderer.get_parameters()
        print("\n".join(params.keys()))
import os
import soundfile as sf
import numpy as np
from neural_synth_modeler.surge import createSurge
import json
from neural_synth_modeler.utils.surge_constants import (
        cg_GLOBAL, cg_OSC, cg_MIX, cg_FILTER, cg_ENV, cg_LFO, cg_FX
    )
class SurgeRenderer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.synth = createSurge(sample_rate)
        self._parameter_ids = {}  # Cache for parameter name->ID mapping
        
        # Temporary debug print
        # print("\n=== Surge Synthesizer Inspection ===")
        # print("Available attributes/methods:")
        # print(dir(self.synth))
        # if hasattr(self.synth, 'parameters'):
        #     print("\nFirst parameter example:", self.synth.parameters[0])
        
    def load_preset(self, preset_path: str):
        """Load an FXP preset into surge instance"""
        if not os.path.exists(preset_path):
            raise FileNotFoundError(f"Preset {preset_path} not found!")
        self.synth.loadPatch(str(preset_path))
        return self.synth
    
    def get_parameters(self) -> dict:
        """Extract parameters using control groups for better compatibility."""
        control_groups = [
            cg_GLOBAL, cg_OSC, cg_MIX,
            cg_FILTER, cg_ENV, cg_LFO, cg_FX
        ]

        param_data = {}

        for cg in control_groups:
            try:
                control_group = self.synth.getControlGroup(cg)
                entries = getattr(control_group, "getEntries", lambda: [])()

                for entry_idx, entry in enumerate(entries):
                    try:
                        params = getattr(entry, "getParams", lambda: [])()

                        for param_idx, param in enumerate(params):
                            try:
                                name = param.getName() if hasattr(param, "getName") else f"UnnamedParam_{param_idx}"
                                value = self.synth.getParamVal(param)
                                min_val = self.synth.getParamMin(param)
                                max_val = self.synth.getParamMax(param)
                                default_val = self.synth.getParamDef(param)
                                value_type = self.synth.getParamValType(param)
                                display = self.synth.getParamDisplay(param)

                                param_data[name] = {
                                    "value": value,
                                    "min": min_val,
                                    "max": max_val,
                                    "default": default_val,
                                    "type": value_type,
                                    "display": display
                                }
                                print(f"Extracted {len(param_data)} parameters.")

                            except Exception as e:
                                print(f"⚠️ Failed to read parameter at entry {entry_idx}, param {param_idx}: {e}")
                    except Exception as e:
                        print(f"⚠️ Failed to process entry {entry_idx} in control group {cg}: {e}")

            except Exception as e:
                print(f"⚠️ Failed to load control group {cg}: {e}")

        return param_data

    
    def set_parameters(self, parameters: dict):
        """Set parameters using cached IDs"""
        for name, value in parameters.items():
            if name in self._parameter_ids:
                param_id = self._parameter_ids[name]
                self.synth.setParamVal(param_id, value)
    
    def render_note(self, note: int = 60, duration: float = 2.0) -> np.ndarray:
        """Render a single note to audio buffer"""
        block_size = self.synth.getBlockSize()
        total_blocks = int(duration * self.sample_rate / block_size)
        audio = np.zeros((total_blocks * block_size, 2), dtype=np.float32)
        
        self.synth.playNote(0, note, 100)  # channel, midi_note, velocity
        
        try:
            for i in range(total_blocks):
                self.synth.process()
                block = self.synth.getOutput().T
                start = i * block_size
                end = start + block_size
                audio[start:end] = block
                
            return audio[:int(duration * self.sample_rate)]
        finally:
            self.synth.allNotesOff()
def inspect_surge_methods():
    synth = createSurge(44100)
    
    print("\n=== Proper Method Inspection ===")
    
    # First find how to get parameter objects
    print("\nLooking for parameter access methods...")
    print("All attributes:", [attr for attr in dir(synth) if 'param' in attr.lower()])
    
    # Try to find parameter enumeration
    if hasattr(synth, 'getAllParams'):
        params = synth.getAllParams()
        print(f"\nFound {len(params)} parameters via getAllParams()")
        for i, param in enumerate(params[:3]):  # Print first 3 as examples
            print(f"Param {i}: {param}")
            print(f"  Name: {param.name if hasattr(param, 'name') else 'N/A'}")
            print(f"  Value: {param.value if hasattr(param, 'value') else 'N/A'}")
    else:
        print("\nNo direct parameter enumeration found")
    
    # Try alternative access patterns
    print("\nTrying alternative access methods...")
    try:
        first_param = synth.osc1_waveform  # Example common parameter
        print(f"Direct attribute access: osc1_waveform = {first_param}")
    except AttributeError:
        print("Direct attribute access failed")


def render_fxp(fxp_path, output_wav, note=60, duration=2.0, sr=44100):
    """Compatibility function matching your original format"""
    renderer = SurgeRenderer(sr)
    renderer.load_preset(fxp_path)
    audio = renderer.render_note(note, duration)
    sf.write(str(output_wav), audio, sr)


if __name__ == "__main__":
    import sys
    #inspect_surge_methods()
    test_fxp_path = "/Users/sayantanm/code/NeuralSynthModeler/core/src/neural_synth_modeler/data/presets/Pulsii.fxp"

    if not os.path.exists(test_fxp_path):
        print(f"❌ Preset file not found: {test_fxp_path}")
        sys.exit(1)

    try:
        renderer = SurgeRenderer(sample_rate=44100)
        print("✅ Renderer initialized.")

        loaded = renderer.load_preset(test_fxp_path)
        if not loaded:
            print(f"❌ Failed to load preset: {test_fxp_path}")
            sys.exit(1)

        print("🔍 Extracting parameters...")
        params = renderer.get_parameters()

        if not params:
            print("❌ No parameters returned.")
        else:
            print("✅ Parameters extracted successfully.")

            def sanitize(obj):
                if isinstance(obj, dict):
                    return {k: sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize(v) for v in obj]
                elif isinstance(obj, (str, int, float, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)

            print(json.dumps(sanitize(params), indent=2))

    except Exception as e:
        print(f"🔥 Exception in test: {str(e)}")

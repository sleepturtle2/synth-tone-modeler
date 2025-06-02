from pathlib import Path
from neural_synth_modeler.surge import createSurge

def export_modified_patch(base_fxp, new_fxp, param_values: dict, sample_rate=44100):
    # # Debug section - print all available methods and their signatures
    # print("\n=== Available Methods ===")
    # for method_name in dir(synth):
    #     # Skip private methods
    #     if not method_name.startswith('_'):
    #         method = getattr(synth, method_name)
    #         if callable(method):
    #             try:
    #                 # Try to get signature (Python 3.3+)
    #                 from inspect import signature
    #                 sig = signature(method)
    #                 print(f"{method_name}{sig}")
    #             except (ImportError, ValueError):
    #                 # Fallback for older Python versions
    #                 print(f"{method_name}() - (signature not available)")
    
    # print("\n=== Method Docstrings ===")
    # for method_name in dir(synth):
    #     if not method_name.startswith('_'):
    #         method = getattr(synth, method_name)
    #         if callable(method) and method.__doc__:
    #             print(f"{method_name}: {method.__doc__.strip()}")



    # Commented code for listing all surge parameters
    # patch = synth.getPatch()
    
    # # Build parameter name to object mapping
    # param_name_to_obj = {}
    
    # def collect_params(section, prefix=""):
    #     for key, value in section.items():
    #         if isinstance(value, dict):
    #             collect_params(value, prefix=f"{prefix}{key}.")
    #         else:
    #             full_name = f"{prefix}{key}"
    #             param_name_to_obj[full_name] = value

    # collect_params(patch)
    # print("Parameter name to object mapping:")
    # for name, obj in param_name_to_obj.items():
    #     print(f"{name}: {obj}")
    synth = createSurge(sample_rate)
    try:
        if not synth.loadPatch(str(base_fxp)):
            print(f"Failed to load patch: {base_fxp}")
    except Exception as e:
        print("\n Error loading patch:")
        print(f"Error: {str(e)}")
        raise  

    # Get the complete patch structure
    patch = synth.getPatch()
    
    # Build a mapping of display names to parameter objects
    param_map = {}
    
    # Scene parameters (A and B)
    for scene_idx, scene in enumerate(patch['scene']):
        scene_prefix = 'A' if scene_idx == 0 else 'B'
        
        # Oscillators
        for osc_idx, osc in enumerate(scene['osc'], start=1):
            param_map[f"{scene_prefix} Osc {osc_idx} Type"] = osc['type']
            param_map[f"{scene_prefix} Osc {osc_idx} Pitch"] = osc['pitch']
            param_map[f"{scene_prefix} Osc {osc_idx} Octave"] = osc['octave']
            # Add other osc parameters similarly...
            
        # Filters
        for filter_idx, filt in enumerate(scene['filterunit'], start=1):
            param_map[f"{scene_prefix} Filter {filter_idx} Type"] = filt['type']
            param_map[f"{scene_prefix} Filter {filter_idx} Cutoff"] = filt['cutoff']
            # Add other filter parameters...
            
        # Global scene controls
        param_map[f"{scene_prefix} Volume"] = scene['volume']
        param_map[f"{scene_prefix} Pan"] = scene['pan']
        # Add other scene globals...
    
    # FX parameters
    for fx_idx, fx in enumerate(patch['fx'], start=1):
        fx_slot = ['A1', 'A2', 'B1', 'B2', 'S1', 'S2', 'G1', 'G2', 
                  'A3', 'A4', 'B3', 'B4', 'S3', 'S4', 'G3', 'G4'][fx_idx-1]
        param_map[f"FX {fx_slot} FX Type"] = fx['type']
        for param_idx, param in enumerate(fx['p'], start=1):
            param_map[f"FX {fx_slot} Param {param_idx}"] = param
    
    # Global parameters
    param_map['Active Scene'] = patch['scene_active']
    param_map['Scene Mode'] = patch['scenemode']
    param_map['Global Volume'] = patch['volume']
    param_map['Split Point'] = patch['splitpoint']
    # Add other globals...

    # Set parameters
    for param_name, target_value in param_values.items():
        if param_name not in param_map:
            print(f"⚠️ Parameter '{param_name}' not found. Available parameters:")
            print("\n".join(sorted(param_map.keys())))
            continue
            
        param_obj = param_map[param_name]
        try:
            current_val = synth.getParamVal(param_obj)
            print(f"Setting {param_name} from {current_val:.4f} to {target_value}")
            synth.setParamVal(param_obj, float(target_value))
        except Exception as e:
            print(f"❌ Failed to set {param_name}: {e}")
    try:
        synth.savePatch(str(new_fxp))
    except Exception as e:
        print("\n❌ Error saving patch:")
        print(f"Error: {str(e)}")
        raise

    print(f"✅ Patch exported to: {new_fxp}")
    return new_fxp

def get_fxp_param_names(fxp_path: str) -> list[str]:
    from xml.etree import ElementTree as ET
    with open(fxp_path, 'rb') as f:
        data = f.read()
    xml_start = data.find(b'<?xml')
    xml_end = data.find(b'</patch>') + len(b'</patch>')
    xml_data = data[xml_start:xml_end].decode('utf-8')
    root = ET.fromstring(xml_data)
    return [param.tag for param in root.find('parameters')]



if __name__ == "__main__":
    base_fxp_path = "data/presets/Bowed Plucked Pipe.fxp"
    output_fxp_path = "data/modified/Bowed Plucked Pipe_MOD.fxp"
    output_fxp_path.parent.mkdir(parents=True, exist_ok=True)
    test_values = {
    'A Volume': 0.7,  # Scene A volume
    'B Osc 1 Type': 0.5,  # Scene B Oscillator 1 type
    'FX A1 FX Type': 0.2,  # FX slot A1 type
    'Global Volume': 0.8  # Master volume
}

    try:
        export_modified_patch(
            base_fxp=str(base_fxp_path),
            new_fxp=str(output_fxp_path),
            param_values=test_values,
            sample_rate=44100
        )
    except Exception as e:
        print(f"❌ Error: {e}")

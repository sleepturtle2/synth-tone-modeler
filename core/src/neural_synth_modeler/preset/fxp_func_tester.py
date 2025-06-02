import os
import json
from neural_synth_modeler.surge import createSurge

def extract_all_params_via_control_groups(synth):
    from neural_synth_modeler.utils.surge_constants import (
        cg_GLOBAL, cg_OSC, cg_MIX, cg_FILTER, cg_ENV, cg_LFO, cg_FX
    )

    control_groups = [
        cg_GLOBAL, cg_OSC, cg_MIX,
        cg_FILTER, cg_ENV, cg_LFO, cg_FX
    ]

    param_data = {}

    for cg in control_groups:
        try:
            control_group = synth.getControlGroup(cg)
            entries = getattr(control_group, "getEntries", lambda: [])()

            for entry_idx, entry in enumerate(entries):
                try:
                    # entry is of type SurgeControlGroupEntry
                    # entry.getParams() returns list of SurgePyNamedParam
                    params = getattr(entry, "getParams", lambda: [])()

                    for param_idx, param in enumerate(params):
                        try:
                            name = param.getName() if hasattr(param, "getName") else f"UnnamedParam_{param_idx}"
                            value = synth.getParamVal(param)
                            min_val = synth.getParamMin(param)
                            max_val = synth.getParamMax(param)
                            default_val = synth.getParamDef(param)
                            value_type = synth.getParamValType(param)
                            display = synth.getParamDisplay(param)

                            param_data[name] = {
                                "value": value,
                                "min": min_val,
                                "max": max_val,
                                "default": default_val,
                                "type": value_type,
                                "display": display
                            }

                        except Exception as e:
                            print(f"⚠️ Failed to read parameter at entry {entry_idx}, param {param_idx}: {e}")
                except Exception as e:
                    print(f"⚠️ Failed to process entry {entry_idx} in control group {cg}: {e}")

        except Exception as e:
            print(f"⚠️ Failed to load control group {cg}: {e}")

    return param_data


def save_json(output_path, data):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    fxp_path = "neural_synth_modeler/data/training_data/simple/fxp/1804.fxp"
    output_path = "neural_synth_modeler/data/params/1804_full_param_metadata.json"

    if not os.path.exists(fxp_path):
        print(f"❌ FXP not found: {fxp_path}")
        exit(1)

    synth = createSurge(44100.0)
    print("✅ Synth created")

    if not synth.loadPatch(fxp_path):
        print(f"❌ Could not load patch: {fxp_path}")
        exit(1)

    print("🔍 Extracting parameters...")
    param_data = extract_all_params_via_control_groups(synth)
    save_json(output_path, param_data)
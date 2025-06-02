import shutil
import neural_synth_modeler.fxp_modification_test as fxp_mod

shutil.copy("data/presets/Bowed Plucked Pipe.fxp", "data/modified/test_copy.fxp")
tester = fxp_mod.FXPTester()
tester.synth.loadPatch("data/modified/test_copy.fxp")

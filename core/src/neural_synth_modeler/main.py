import argparse
from pathlib import Path
from neural_synth_modeler.preset.preset_processor import PresetProcessor
from neural_synth_modeler.training.data_generator import TrainingDataGenerator

def process_presets(args):
    processor = PresetProcessor(
        preset_dir=args.preset_dir,
        output_dir=args.output_dir
    )
    
    if args.operation == "process-all":
        processor.process_all_presets()
    elif args.operation == "list-params":
        processor.list_preset_parameters(Path(args.preset_path))
    elif args.operation == "process-single":
        processor.modify_preset(Path(args.preset_path))
        processor.render_preset(Path(args.preset_path))

def training_data_pipeline(args):
    generator = TrainingDataGenerator()
    
    if args.all:
        generator.generate_batches()
        generator.create_splits()
        generator.analyze_failures()
    else:
        if args.generate:
            generator.generate_batches()
        if args.split:
            generator.create_splits()
        if args.validate:
            generator.validate_data_integrity()
        if args.analyze:
            generator.analyze_failures()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Synth Modeler Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Preset processing commands
    preset_parser = subparsers.add_parser("process-presets", 
                                        help="Process individual presets")
    preset_parser.add_argument("--preset-dir", default="data/presets",
                             help="Input directory for presets")
    preset_parser.add_argument("--output-dir", default="data/processed",
                             help="Output directory for processed presets")
    preset_parser.add_argument("operation", choices=[
        "process-all", 
        "list-params",
        "process-single"
    ], help="Operation to perform")
    preset_parser.add_argument("--preset-path", 
                             help="Path to specific preset for single operations")
    
    # Training data pipeline commands
    data_parser = subparsers.add_parser("generate-data", 
                                      help="Generate training dataset")
    data_group = data_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--all", action="store_true",
                         help="Run full pipeline (generate+split+analyze)")
    data_group.add_argument("--generate", action="store_true",
                         help="Generate raw training data")
    data_group.add_argument("--split", action="store_true",
                         help="Create dataset splits")
    data_group.add_argument("--validate", action="store_true",
                         help="Validate data integrity")
    data_group.add_argument("--analyze", action="store_true",
                         help="Generate failure analysis report")
    
    args = parser.parse_args()
    
    if args.command == "process-presets":
        process_presets(args)
    elif args.command == "generate-data":
        training_data_pipeline(args)
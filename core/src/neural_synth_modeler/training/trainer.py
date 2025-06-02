import argparse
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any
from neural_synth_modeler.utils import logger, config_loader
from neural_synth_modeler.model import create_model  

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Configure hardware
        self._setup_hardware()
        
    def _setup_hardware(self):
        """Configure for M1/M4 GPU acceleration"""
        tf.config.set_soft_device_placement(True)
        if self.config['training'].get('mixed_precision', False):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def load_data(self):
        """Load processed training data"""
        data_root = Path(self.config['data']['training_data_root'])
        
        self.train_dataset = self._create_dataset(
            data_root / self.config['data']['train_split']
        )
        
        self.val_dataset = self._create_dataset(
            data_root / self.config['data']['val_split']
        )

    def _create_dataset(self, data_dir: Path) -> tf.data.Dataset:
        """Create TF Dataset from processed files"""
        def generator():
            for preset_dir in data_dir.glob('*/*'):  # Assuming structure: complexity/preset_name
                if preset_dir.is_dir():
                    yield self._load_preset_data(preset_dir)

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 128), dtype=tf.float32),  # Mel spectrogram
                tf.TensorSpec(shape=(self.config['model']['num_params'],), dtype=tf.float32)
            )
        ).batch(self.config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)

    def _load_preset_data(self, preset_dir: Path):
        """Load individual preset data"""
        # Load features
        mel = np.load(preset_dir / 'audio' / f'{preset_dir.name}_mel.npy').T
        params = np.load(preset_dir / 'params' / f'{preset_dir.name}.npy')
        
        return mel, params

    def build_model(self):
        """Initialize model architecture"""
        self.model = create_model(  # Assume this exists in model/
            num_params=self.config['model']['num_params'],
            **self.config['model']['architecture']
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss=self.config['training']['loss'],
            metrics=self.config['training']['metrics']
        )

    def train(self):
        """Execute training pipeline"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(Path(self.config['training']['output_dir']) / 'checkpoints' / 'model_{epoch}'),
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path(self.config['training']['output_dir']) / 'logs')
            )
        ]

        self.model.fit(
            self.train_dataset,
            epochs=self.config['training']['epochs'],
            validation_data=self.val_dataset,
            callbacks=callbacks
        )

    def save(self):
        """Save final model"""
        save_path = Path(self.config['training']['output_dir']) / 'final_model'
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural synth modeler')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--data-root', type=str,
                       default='data/training_data',
                       help='Root directory of processed data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    config = config_loader.get_config(args.config)
    config['data']['training_data_root'] = args.data_root
    
    # Initialize and run training
    trainer = ModelTrainer(config)
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.save()
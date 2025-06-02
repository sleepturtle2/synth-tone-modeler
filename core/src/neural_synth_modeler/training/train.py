import argparse
import logging
from datetime import datetime
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import re

from neural_synth_modeler.model import create_model, validate_model_config
from neural_synth_modeler.model.parameter_loss import ParameterLoss
from neural_synth_modeler.utils.config_loader import get_config

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_file = log_dir / f"training_{timestamp}.log"

logger = logging.getLogger("training")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(train_log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def create_dataset(split_dir: Path, batch_size: int, config: dict) -> tf.data.Dataset:
    param_files = list((split_dir / "params").glob("*.json"))

    def gen():
        for param_path in param_files:
            raw_name = param_path.stem
            base_name = re.sub(r'[^\w\-\.]', '_', raw_name)
            try:
                features = {
                    'mel': np.load(split_dir / "audio" / f"{base_name}_mel.npy").T,
                    'f0': np.load(split_dir / "audio" / f"{base_name}_f0.npy").T,
                    'loudness': np.load(split_dir / "audio" / f"{base_name}_loudness.npy").T
                }

                with open(param_path) as f:
                    raw_params = json.load(f)

                params = {
                    'oscillators': np.array([raw_params[p] for p in config['model']['osc_params']]),
                    'filters': np.array([raw_params[p] for p in config['model']['filter_params']]),
                    'fx': np.array([raw_params[p] for p in config['model']['fx_params']])
                }

                yield features, params
            except Exception as e:
                logger.warning(f"Skipping {raw_name}: {str(e)}")

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'mel': tf.TensorSpec(shape=(None, 128), dtype=tf.float32),
                'f0': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'loudness': tf.TensorSpec(shape=(None,), dtype=tf.float32)
            },
            {
                'oscillators': tf.TensorSpec(shape=(len(config['model']['osc_params']),), dtype=tf.float32),
                'filters': tf.TensorSpec(shape=(len(config['model']['filter_params']),), dtype=tf.float32),
                'fx': tf.TensorSpec(shape=(len(config['model']['fx_params']),), dtype=tf.float32)
            }
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train(config_path: str):
    config = get_config(config_path)
    logger.info(f"Loaded config: {list(config.keys())}")

    try:
        model_config = config.get('model', {})
        validate_model_config(model_config)
        if not model_config:
            raise ValueError("Missing 'model' section in config")

        model = create_model(config)
        logger.info("Model created successfully")

        loss_config = config.get('loss', {
            'osc_weight': 1.0,
            'filter_weight': 1.0,
            'fx_weight': 1.0
        })
        loss = ParameterLoss(**loss_config)

        training_config = config.get('training', {
            'learning_rate': 0.001
        })
        model.compile(
            optimizer=tf.keras.optimizers.Adam(training_config['learning_rate']),
            loss=loss,
            metrics={'oscillators': 'mae', 'filters': 'mae', 'fx': 'mae'}
        )
        logger.info("Model compiled successfully")
    except (KeyError, ValueError) as e:
        logger.error(f"Configuration error: {str(e)}")
        logger.error(f"Current model config: {config.get('model', {})}")
        raise

    data_root = Path(config['data'].get('root', 'data/training_data'))

    def resolve_split_paths(root: Path, split_value):
        if isinstance(split_value, (str, Path)):
            return [root / split_value]
        elif isinstance(split_value, (list, tuple)):
            return [root / s for s in split_value]
        else:
            raise TypeError(f"Unsupported type for split_value: {type(split_value)}")

    train_paths = resolve_split_paths(data_root, config['data']['train_split'])
    val_paths = resolve_split_paths(data_root, config['data']['val_split'])

    logger.info(f"Train paths: {train_paths}")
    logger.info(f"Validation paths: {val_paths}")

    train_ds = tf.data.Dataset.sample_from_datasets(
        [create_dataset(p, config['training']['batch_size'], config) for p in train_paths]
    ).repeat()

    val_ds = tf.data.Dataset.sample_from_datasets(
        [create_dataset(p, config['training']['batch_size'], config) for p in val_paths]
    ).repeat()

    logger.info("\n=== DATA VALIDATION ===")
    for path in train_paths + val_paths:
        audio_dir = path / "audio"
        param_dir = path / "params"

        logger.info(f"Checking {path}:")
        logger.info(f"  Audio files: {len(list(audio_dir.glob('*.npy')))}")
        logger.info(f"  Param files: {len(list(param_dir.glob('*.json')))}")

    try:
        train_size = sum(1 for _ in train_ds.take(1))
        val_size = sum(1 for _ in val_ds.take(1))
        logger.info(f"Train dataset non-empty: {train_size > 0}")
        logger.info(f"Validation dataset non-empty: {val_size > 0}")

        if train_size == 0 or val_size == 0:
            raise ValueError("Empty dataset detected. Check data paths and formats.")
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        raise

    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'checkpoints' / 'model_{epoch}.keras'),
            save_freq='epoch'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs')
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=config['training'].get('patience', 10),
            restore_best_weights=True
        )
    ]

    logger.info(f"Starting training for {config['training']['epochs']} epochs")
    model.fit(
        train_ds,
        steps_per_epoch=config['training'].get('steps_per_epoch', 100),
        validation_data=val_ds,
        validation_steps=config['training'].get('validation_steps', 20),
        epochs=config['training']['epochs'],
        callbacks=callbacks
    )

    # Save Keras model
    keras_path = output_dir / f'final_model_{timestamp}.keras'
    model.save(keras_path)
    logger.info(f"Keras model saved to {keras_path}")

    # Save TensorFlow SavedModel
    saved_model_path = output_dir / f'saved_model_{timestamp}'
    model.export(saved_model_path)
    logger.info(f"SavedModel exported to {saved_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train(args.config)
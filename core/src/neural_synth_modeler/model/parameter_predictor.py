import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict

class ParameterPredictor(Model):
    def __init__(self, num_osc_params: int, num_filter_params: int, num_fx_params: int, config: dict):
        super().__init__()
        # Extract and sanitize model config
        self.osc_params = list(config["model"]["osc_params"])
        self.filter_params = list(config["model"]["filter_params"])
        self.fx_params = list(config["model"]["fx_params"])
        
        num_osc_params = len(self.osc_params)
        num_filter_params = len(self.filter_params)
        num_fx_params = len(self.fx_params)

        # Audio feature encoders
        self.mel_conv = layers.Conv1D(64, 3, activation='relu')
        self.f0_dense = layers.Dense(64)
        self.loudness_dense = layers.Dense(64)
        
        # Temporal processing
        self.lstm = layers.Bidirectional(layers.LSTM(128))
        
        # Parameter decoders
        self.osc_decoder = layers.Dense(num_osc_params, activation='sigmoid')
        self.filter_decoder = layers.Dense(num_filter_params, activation='tanh')
        self.fx_decoder = layers.Dense(num_fx_params, activation='sigmoid')

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Process mel spectrogram [batch, time, bins]
        mel = self.mel_conv(inputs['mel'])  # (batch, time, 64)
        mel = layers.GlobalAveragePooling1D()(mel)  # (batch, 64)
        
        # Process scalar features with proper shape handling
        f0 = self.f0_dense(tf.expand_dims(inputs['f0'], axis=-1))  # (batch, time, 64)
        loudness = self.loudness_dense(tf.expand_dims(inputs['loudness'], axis=-1))  # (batch, time, 64)
        
        # Apply same pooling to all features
        f0 = layers.GlobalAveragePooling1D()(f0)  # (batch, 64)
        loudness = layers.GlobalAveragePooling1D()(loudness)  # (batch, 64)
        
        # Combine features
        x = layers.concatenate([mel, f0, loudness])  # (batch, 192)
        
        # Add temporal dimension for LSTM
        x = tf.expand_dims(x, axis=1)  # (batch, 1, 192)
        x = self.lstm(x)  # (batch, 256)
        
        return {
            'oscillators': self.osc_decoder(x),  # (batch, num_osc_params)
            'filters': self.filter_decoder(x),   # (batch, num_filter_params)
            'fx': self.fx_decoder(x)             # (batch, num_fx_params)
        }

    def get_config(self):
        return {
            'num_osc_params': self.osc_decoder.units,
            'num_filter_params': self.filter_decoder.units,
            'num_fx_params': self.fx_decoder.units
        }
    
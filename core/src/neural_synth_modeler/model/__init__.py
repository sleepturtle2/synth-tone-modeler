from .parameter_predictor import ParameterPredictor
from .parameter_loss import ParameterLoss

def create_model(config: dict) -> ParameterPredictor:
    """Create a ParameterPredictor model with default values for missing parameters"""
    return ParameterPredictor(
        num_osc_params=config.get('osc_params', 10),      # Default 10 if not specified
        num_filter_params=config.get('filter_params', 8), # Default 8 if not specified
        num_fx_params=config.get('fx_params', 6),         # Default 6 if not specified
        config=config
    )

def validate_model_config(config: dict):
    """Validate that the model config contains required parameters"""
    required = ['osc_params', 'filter_params', 'fx_params']
    for param in required:
        if param not in config:
            raise ValueError(
                f"Missing required model parameter: {param}\n"
                f"Current model config: {config}"
            )
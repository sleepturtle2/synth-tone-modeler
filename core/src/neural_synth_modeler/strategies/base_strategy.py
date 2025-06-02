# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List

class ParameterEditStrategy(ABC):
    def __init__(self, config_path: str = "strategies.random"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, path: str) -> Dict:
        """Load strategy-specific config"""
        from utils.config_loader import ConfigLoader
        return ConfigLoader().get(path, {})
    
    @abstractmethod
    def generate_edits(self, 
                      preset_params: List[str],
                      current_values: Dict[str, float]) -> Dict[str, float]:
        """Generate parameter modifications"""
        pass
# strategies/random_edit.py
import random
from typing import Dict, List
import numpy as np
from .base_strategy import ParameterEditStrategy

class RandomEditStrategy(ParameterEditStrategy):
    def __init__(self):
        super().__init__("strategies.random")
        self._validate_config()
        self._init_normalization_rules()
        
    def _validate_config(self):
        """Ensure required config exists"""
        from utils.validation import validate_config
        validate_config(self.config, [
            'min_change',
            'max_change',
            'max_params_to_edit',
            'preset_include_list',
            'preset_exclude_list'
        ])
    
    def _init_normalization_rules(self):
        """Load parameter normalization ranges"""
        self.norm_ranges = {
            param: (min_val, max_val) 
            for param, (min_val, max_val) in 
            self.config.get('normalization_ranges', {}).items()
        }

    def generate_edits(self, 
                     preset_params: List[str],
                     current_values: Dict[str, float],
                     preset_name: str = None) -> Dict[str, float]:
        """
        Generates constrained random parameter edits
        
        Args:
            preset_params: Available parameter names
            current_values: Current {param: value} mapping
            preset_name: Name of current preset for filtering
            
        Returns:
            Dictionary of {parameter_name: new_value}
        """
        # Check preset inclusion/exclusion
        if self._should_skip_preset(preset_name):
            return {}
            
        # Filter editable parameters
        editable = self._filter_editable(preset_params)
        
        # Select random subset
        n_edits = min(
            self.config['max_params_to_edit'], 
            len(editable)
        )
        selected = random.sample(editable, n_edits)
        
        # Generate normalized changes
        edits = {}
        for param in selected:
            current_val = current_values[param]
            if param in self.norm_ranges:
                # Apply normalized random change
                min_val, max_val = self.norm_ranges[param]
                new_val = self._normalized_random(
                    current_val, 
                    min_val, 
                    max_val
                )
            else:
                # Apply absolute random change
                change = random.uniform(
                    self.config['min_change'],
                    self.config['max_change']
                )
                new_val = current_val + change
            
            edits[param] = new_val
        
        return edits

    def _should_skip_preset(self, preset_name: str) -> bool:
        """Check preset against inclusion/exclusion rules"""
        if not preset_name:
            return False
            
        include_list = self.config['preset_include_list']
        exclude_list = self.config['preset_exclude_list']
        
        if include_list and preset_name not in include_list:
            return True
        if exclude_list and preset_name in exclude_list:
            return True
        return False

    def _normalized_random(self, current: float, min_val: float, max_val: float) -> float:
        """Generate random value within normalized range"""
        # Scale change range to parameter's normalized range
        scaled_min = max(min_val, current + (self.config['min_change'] * (max_val-min_val)))
        scaled_max = min(max_val, current + (self.config['max_change'] * (max_val-min_val)))
        return np.clip(random.uniform(scaled_min, scaled_max), min_val, max_val)

    def _filter_editable(self, all_params: List[str]) -> List[str]:
        """Apply parameter inclusion/exclusion rules"""
        excluded = self.config.get('excluded_params', [])
        included = self.config.get('included_params', all_params)
        return [p for p in all_params 
               if p in included and p not in excluded]
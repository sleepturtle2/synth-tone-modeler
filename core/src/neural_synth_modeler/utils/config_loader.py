import os
import tomli
from pathlib import Path
from typing import Optional, Union, Dict, Any

class ConfigLoader:
    _instance = None
    _config_dir = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config_dir = get_project_root() / "config"
            print(f"Config directory set to: {cls._config_dir}")  
            
            # Verify the directory exists
            if not cls._config_dir.exists():
                raise FileNotFoundError(f"Config directory not found at {cls._config_dir}")
            
            cls._instance._load_main_config()
        return cls._instance    
    
    def _load_main_config(self):
        """Load the main config.toml file"""
        main_config_path = self._config_dir / "config.toml"
        print(f"Looking for main config at: {main_config_path}")  # Debug
        if not main_config_path.exists():
            raise FileNotFoundError(f"Main config file not found at {main_config_path}")
            
        with open(main_config_path, "rb") as f:
            self.main_config = tomli.load(f)
            
        # Set environment variables from main config
        if "paths" in self.main_config:
            paths = self.main_config["paths"]
            os.environ["PRESET_DIR"] = str(paths.get("preset_dir", ""))
            os.environ["OUTPUT_DIR"] = str(paths.get("output_dir", ""))
    
    @classmethod
    def get_config(cls, config_name: str, key_path: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """Get config values from any TOML file in the centralized config directory."""
        if not cls._config_dir:
            cls._config_dir = get_project_root() / "config"

        if not config_name.endswith(".toml"):
            config_name += ".toml"

        config_path = cls._config_dir / config_name
        print(f"Looking for config file at: {config_path}")

        if not config_path.exists():
            available_files = [f.name for f in cls._config_dir.glob('*') if f.is_file()]
            raise FileNotFoundError(
                f"Config file '{config_name}' not found in {cls._config_dir}\n"
                f"Available files: {', '.join(available_files)}"
            )

        with open(config_path, 'rb') as f:
            config = tomli.load(f)

        if not key_path:
            return config

        # Traverse nested keys
        current = config
        for key in key_path.split('.'):
            if key not in current:
                raise KeyError(f"Key '{key}' not found in config path '{key_path}'")
            current = current[key]

        return current
    
def get_project_root() -> Path:
        """Get root of the project dynamically (adjust levels if needed)."""
        return Path(__file__).resolve().parents[1]  # assumes /core/src/neural_synth_modeler/utils/common.py\

# Module-level convenience function
def get_config(config_name: str, key_path: Optional[str] = None) -> Union[Dict[str, Any], Any]:
    """Convenience function to access ConfigLoader's get_config method"""
    return ConfigLoader.get_config(config_name, key_path)
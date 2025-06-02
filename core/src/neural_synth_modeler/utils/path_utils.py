from pathlib import Path
from .config_loader import ConfigLoader

class PathUtils:
    @staticmethod
    def validate_path(path: str) -> str:
        """Convert to absolute path string and validate"""
        path_obj = Path(path).resolve()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return str(path_obj)
    
    @staticmethod
    def get_preset_path(preset_name: str) -> str:
        config = ConfigLoader().get("paths")
        return str(Path(config["preset_dir"]) / preset_name)
    
    @staticmethod
    def get_output_path(category: str, filename: str) -> str:
        config = ConfigLoader().get("paths")
        base_dir = Path(config["output_dir"])
        subdir = config.get(f"{category}_subdir", category)
        return str(base_dir / subdir / filename)
import os
import re
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import Dict, List, Set

class FXPParser:
    def __init__(self):
        self.all_parameters: Set[str] = set()
        self.preset_db: List[Dict] = []
        
    def extract_xml_from_fxp(self, fxp_path: str) -> str:
        """Extract XML portion from binary-wrapped FXP file"""
        with open(fxp_path, 'rb') as f:
            data = f.read()
        
        # Find XML start/end markers
        xml_start = data.find(b'<?xml')
        xml_end = data.find(b'</patch>')
        
        if xml_start == -1 or xml_end == -1:
            raise ValueError("No valid XML found in FXP file")
        
        # Extract and decode XML portion
        xml_data = data[xml_start:xml_end + len(b'</patch>')]
        return xml_data.decode('utf-8', errors='ignore')

    def parse_fxp(self, fxp_path: str) -> Dict:
        """Parse a single FXP file and return its parameters"""
        try:
            xml_content = self.extract_xml_from_fxp(fxp_path)
            root = ET.fromstring(xml_content)
            
            preset_info = {
                'name': root.find('meta').get('name', ''),
                'category': root.find('meta').get('category', ''),
                'author': root.find('meta').get('author', ''),
                'file': os.path.basename(fxp_path),
                'parameters': {}
            }
            
            params = root.find('parameters')
            for param in params:
                param_name = param.tag
                param_value = float(param.get('value', 0))
                param_type = int(param.get('type', 0))
                
                preset_info['parameters'][param_name] = {
                    'value': param_value,
                    'type': param_type
                }
                self.all_parameters.add(param_name)
                
            return preset_info
            
        except Exception as e:
            print(f"Error parsing {fxp_path}: {str(e)}")
            return None

    def scan_presets(self, preset_dir: str) -> None:
        """Scan a directory for FXP files and parse them"""
        preset_dir = Path(preset_dir)
        fxp_files = list(preset_dir.glob('**/*.fxp'))
        
        print(f"Found {len(fxp_files)} FXP files to process...")
        
        for fxp_path in fxp_files:
            preset_data = self.parse_fxp(str(fxp_path))
            if preset_data:
                self.preset_db.append(preset_data)
                
        print(f"Processed {len(self.preset_db)} presets")
        print(f"Found {len(self.all_parameters)} unique parameters")

    def save_database(self, output_file: str) -> None:
        """Save the parameter database to a JSON file"""
        db = {
            'parameters': sorted(list(self.all_parameters)),
            'presets': self.preset_db
        }
        
        with open(output_file, 'w') as f:
            json.dump(db, f, indent=2)
            
        print(f"Database saved to {output_file}")

if __name__ == "__main__":
    parser = FXPParser()
    
    # Configure these paths
    PRESET_DIR = "data/presets"  # Directory containing FXP files
    OUTPUT_DB = "data/parameter_db.json"  # Output database file
    
    # Process all presets
    parser.scan_presets(PRESET_DIR)
    parser.save_database(OUTPUT_DB)
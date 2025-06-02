import xml.etree.ElementTree as ET
import struct
from collections import OrderedDict

def parse_hybrid_fxp(fxp_path):
    """
    Parses Surge XT .fxp files that contain both XML and binary data.
    Returns parameter values and metadata.
    """
    with open(fxp_path, 'rb') as f:
        # Read entire file
        data = f.read()
        
        # Find XML start/end markers
        xml_start = data.find(b'<?xml')
        xml_end = data.find(b'</patch>')
        
        if xml_start == -1 or xml_end == -1:
            raise ValueError("XML markers not found in file")
            
        # Extract and parse XML portion
        xml_data = data[xml_start:xml_end+8]  # +8 to include </patch>
        root = ET.fromstring(xml_data)
        
        # Parse parameters from XML
        params = OrderedDict()
        for param in root.findall('.//parameters/*[@value]'):
            param_name = param.tag
            param_value = float(param.get('value'))
            params[param_name] = param_value
            
        # Parse binary section if needed (for additional parameters)
        binary_start = xml_end + 8
        if len(data) > binary_start:
            # Example: Read 4 bytes as float from binary section
            # Adjust this based on your needs
            binary_params = struct.unpack_from('>f', data, offset=binary_start)
            params['binary_param1'] = binary_params[0]
            
        # Get metadata
        meta = root.find('.//meta')
        metadata = {
            'name': meta.get('name'),
            'category': meta.get('category'),
            'author': meta.get('author')
        }
        
        return params, metadata

# Example usage
if __name__ == "__main__":
    try:
        params, meta = parse_hybrid_fxp("Bowed Plucked Pipe.fxp")
        print(f"Preset: {meta['name']} ({meta['category']})")
        print(f"Found {len(params)} parameters")
        print("\nFirst 10 parameters:")
        for name, value in list(params.items())[:10]:
            print(f"{name}: {value:.6f}")
            
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
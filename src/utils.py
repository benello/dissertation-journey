import yaml
import logging
from pathlib import Path

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', mode='w')
        ]
    )

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_output_dirs():
    """Create necessary output directories if they don't exist."""
    dirs = ['data', 'outputs', 'outputs/figures']
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    return Path('outputs')
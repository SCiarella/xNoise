"""Configuration utilities for loading and managing YAML configs"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for DDPM-CLIP training"""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary"""
        self._config = config_dict
        self._update_paths()

    def _update_paths(self):
        """Update paths to include model name"""
        model_name = self.get('model_name', 'default')

        # Update save_dir and checkpoint_dir to include model name
        if 'paths' in self._config:
            base_save_dir = self._config['paths'].get('save_dir', 'images/')
            base_checkpoint_dir = self._config['paths'].get(
                'checkpoint_dir', 'model_checkpoint/')

            # Create paths with model name
            self._config['paths']['save_dir'] = os.path.join(
                base_save_dir, model_name)
            self._config['paths']['checkpoint_dir'] = os.path.join(
                base_checkpoint_dir, model_name)

    def get(self, key: str, default=None):
        """Get a top-level config value"""
        return self._config.get(key, default)

    def __getitem__(self, key: str):
        """Access config sections like a dictionary"""
        return self._config[key]

    def __contains__(self, key: str):
        """Check if a key exists in config"""
        return key in self._config

    def to_dict(self):
        """Return the config as a dictionary"""
        return self._config.copy()

    @property
    def model_name(self) -> str:
        """Get the model name"""
        return self.get('model_name', 'default')

    @property
    def save_dir(self) -> str:
        """Get the save directory path"""
        return self._config['paths']['save_dir']

    @property
    def checkpoint_dir(self) -> str:
        """Get the checkpoint directory path"""
        return self._config['paths']['checkpoint_dir']

    def create_directories(self):
        """Create necessary directories for saving outputs"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Created directories for model '{self.model_name}':")
        print(f'  Save dir: {self.save_dir}')
        print(f'  Checkpoint dir: {self.checkpoint_dir}')


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config object with loaded configuration

    Example:
        >>> config = load_config('config/model_v2.yaml')
        >>> print(config.model_name)
        >>> print(config['training']['epochs'])
    """
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path_obj, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to a YAML file

    Args:
        config: Config object to save
        save_path: Path where to save the YAML file
    """
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path_obj, 'w') as f:
        yaml.dump(config.to_dict(),
                  f,
                  default_flow_style=False,
                  sort_keys=False)

    print(f'Configuration saved to: {save_path}')

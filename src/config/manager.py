"""
Configuration management module for Claude.md token reduction project.

This module provides secure configuration loading, validation, and management
with proper error handling and security checks.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.security.validator import validator


class ConfigurationManager:
    """
    Secure configuration manager for the token reduction system.
    
    Features:
    - Secure configuration loading
    - Environment variable support
    - Configuration validation
    - Default value management
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        
        self.config_cache = {}
        self.default_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'security': {
                'max_file_size': 10485760,  # 10MB
                'allowed_extensions': ['.md', '.txt', '.yaml', '.yml', '.json'],
                'safe_directories': [
                    '/tmp/claude_md_processing',
                    '/var/tmp/secure_processing',
                    os.path.expanduser('~/Documents/claude_md_safe'),
                    os.path.join(os.path.dirname(__file__), '..', '..', 'temp')
                ],
                'enable_audit_log': True
            },
            'token_reduction': {
                'target_reduction': 0.6,  # 60% reduction target
                'compression_level': 'medium',
                'preserve_formatting': True,
                'cache_results': True
            },
            'processing': {
                'max_concurrent_files': 5,
                'timeout_seconds': 30,
                'chunk_size': 8192
            },
            'output': {
                'format': 'markdown',
                'include_metadata': True,
                'backup_original': True
            }
        }
    
    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        Load configuration from file with security validation.
        
        Args:
            config_file: Name of the configuration file
            
        Returns:
            Dict containing configuration settings
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = self.config_dir / config_file
        
        # Validate the configuration file path
        if not validator.validate_file_path(str(config_path)):
            raise ValueError(f"Invalid configuration file path: {config_path}")
        
        # Check if file exists
        if not config_path.exists():
            validator.log_security_event("CONFIG_NOT_FOUND", f"Configuration file not found: {config_path}")
            return self.default_config.copy()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(file)
                elif config_file.endswith('.json'):
                    config_data = json.load(file)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_file}")
            
            # Validate configuration
            if not validator.validate_config(config_data):
                raise ValueError("Configuration failed security validation")
            
            # Merge with defaults
            merged_config = self._merge_configs(self.default_config, config_data)
            
            # Cache the configuration
            self.config_cache[config_file] = merged_config
            
            validator.log_security_event("CONFIG_LOADED", f"Configuration loaded: {config_file}")
            return merged_config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            validator.log_security_event("CONFIG_PARSE_ERROR", f"Failed to parse {config_file}: {e}")
            raise ValueError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            validator.log_security_event("CONFIG_LOAD_ERROR", f"Failed to load {config_file}: {e}")
            raise
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_setting(self, key_path: str, config_file: str = "config.yaml", default: Any = None) -> Any:
        """
        Get a specific configuration setting using dot notation.
        
        Args:
            key_path: Dot-separated path to the setting (e.g., 'security.max_file_size')
            config_file: Configuration file to load from
            default: Default value if setting not found
            
        Returns:
            The configuration value
        """
        if config_file not in self.config_cache:
            self.load_config(config_file)
        
        config = self.config_cache[config_file]
        keys = key_path.split('.')
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_setting(self, key_path: str, value: Any, config_file: str = "config.yaml") -> None:
        """
        Set a configuration setting in memory.
        
        Args:
            key_path: Dot-separated path to the setting
            value: Value to set
            config_file: Configuration file to modify
        """
        if config_file not in self.config_cache:
            self.load_config(config_file)
        
        config = self.config_cache[config_file]
        keys = key_path.split('.')
        
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        validator.log_security_event("CONFIG_MODIFIED", f"Setting modified: {key_path}")
    
    def save_config(self, config_file: str = "config.yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Configuration file to save to
        """
        if config_file not in self.config_cache:
            raise ValueError(f"No configuration loaded for: {config_file}")
        
        config_path = self.config_dir / config_file
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as file:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.safe_dump(self.config_cache[config_file], file, default_flow_style=False)
                elif config_file.endswith('.json'):
                    json.dump(self.config_cache[config_file], file, indent=2)
            
            validator.log_security_event("CONFIG_SAVED", f"Configuration saved: {config_file}")
            
        except Exception as e:
            validator.log_security_event("CONFIG_SAVE_ERROR", f"Failed to save {config_file}: {e}")
            raise
    
    def get_env_setting(self, env_var: str, default: Any = None) -> Any:
        """
        Get setting from environment variable.
        
        Args:
            env_var: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        value = os.getenv(env_var, default)
        
        # Convert string booleans
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
            
            # Try to convert to number
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                pass
        
        return value


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Convenience function to get configuration."""
    return config_manager.load_config(config_file)


def get_setting(key_path: str, default: Any = None) -> Any:
    """Convenience function to get a specific setting."""
    return config_manager.get_setting(key_path, default=default)


if __name__ == "__main__":
    # Test configuration management
    print("Testing configuration management...")
    
    try:
        config = get_config()
        print("Default configuration loaded successfully")
        print(f"Max file size: {get_setting('security.max_file_size')}")
        print(f"Target reduction: {get_setting('token_reduction.target_reduction')}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("Configuration management test complete.")
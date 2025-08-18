"""
Security validation module for Claude.md token reduction project.

This module provides comprehensive security validation including:
- Input sanitization
- File path validation
- Configuration security checks
- Access control validation
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
import validators


class SecurityValidator:
    """
    Comprehensive security validator for the token reduction system.
    
    This class implements security best practices including:
    - Path traversal prevention
    - Input sanitization
    - Configuration validation
    - Access control checks
    """
    
    # Safe file extensions for Claude.md files
    SAFE_EXTENSIONS = {'.md', '.txt', '.yaml', '.yml', '.json'}
    
    # Maximum file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Forbidden path patterns
    FORBIDDEN_PATTERNS = [
        r'\.\./',  # Path traversal
        r'\.\.\\',  # Windows path traversal
        r'/etc/',   # System directories
        r'/proc/',  # Process filesystem
        r'/sys/',   # System filesystem
        r'C:\\Windows\\',  # Windows system
        r'C:\\Program Files',  # Program files
        r'^//',    # UNC paths
        r'^\\\\',  # Windows UNC paths
    ]
    
    def __init__(self):
        """Initialize the security validator."""
        self.audit_log = []
    
    def validate_file_path(self, file_path: str) -> bool:
        """
        Validate that a file path is safe to access.
        
        Args:
            file_path: The file path to validate
            
        Returns:
            bool: True if the path is safe, False otherwise
            
        Raises:
            ValueError: If the path is malformed or dangerous
        """
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        # Check for forbidden patterns in the original path first
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                self.log_security_event("FORBIDDEN_PATH", f"Forbidden pattern detected: {pattern} in {file_path}")
                return False
        
        # Convert to Path object for safer handling
        try:
            path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            self.log_security_event("INVALID_PATH", f"Malformed path: {file_path}, error: {e}")
            return False
        
        # Check for forbidden patterns in resolved path too
        path_str = str(path)
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                self.log_security_event("FORBIDDEN_PATH", f"Forbidden pattern detected: {pattern} in {path_str}")
                return False
        
        # Check file extension
        if path.suffix.lower() not in self.SAFE_EXTENSIONS:
            self.log_security_event("UNSAFE_EXTENSION", f"Unsafe file extension: {path.suffix}")
            return False
        
        # Check if file exists and is accessible
        if path.exists():
            try:
                # Check file size
                file_size = path.stat().st_size
                if file_size > self.MAX_FILE_SIZE:
                    self.log_security_event("FILE_TOO_LARGE", f"File too large: {file_size} bytes")
                    return False
                
                # Check if it's actually a file
                if not path.is_file():
                    self.log_security_event("NOT_A_FILE", f"Path is not a file: {path}")
                    return False
                    
            except (OSError, PermissionError) as e:
                self.log_security_event("ACCESS_ERROR", f"Cannot access file: {path}, error: {e}")
                return False
        
        return True
    
    def sanitize_input(self, user_input: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            str: Sanitized input string
        """
        if not user_input:
            return ""
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', user_input)
        
        # Remove script tags and other dangerous HTML
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
            self.log_security_event("INPUT_TRUNCATED", "Input truncated due to length")
        
        return sanitized.strip()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration settings for security.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if configuration is secure, False otherwise
        """
        if not isinstance(config, dict):
            self.log_security_event("INVALID_CONFIG_TYPE", "Configuration must be a dictionary")
            return False
        
        # Check for required security settings
        required_settings = ['max_file_size', 'allowed_extensions', 'safe_directories']
        for setting in required_settings:
            if setting not in config:
                self.log_security_event("MISSING_SECURITY_SETTING", f"Missing required setting: {setting}")
                return False
        
        # Validate file size limits
        max_size = config.get('max_file_size', 0)
        if not isinstance(max_size, int) or max_size <= 0 or max_size > self.MAX_FILE_SIZE:
            self.log_security_event("INVALID_FILE_SIZE", f"Invalid max_file_size: {max_size}")
            return False
        
        # Validate allowed extensions
        extensions = config.get('allowed_extensions', [])
        if not isinstance(extensions, list):
            self.log_security_event("INVALID_EXTENSIONS", "Invalid allowed_extensions format")
            return False
        
        # Check each extension is a string and starts with dot
        for ext in extensions:
            if not isinstance(ext, str) or not ext.startswith('.'):
                self.log_security_event("INVALID_EXTENSIONS", "Invalid allowed_extensions format")
                return False
        
        return True
    
    def log_security_event(self, event_type: str, message: str) -> None:
        """
        Log a security event for auditing.
        
        Args:
            event_type: Type of security event
            message: Detailed message about the event
        """
        import datetime
        
        event = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'type': event_type,
            'message': message
        }
        
        self.audit_log.append(event)
        
        # In production, this would write to a secure log file
        print(f"SECURITY EVENT [{event_type}]: {message}")
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get the current audit log.
        
        Returns:
            List of security events
        """
        return self.audit_log.copy()


# Global validator instance
validator = SecurityValidator()


def validate_file_path(file_path: str) -> bool:
    """Convenience function for file path validation."""
    return validator.validate_file_path(file_path)


def sanitize_input(user_input: str) -> str:
    """Convenience function for input sanitization."""
    return validator.sanitize_input(user_input)


if __name__ == "__main__":
    # Run security validation checks
    print("Running security validation checks...")
    
    # Test cases
    test_cases = [
        "/tmp/test.md",  # Valid
        "../../../etc/passwd",  # Invalid - path traversal
        "normal_file.txt",  # Valid
        "C:\\Windows\\system32\\config",  # Invalid - Windows system
        "file.exe",  # Invalid - unsafe extension
    ]
    
    for test_path in test_cases:
        try:
            result = validate_file_path(test_path)
            print(f"Path: {test_path} -> {'SAFE' if result else 'UNSAFE'}")
        except Exception as e:
            print(f"Path: {test_path} -> ERROR: {e}")
    
    print("\nSecurity validation complete.")
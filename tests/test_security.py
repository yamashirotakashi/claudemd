"""
Test suite for security validation module.

This module tests all security-related functionality to ensure
the system is protected against common vulnerabilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.security.validator import SecurityValidator, validate_file_path, sanitize_input


class TestSecurityValidator:
    """Test cases for SecurityValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SecurityValidator()
    
    def test_validate_safe_file_path(self):
        """Test validation of safe file paths."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            assert self.validator.validate_file_path(tmp_path) == True
        finally:
            os.unlink(tmp_path)
    
    def test_validate_unsafe_file_path(self):
        """Test validation rejects unsafe file paths."""
        unsafe_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\system32\\config",
            "//server/share/file.txt",
            "/proc/self/mem"
        ]
        
        for path in unsafe_paths:
            assert self.validator.validate_file_path(path) == False
    
    def test_validate_unsafe_extensions(self):
        """Test validation rejects unsafe file extensions."""
        unsafe_files = [
            "malware.exe",
            "script.bat",
            "virus.com",
            "trojan.scr"
        ]
        
        for file_path in unsafe_files:
            assert self.validator.validate_file_path(file_path) == False
    
    def test_empty_file_path(self):
        """Test validation of empty file path."""
        with pytest.raises(ValueError):
            self.validator.validate_file_path("")
    
    def test_none_file_path(self):
        """Test validation of None file path."""
        with pytest.raises(ValueError):
            self.validator.validate_file_path(None)
    
    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        test_input = "normal text input"
        result = self.validator.sanitize_input(test_input)
        assert result == "normal text input"
    
    def test_sanitize_input_script_removal(self):
        """Test removal of script tags."""
        malicious_input = "<script>alert('xss')</script>Hello World"
        result = self.validator.sanitize_input(malicious_input)
        assert "script" not in result.lower()
        assert "Hello World" in result
    
    def test_sanitize_input_control_chars(self):
        """Test removal of control characters."""
        input_with_control = "Hello\x00\x1f\x7fWorld"
        result = self.validator.sanitize_input(input_with_control)
        assert result == "HelloWorld"
    
    def test_sanitize_input_length_limit(self):
        """Test input length limiting."""
        long_input = "A" * 2000
        result = self.validator.sanitize_input(long_input)
        assert len(result) <= 1000
    
    def test_sanitize_empty_input(self):
        """Test sanitization of empty input."""
        assert self.validator.sanitize_input("") == ""
        assert self.validator.sanitize_input(None) == ""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        valid_config = {
            'max_file_size': 1048576,
            'allowed_extensions': ['.md', '.txt'],
            'safe_directories': ['/tmp']
        }
        
        assert self.validator.validate_config(valid_config) == True
    
    def test_validate_invalid_config_type(self):
        """Test validation rejects non-dict config."""
        invalid_configs = ["string", 123, None, []]
        
        for config in invalid_configs:
            assert self.validator.validate_config(config) == False
    
    def test_validate_config_missing_required(self):
        """Test validation rejects config with missing required fields."""
        incomplete_config = {
            'max_file_size': 1048576
            # Missing allowed_extensions and safe_directories
        }
        
        assert self.validator.validate_config(incomplete_config) == False
    
    def test_validate_config_invalid_file_size(self):
        """Test validation rejects invalid file sizes."""
        invalid_configs = [
            {'max_file_size': -1, 'allowed_extensions': ['.md'], 'safe_directories': []},
            {'max_file_size': 0, 'allowed_extensions': ['.md'], 'safe_directories': []},
            {'max_file_size': 100000000, 'allowed_extensions': ['.md'], 'safe_directories': []},  # Too large
            {'max_file_size': "invalid", 'allowed_extensions': ['.md'], 'safe_directories': []}
        ]
        
        for config in invalid_configs:
            assert self.validator.validate_config(config) == False
    
    def test_validate_config_invalid_extensions(self):
        """Test validation rejects invalid extensions format."""
        invalid_configs = [
            {'max_file_size': 1024, 'allowed_extensions': "invalid", 'safe_directories': []},
            {'max_file_size': 1024, 'allowed_extensions': ["invalid"], 'safe_directories': []},  # No dot
            {'max_file_size': 1024, 'allowed_extensions': [123], 'safe_directories': []}
        ]
        
        for config in invalid_configs:
            assert self.validator.validate_config(config) == False
    
    def test_audit_logging(self):
        """Test security event logging."""
        initial_log_count = len(self.validator.get_audit_log())
        
        self.validator.log_security_event("TEST_EVENT", "Test message")
        
        audit_log = self.validator.get_audit_log()
        assert len(audit_log) == initial_log_count + 1
        
        latest_event = audit_log[-1]
        assert latest_event['type'] == "TEST_EVENT"
        assert latest_event['message'] == "Test message"
        assert 'timestamp' in latest_event
    
    def test_file_size_validation(self):
        """Test validation of file size limits."""
        # Create a temporary large file
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            # Write data exceeding the limit
            large_data = "A" * (self.validator.MAX_FILE_SIZE + 1)
            tmp.write(large_data.encode())
            tmp_path = tmp.name
        
        try:
            assert self.validator.validate_file_path(tmp_path) == False
        finally:
            os.unlink(tmp_path)


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    def test_validate_file_path_function(self):
        """Test the standalone validate_file_path function."""
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            assert validate_file_path(tmp_path) == True
        finally:
            os.unlink(tmp_path)
        
        assert validate_file_path("../../../etc/passwd") == False
    
    def test_sanitize_input_function(self):
        """Test the standalone sanitize_input function."""
        test_input = "<script>alert('test')</script>Clean text"
        result = sanitize_input(test_input)
        assert "script" not in result.lower()
        assert "Clean text" in result


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])
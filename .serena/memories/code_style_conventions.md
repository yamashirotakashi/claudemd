# Code Style and Conventions

## Code Style Guidelines

### General Principles
1. **Security First**: All code must pass security review
2. **Clean Implementation**: No legacy code contamination
3. **Modular Design**: Clear separation of concerns
4. **Test Coverage**: Minimum 80% test coverage
5. **Documentation**: Comprehensive inline documentation

### Python Conventions
- **Type Hints**: Full type annotations required using `typing` module
- **Docstrings**: Comprehensive Google-style docstrings for all classes and methods
- **Naming**: Snake_case for functions/variables, PascalCase for classes
- **Import Organization**: Standard library first, then third-party, then local imports

### Security Patterns
- **Input Validation**: All user inputs must be validated through SecurityValidator
- **Path Handling**: Use pathlib.Path for all file operations
- **Error Handling**: Comprehensive exception handling with security event logging
- **Audit Logging**: All security events logged through SecurityValidator.log_security_event()

### Documentation Standards
- Class docstrings include purpose and security considerations
- Method docstrings include Args, Returns, and Raises sections
- Critical security sections marked with comments
- Code blocks use proper type hints and validation

### Constants and Configuration
- Constants defined at class level (CAPS_CASE)
- Security patterns centralized in SecurityValidator
- File size limits and extension validation standardized
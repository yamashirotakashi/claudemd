# Suggested Commands

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=src

# Run specific test file
python -m pytest tests/test_security.py
python -m pytest tests/test_tokenizer.py
```

### Security Validation
```bash
# Run security checks
python -m src.security.validator

# Validate specific file path
python -c "from src.security.validator import validate_file_path; print(validate_file_path('test.md'))"
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Core Application
```bash
# Analyze a Claude.md file
python -c "from src.core.tokenizer import ClaudeMdTokenizer; t = ClaudeMdTokenizer(); print(t.analyze_file('path/to/file.md'))"

# Optimize a file
python -c "from src.core.tokenizer import ClaudeMdTokenizer; t = ClaudeMdTokenizer(); t.optimize_file('input.md', 'output.md')"
```

### System Commands (Linux)
```bash
# File operations
ls -la                    # List files with details
find . -name "*.py"       # Find Python files
grep -r "pattern" src/    # Search for patterns
cd /path/to/directory     # Change directory

# Git operations
git status               # Check repository status
git add .               # Stage all changes
git commit -m "message" # Commit changes
git push                # Push to remote
```
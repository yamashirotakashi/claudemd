# Task Completion Workflow

## Required Steps After Code Changes

### 1. Security Validation
```bash
# Run security validator to check for vulnerabilities
python -m src.security.validator
```

### 2. Run Test Suite
```bash
# Execute all tests to ensure no regressions
python -m pytest tests/ -v
```

### 3. Code Quality Checks
```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### 4. Security-Specific Validation
For security fixes specifically:
- Verify no hardcoded paths remain in test cases
- Confirm cryptographic functions use secure algorithms (SHA-256, not MD5)
- Check all file operations use proper validation
- Ensure error handling doesn't leak sensitive information

### 5. Integration Testing
```bash
# Test the tokenizer with sample files
python -c "
from src.core.tokenizer import ClaudeMdTokenizer
t = ClaudeMdTokenizer()
# Test with a small sample
analysis = t.analyze_file('tests/sample.md')  # if exists
print(f'Token reduction: {analysis.reduction_ratio:.2%}')
"
```

### 6. Documentation Updates
- Update docstrings if method signatures change
- Update security documentation if security patterns change
- Verify all type hints are accurate

### 7. Git Workflow
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "security: fix MD5 vulnerability and hardcoded paths"

# Push changes
git push origin main
```

## Pre-commit Requirements
- All tests must pass (36/36)
- No security vulnerabilities in audit
- Code coverage maintained above 80%
- All linting checks pass
- Type checking passes without errors
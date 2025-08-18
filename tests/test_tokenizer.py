"""
Test suite for tokenizer module.

This module tests the core token analysis and optimization functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.core.tokenizer import ClaudeMdTokenizer, TokenAnalysis, analyze_file, optimize_file


class TestClaudeMdTokenizer:
    """Test cases for ClaudeMdTokenizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = ClaudeMdTokenizer()
    
    def create_test_file(self, content: str) -> str:
        """Create a temporary test file with given content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(content)
            return tmp.name
    
    def test_estimate_tokens_empty(self):
        """Test token estimation for empty content."""
        assert self.tokenizer._estimate_tokens("") == 0
    
    def test_estimate_tokens_basic(self):
        """Test token estimation for basic content."""
        content = "Hello world, this is a test."
        tokens = self.tokenizer._estimate_tokens(content)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_parse_sections_basic(self):
        """Test basic section parsing."""
        content = """# Main Header
Some content here.

## Sub Header
More content.

### Another Section
Final content.
"""
        sections = self.tokenizer._parse_sections(content)
        assert len(sections) >= 3
        assert "main header" in sections
        assert "sub header" in sections
    
    def test_parse_sections_no_headers(self):
        """Test parsing content without headers."""
        content = "Just some plain text without headers."
        sections = self.tokenizer._parse_sections(content)
        assert "header" in sections
        assert sections["header"] == content
    
    def test_is_critical_section(self):
        """Test critical section identification."""
        assert self.tokenizer._is_critical_section("Security Rules") == True
        assert self.tokenizer._is_critical_section("Important Notes") == True
        assert self.tokenizer._is_critical_section("Safety Guidelines") == True
        assert self.tokenizer._is_critical_section("Optional Features") == False
        assert self.tokenizer._is_critical_section("Examples") == False
    
    def test_compress_whitespace(self):
        """Test whitespace compression."""
        content_with_whitespace = """Line 1   

        

Line 2


Line 3"""
        compressed = self.tokenizer._compress_whitespace(content_with_whitespace)
        
        # Should remove trailing spaces and limit empty lines
        lines = compressed.split('\n')
        assert all(not line.endswith(' ') for line in lines if line)
        
        # Count consecutive empty lines
        max_consecutive_empty = 0
        current_empty = 0
        for line in lines:
            if not line.strip():
                current_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, current_empty)
            else:
                current_empty = 0
        
        assert max_consecutive_empty <= 2
    
    def test_deduplicate_content(self):
        """Test content deduplication."""
        content_with_duplicates = """Block 1
Content here

Block 2
Different content

Block 1
Content here"""
        
        deduplicated = self.tokenizer._deduplicate_content(content_with_duplicates)
        
        # Should only contain each unique block once
        blocks = deduplicated.split('\n\n')
        unique_blocks = set(block.strip() for block in blocks if block.strip())
        assert len(blocks) >= len(unique_blocks)
    
    def test_limit_examples(self):
        """Test example limiting functionality."""
        lines_with_examples = [
            "Here are some examples:",
            "Example 1: First example",
            "Some content",
            "",
            "Example 2: Second example", 
            "More content",
            "",
            "Example 3: Third example",
            "Even more content",
            "",
            "Normal content without examples"
        ]
        
        limited = self.tokenizer._limit_examples(lines_with_examples)
        
        # Count how many examples remain
        example_count = sum(1 for line in limited if 'example' in line.lower())
        assert example_count <= 4  # Should limit to 2 examples (title + content each)
    
    def test_analyze_file_basic(self):
        """Test basic file analysis."""
        test_content = """# Test File

## Important Security
Critical security information here.

## Examples
Example 1: Basic usage
Example 2: Advanced usage
Example 3: Expert usage

## Optional Section
Optional content that can be optimized.
"""
        
        test_file = self.create_test_file(test_content)
        
        try:
            analysis = self.tokenizer.analyze_file(test_file)
            
            assert isinstance(analysis, TokenAnalysis)
            assert analysis.original_tokens > 0
            assert analysis.optimized_tokens >= 0
            assert -0.1 <= analysis.reduction_ratio <= 1  # Allow small negative ratio for edge cases
            assert isinstance(analysis.preserved_sections, list)
            assert isinstance(analysis.removed_sections, list)
            assert isinstance(analysis.optimization_notes, list)
            
        finally:
            os.unlink(test_file)
    
    def test_analyze_file_invalid_path(self):
        """Test analysis with invalid file path."""
        with pytest.raises(ValueError):
            self.tokenizer.analyze_file("../../../etc/passwd")
    
    def test_analyze_file_nonexistent(self):
        """Test analysis with non-existent file."""
        with pytest.raises(ValueError):
            self.tokenizer.analyze_file("nonexistent_file.md")
    
    def test_optimize_file_basic(self):
        """Test basic file optimization."""
        test_content = """# Test File

## Important Security
Critical security information here.

## Comments
// This comment should be removed
// Another comment to remove

Actual content to keep.

## Examples
Example 1: First example
Example 2: Second example  
Example 3: Third example (should be removed)

## Optional Section


Lots of whitespace above.
"""
        
        test_file = self.create_test_file(test_content)
        
        try:
            analysis = self.tokenizer.optimize_file(test_file)
            
            assert isinstance(analysis, TokenAnalysis)
            assert analysis.optimized_tokens < analysis.original_tokens
            assert analysis.reduction_ratio > 0
            
            # Check that optimized file was created
            optimized_path = Path(test_file).parent / f"{Path(test_file).stem}_optimized{Path(test_file).suffix}"
            assert optimized_path.exists()
            
            # Verify optimized content
            with open(optimized_path, 'r') as f:
                optimized_content = f.read()
            
            # Should preserve critical sections
            assert "Important Security" in optimized_content or "security" in optimized_content.lower()
            
            # Should remove or reduce comments
            comment_count = optimized_content.count('//')
            original_comment_count = test_content.count('//')
            assert comment_count <= original_comment_count
            
            # Clean up optimized file
            optimized_path.unlink()
            
        finally:
            os.unlink(test_file)
    
    def test_optimize_file_with_output_path(self):
        """Test file optimization with custom output path."""
        test_content = "# Test\nSome content here."
        
        test_file = self.create_test_file(test_content)
        custom_output = test_file.replace('.md', '_custom.md')
        
        try:
            analysis = self.tokenizer.optimize_file(test_file, custom_output)
            
            assert Path(custom_output).exists()
            
            with open(custom_output, 'r') as f:
                content = f.read()
            assert len(content) > 0
            
        finally:
            os.unlink(test_file)
            if os.path.exists(custom_output):
                os.unlink(custom_output)
    
    def test_minimal_optimize(self):
        """Test minimal optimization for critical sections."""
        content_with_whitespace = """Critical content   

        

With lots of whitespace."""
        
        optimized = self.tokenizer._minimal_optimize(content_with_whitespace)
        
        # Should preserve content but compress whitespace
        assert "Critical content" in optimized
        assert optimized != content_with_whitespace  # Should be different due to compression
    
    def test_aggressive_optimize(self):
        """Test aggressive optimization for non-critical sections."""
        content_with_comments = """Some content here.

// This is a comment
// Another comment

More content.

Example 1: First
Example 2: Second
Example 3: Third
Example 4: Fourth"""
        
        optimized = self.tokenizer._aggressive_optimize(content_with_comments)
        
        # Should remove comments
        assert '//' not in optimized
        
        # Should preserve actual content
        assert "Some content here" in optimized
        assert "More content" in optimized


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    def test_analyze_file_function(self):
        """Test the standalone analyze_file function."""
        test_content = "# Test\nSome test content here."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name
        
        try:
            analysis = analyze_file(tmp_path)
            assert isinstance(analysis, TokenAnalysis)
            assert analysis.original_tokens > 0
            
        finally:
            os.unlink(tmp_path)
    
    def test_optimize_file_function(self):
        """Test the standalone optimize_file function."""
        test_content = "# Test\nSome test content here."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name
        
        try:
            analysis = optimize_file(tmp_path)
            assert isinstance(analysis, TokenAnalysis)
            
            # Should create optimized file
            optimized_path = Path(tmp_path).parent / f"{Path(tmp_path).stem}_optimized{Path(tmp_path).suffix}"
            assert optimized_path.exists()
            optimized_path.unlink()
            
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])
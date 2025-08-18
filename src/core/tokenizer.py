"""
Token analysis and optimization module for Claude.md files.

This module provides the core functionality for analyzing and optimizing
token usage in Claude.md files while maintaining functionality.
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from src.security.validator import validator
from src.config.manager import get_setting


@dataclass
class TokenAnalysis:
    """Results of token analysis for a Claude.md file."""
    original_tokens: int
    optimized_tokens: int
    reduction_ratio: float
    preserved_sections: List[str]
    removed_sections: List[str]
    optimization_notes: List[str]


class ClaudeMdTokenizer:
    """
    Secure tokenizer for Claude.md files with optimization capabilities.
    
    This class provides:
    - Token counting and analysis
    - Content optimization
    - Section deduplication
    - Smart compression
    """
    
    # Approximate tokens per character (conservative estimate)
    TOKENS_PER_CHAR = 0.25
    
    # Critical sections that must be preserved
    CRITICAL_SECTIONS = {
        'rules', 'safety', 'security', 'important', 'critical',
        'mandatory', 'required', 'essential', 'core'
    }
    
    # Optimization patterns
    DEDUPLICATION_PATTERNS = [
        r'```[^`]*```',  # Code blocks
        r'#{1,6}\s+.*?(?=\n)',  # Headers
        r'\*\*.*?\*\*',  # Bold text
        r'`[^`]+`',  # Inline code
    ]
    
    def __init__(self):
        """Initialize the tokenizer with security validation."""
        self.seen_content = {}  # For deduplication
        self.optimization_stats = {}
    
    def analyze_file(self, file_path: str) -> TokenAnalysis:
        """
        Analyze a Claude.md file for token optimization opportunities.
        
        Args:
            file_path: Path to the Claude.md file
            
        Returns:
            TokenAnalysis object with optimization results
            
        Raises:
            ValueError: If file path is invalid or file cannot be processed
        """
        # Validate file path
        if not validator.validate_file_path(file_path):
            raise ValueError(f"Invalid or unsafe file path: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Read file content securely
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            validator.log_security_event("FILE_READ_ERROR", f"Cannot read file {file_path}: {e}")
            raise ValueError(f"Cannot read file: {e}")
        
        # Analyze content
        original_tokens = self._estimate_tokens(content)
        sections = self._parse_sections(content)
        optimized_content, optimization_notes = self._optimize_content(content, sections)
        optimized_tokens = self._estimate_tokens(optimized_content)
        
        reduction_ratio = (original_tokens - optimized_tokens) / original_tokens if original_tokens > 0 else 0
        
        return TokenAnalysis(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            reduction_ratio=reduction_ratio,
            preserved_sections=self._get_preserved_sections(sections),
            removed_sections=self._get_removed_sections(sections),
            optimization_notes=optimization_notes
        )
    
    def optimize_file(self, file_path: str, output_path: Optional[str] = None) -> TokenAnalysis:
        """
        Optimize a Claude.md file and save the result.
        
        Args:
            file_path: Path to the input file
            output_path: Path for the optimized output (optional)
            
        Returns:
            TokenAnalysis object with optimization results
        """
        # Validate paths
        if not validator.validate_file_path(file_path):
            raise ValueError(f"Invalid input file path: {file_path}")
        
        if output_path and not validator.validate_file_path(output_path):
            raise ValueError(f"Invalid output file path: {output_path}")
        
        # Analyze the file
        analysis = self.analyze_file(file_path)
        
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Optimize content
        sections = self._parse_sections(content)
        optimized_content, _ = self._optimize_content(content, sections)
        
        # Determine output path
        if not output_path:
            path = Path(file_path)
            output_path = path.parent / f"{path.stem}_optimized{path.suffix}"
        
        # Save optimized content
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(optimized_content)
            
            validator.log_security_event("FILE_OPTIMIZED", f"File optimized: {file_path} -> {output_path}")
            
        except Exception as e:
            validator.log_security_event("FILE_WRITE_ERROR", f"Cannot write optimized file: {e}")
            raise ValueError(f"Cannot write optimized file: {e}")
        
        return analysis
    
    def _estimate_tokens(self, content: str) -> int:
        """
        Estimate the number of tokens in the content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Estimated token count
        """
        if not content:
            return 0
        
        # Basic estimation based on character count and word count
        char_based = len(content) * self.TOKENS_PER_CHAR
        word_based = len(content.split()) * 1.3  # Average tokens per word
        
        # Use the more conservative estimate
        return int(max(char_based, word_based))
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """
        Parse Claude.md content into logical sections.
        
        Args:
            content: File content to parse
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        current_section = "header"
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            # Check for section headers
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.strip('#').strip().lower()
                current_content = [line]
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _optimize_content(self, content: str, sections: Dict[str, str]) -> Tuple[str, List[str]]:
        """
        Optimize content for token reduction.
        
        Args:
            content: Original content
            sections: Parsed sections
            
        Returns:
            Tuple of (optimized_content, optimization_notes)
        """
        optimization_notes = []
        optimized_sections = {}
        
        for section_name, section_content in sections.items():
            if self._is_critical_section(section_name):
                # Preserve critical sections with minimal optimization
                optimized_sections[section_name] = self._minimal_optimize(section_content)
                optimization_notes.append(f"Preserved critical section: {section_name}")
            else:
                # Apply aggressive optimization to non-critical sections
                optimized_content = self._aggressive_optimize(section_content)
                if optimized_content:
                    optimized_sections[section_name] = optimized_content
                    optimization_notes.append(f"Optimized section: {section_name}")
                else:
                    optimization_notes.append(f"Removed empty section: {section_name}")
        
        # Rebuild content
        optimized_content = '\n\n'.join(optimized_sections.values())
        
        # Apply global optimizations
        optimized_content = self._deduplicate_content(optimized_content)
        optimized_content = self._compress_whitespace(optimized_content)
        
        return optimized_content, optimization_notes
    
    def _is_critical_section(self, section_name: str) -> bool:
        """Check if a section is critical and should be preserved."""
        section_lower = section_name.lower()
        return any(keyword in section_lower for keyword in self.CRITICAL_SECTIONS)
    
    def _minimal_optimize(self, content: str) -> str:
        """Apply minimal optimization that preserves functionality."""
        # Only remove excessive whitespace
        return self._compress_whitespace(content)
    
    def _aggressive_optimize(self, content: str) -> str:
        """Apply aggressive optimization for non-critical content."""
        if not content.strip():
            return ""
        
        # Remove comments (lines starting with //)
        lines = content.split('\n')
        lines = [line for line in lines if not line.strip().startswith('//')]
        
        # Remove excessive examples (keep only first 2)
        if 'example' in content.lower():
            lines = self._limit_examples(lines)
        
        # Compress whitespace
        content = '\n'.join(lines)
        content = self._compress_whitespace(content)
        
        return content
    
    def _deduplicate_content(self, content: str) -> str:
        """Remove duplicate content blocks."""
        # Hash content blocks to find duplicates
        blocks = content.split('\n\n')
        seen_hashes = set()
        unique_blocks = []
        
        for block in blocks:
            block_hash = hashlib.sha256(block.strip().encode()).hexdigest()
            if block_hash not in seen_hashes:
                seen_hashes.add(block_hash)
                unique_blocks.append(block)
        
        return '\n\n'.join(unique_blocks)
    
    def _compress_whitespace(self, content: str) -> str:
        """Compress excessive whitespace."""
        # Remove trailing whitespace
        lines = [line.rstrip() for line in content.split('\n')]
        
        # Remove excessive empty lines (max 2 consecutive)
        compressed_lines = []
        empty_count = 0
        
        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:
                    compressed_lines.append(line)
            else:
                empty_count = 0
                compressed_lines.append(line)
        
        return '\n'.join(compressed_lines)
    
    def _limit_examples(self, lines: List[str]) -> List[str]:
        """Limit the number of examples to reduce token usage."""
        result = []
        example_count = 0
        in_example = False
        
        for line in lines:
            if 'example' in line.lower() and not in_example:
                example_count += 1
                in_example = True
                if example_count <= 2:
                    result.append(line)
            elif in_example and line.strip() == "":
                in_example = False
                if example_count <= 2:
                    result.append(line)
            elif not in_example:
                result.append(line)
            elif example_count <= 2:
                result.append(line)
        
        return result
    
    def _get_preserved_sections(self, sections: Dict[str, str]) -> List[str]:
        """Get list of sections that were preserved."""
        return [name for name in sections.keys() if self._is_critical_section(name)]
    
    def _get_removed_sections(self, sections: Dict[str, str]) -> List[str]:
        """Get list of sections that could be removed."""
        return [name for name in sections.keys() if not self._is_critical_section(name)]


# Global tokenizer instance
tokenizer = ClaudeMdTokenizer()


def analyze_file(file_path: str) -> TokenAnalysis:
    """Convenience function for file analysis."""
    return tokenizer.analyze_file(file_path)


def optimize_file(file_path: str, output_path: Optional[str] = None) -> TokenAnalysis:
    """Convenience function for file optimization."""
    return tokenizer.optimize_file(file_path, output_path)


if __name__ == "__main__":
    # Test tokenizer functionality
    print("Testing Claude.md tokenizer...")
    
    # Create a test file
    test_content = """# Test Claude.md File

## Important Security Rules
This section contains critical security information.

## Examples Section
Here are some examples:

Example 1: Basic usage
Example 2: Advanced usage
Example 3: Expert usage

## Optional Features
These features are optional and can be optimized.

## Comments Section
// This is a comment that can be removed
// Another comment

Some actual content here.
"""
    
    test_file = Path("test_claude.md")
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        analysis = analyze_file(str(test_file))
        print(f"Original tokens: {analysis.original_tokens}")
        print(f"Optimized tokens: {analysis.optimized_tokens}")
        print(f"Reduction ratio: {analysis.reduction_ratio:.2%}")
        print(f"Preserved sections: {analysis.preserved_sections}")
        print(f"Optimization notes: {analysis.optimization_notes}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
    
    print("Tokenizer test complete.")
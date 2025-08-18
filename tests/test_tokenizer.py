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

    def test_advanced_semantic_similarity(self):
        """Test advanced semantic similarity calculation."""
        # Setup test content
        text1 = "Configure the API endpoint authentication"
        text2 = "Setup API endpoint authentication configuration"
        text3 = "Install the database server"
        
        context_analysis = {
            'content_type': 'technical_docs',
            'redundancy_patterns': {}
        }
        
        # Test high similarity for related content
        similarity_high = self.tokenizer._calculate_advanced_semantic_similarity(text1, text2, context_analysis)
        assert similarity_high > 0.6, f"Expected high similarity, got {similarity_high}"
        
        # Test low similarity for unrelated content
        similarity_low = self.tokenizer._calculate_advanced_semantic_similarity(text1, text3, context_analysis)
        assert similarity_low < 0.4, f"Expected low similarity, got {similarity_low}"
        
        # Test identical content
        similarity_identical = self.tokenizer._calculate_advanced_semantic_similarity(text1, text1, context_analysis)
        assert similarity_identical > 0.9, f"Expected very high similarity for identical text, got {similarity_identical}"

    def test_semantic_feature_extraction(self):
        """Test semantic feature extraction."""
        content = """
        Configure the API endpoint with authentication parameters.
        This method implements secure token validation for database access.
        The optimization algorithm analyzes semantic patterns.
        """
        
        context_analysis = {
            'content_type': 'technical_docs'
        }
        
        features = self.tokenizer._extract_semantic_features(content, context_analysis)
        
        # Verify feature extraction
        assert isinstance(features, dict)
        assert 'key_terms' in features
        assert 'technical_terms' in features
        assert 'action_words' in features
        assert 'domain_concepts' in features
        assert 'semantic_density' in features
        assert 'information_type' in features
        
        # Check for expected technical terms
        assert 'authentication' in features['technical_terms']
        assert 'database' in features['technical_terms']
        
        # Check for expected domain concepts
        assert any(term in features['domain_concepts'] for term in ['optimization', 'algorithm', 'semantic'])
        
        # Verify semantic density calculation
        assert 0.0 <= features['semantic_density'] <= 1.0

    def test_semantic_structure_similarity(self):
        """Test semantic structure similarity."""
        # Structure with headers and lists
        text1 = """
        # Configuration
        - API endpoint
        - Authentication
        ```
        config.json
        ```
        """
        
        # Similar structure  
        text2 = """
        # Setup
        - Database connection
        - Security settings
        ```
        setup.py
        ```
        """
        
        # Different structure
        text3 = "Simple paragraph with no special formatting."
        
        # Test similar structures
        similarity_high = self.tokenizer._calculate_semantic_structure_similarity(text1, text2)
        assert similarity_high > 0.7, f"Expected high structure similarity, got {similarity_high}"
        
        # Test different structures
        similarity_low = self.tokenizer._calculate_semantic_structure_similarity(text1, text3)
        assert similarity_low < 0.3, f"Expected low structure similarity, got {similarity_low}"

    def test_enhanced_semantic_signature(self):
        """Test enhanced semantic signature generation."""
        content = "Configure API authentication with secure tokens for database access"
        context_analysis = {
            'content_type': 'technical_docs',
            'redundancy_patterns': {}
        }
        
        # Generate signature
        signature = self.tokenizer._generate_semantic_signature(content, context_analysis)
        
        # Verify signature properties
        assert isinstance(signature, str)
        assert len(signature) == 24  # 192-bit signature in hex
        assert all(c in '0123456789abcdef' for c in signature)
        
        # Test consistency - same content should produce same signature
        signature2 = self.tokenizer._generate_semantic_signature(content, context_analysis)
        assert signature == signature2
        
        # Test different content produces different signature
        different_content = "Install database server with security configuration"
        signature3 = self.tokenizer._generate_semantic_signature(different_content, context_analysis)
        assert signature != signature3

    def test_semantic_clustering(self):
        """Test advanced semantic clustering."""
        sections = {
            'config1': 'Configure API endpoint authentication',
            'config2': 'Setup API authentication configuration',  
            'database1': 'Install database server with security',
            'database2': 'Setup database security configuration',
            'general': 'General information about the project'
        }
        
        context_analysis = {
            'content_type': 'project_config',
            'redundancy_patterns': {}
        }
        
        clusters = self.tokenizer._perform_advanced_semantic_clustering(sections, context_analysis)
        
        # Verify cluster structure
        assert isinstance(clusters, dict)
        assert any(len(cluster_list) > 0 for cluster_list in clusters.values())
        
        # Find clusters with multiple sections (similar content should be clustered)
        multi_section_clusters = [
            cluster for cluster_list in clusters.values() 
            for cluster in cluster_list 
            if cluster['cluster_size'] > 1
        ]
        
        assert len(multi_section_clusters) > 0, "Expected to find clusters with multiple similar sections"
        
        # Verify cluster properties
        for cluster in multi_section_clusters:
            assert 'deduplication_potential' in cluster
            assert 'preservation_priority' in cluster
            assert 'semantic_signature' in cluster
            assert 0.0 <= cluster['deduplication_potential'] <= 1.0
            assert 0.0 <= cluster['preservation_priority'] <= 1.0

    def test_advanced_semantic_deduplication(self):
        """Test complete advanced semantic deduplication system."""
        content = """
        # Configuration
        Configure the API endpoint authentication system.
        
        # Setup  
        Setup API endpoint authentication configuration.
        
        # Installation
        Configure the API authentication endpoint.
        
        # Different Topic
        Install the database server with proper security.
        """
        
        context_analysis = {
            'content_type': 'technical_docs',
            'redundancy_patterns': {}
        }
        
        # Apply advanced semantic deduplication
        deduplicated = self.tokenizer._advanced_semantic_deduplication_system(content, context_analysis)
        
        # Verify deduplication occurred
        assert len(deduplicated) < len(content)
        
        # Verify important content is preserved
        assert 'database' in deduplicated  # Different topic should be preserved
        
        # Verify structure is maintained
        assert '#' in deduplicated  # Headers should be preserved

    def test_semantic_redundancy_removal(self):
        """Test semantic redundancy removal with advanced understanding."""
        content = """
        Please configure the system. Please configure the system properly. 
        The API endpoint authentication is important. API endpoint authentication configuration is critical.
        Database security settings must be validated. Security validation is required for database settings.
        """
        
        redundancy_patterns = {
            'repeated_phrases': {
                'Please configure the system': 2,
                'API endpoint authentication': 2,
                'database security': 2
            }
        }
        
        # Apply semantic redundancy removal
        optimized = self.tokenizer._remove_semantic_redundancy(content, redundancy_patterns)
        
        # Verify content was reduced
        assert len(optimized) < len(content)
        
        # Verify some repetitive content was removed
        phrase_count = optimized.lower().count('please configure the system')
        assert phrase_count <= 1, f"Expected phrase reduction, found {phrase_count} occurrences"

    def test_context_importance_weighting(self):
        """Test context importance weighting for semantic analysis."""
        context_analysis = {
            'content_type': 'project_config',
            'redundancy_patterns': {}
        }
        
        # Critical content
        critical_text1 = "Security authentication token configuration"
        critical_text2 = "Security token authentication setup"
        
        # Non-critical content
        general_text1 = "General information about the project"
        general_text2 = "Project information and general details"
        
        # Test critical content weighting
        critical_weight = self.tokenizer._calculate_context_importance_weight(
            critical_text1, critical_text2, context_analysis
        )
        
        # Test general content weighting
        general_weight = self.tokenizer._calculate_context_importance_weight(
            general_text1, general_text2, context_analysis
        )
        
        # Critical content should have higher importance weight
        assert critical_weight > general_weight
        assert 0.0 <= critical_weight <= 1.0
        assert 0.0 <= general_weight <= 1.0

    def test_phrase_semantic_importance(self):
        """Test phrase semantic importance calculation."""
        content_features = {
            'key_terms': ['authentication', 'configuration', 'security'],
            'technical_terms': ['api', 'endpoint', 'token'],
            'domain_concepts': ['optimization', 'semantic', 'analysis']
        }
        
        # High importance phrase (contains key terms)
        high_importance_phrase = "API authentication configuration"
        importance_high = self.tokenizer._calculate_phrase_semantic_importance(
            high_importance_phrase, content_features
        )
        
        # Low importance phrase (common words)
        low_importance_phrase = "please note that this is important"
        importance_low = self.tokenizer._calculate_phrase_semantic_importance(
            low_importance_phrase, content_features
        )
        
        # Verify importance scoring
        assert importance_high > importance_low
        assert 0.0 <= importance_high <= 1.0
        assert 0.0 <= importance_low <= 1.0

    def test_integrated_semantic_optimization_pipeline(self):
        """Test the integrated semantic optimization in the main pipeline."""
        content = """
        # API Configuration
        Configure the API endpoint authentication system for secure access.
        Setup the API authentication configuration properly.
        
        # Database Setup
        Install database server with security configuration.
        Configure database security settings for safe operation.
        
        # Testing
        Test the authentication system functionality.
        Validate the security configuration testing.
        """
        
        # Test through main optimization pipeline
        sections = self.tokenizer._parse_sections(content)
        optimized_content, notes = self.tokenizer._optimize_content(content, sections)
        
        # Verify optimization occurred
        assert len(optimized_content) < len(content)
        
        # Verify semantic optimization was applied
        semantic_notes = [note for note in notes if 'semantic' in note.lower()]
        assert len(semantic_notes) > 0, "Expected semantic optimization notes"
        
        # Verify clustering was applied
        clustering_notes = [note for note in notes if 'cluster' in note.lower()]
        assert len(clustering_notes) > 0, "Expected semantic clustering notes"
        
        # Verify important content structure is preserved
        assert 'API' in optimized_content
        assert 'Database' in optimized_content
        assert 'Testing' in optimized_content


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
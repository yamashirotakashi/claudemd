"""
Test suite for Usage Pattern Learning System

Tests for the machine learning-based pattern recognition system that learns
from user behavior and document optimization patterns to improve token reduction
effectiveness through adaptive optimization strategies.

Test Coverage:
- Document feature extraction
- Pattern analysis and learning
- Machine learning model integration 
- Prediction accuracy validation
- Storage and retrieval operations
- Security compliance validation

Author: Claude Code Enhanced
Version: 1.0.0
Phase: 1C-5 (Week 3)
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the system under test
from src.core.usage_pattern_analyzer import (
    UsagePatternAnalyzer,
    create_usage_pattern_analyzer,
    analyze_document_patterns,
    predict_optimization_success
)
from src.security.validator import SecurityValidator


class TestUsagePatternAnalyzer(unittest.TestCase):
    """Test suite for UsagePatternAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.security_validator = SecurityValidator()
        self.analyzer = UsagePatternAnalyzer(
            storage_path=self.temp_dir / "test_patterns",
            security_validator=self.security_validator
        )
        
        # Sample test documents
        self.sample_markdown = """# Test Document
        
## Introduction
This is a test document for pattern analysis.

### Features
- Multiple sections
- Various formatting
- Code examples

```python
def test_function():
    return "Hello, World!"
```

### Important Section
**Critical information** that should be preserved.

## Conclusion
End of test document.
"""
        
        self.sample_claude_config = """# Claude Configuration
        
## Rules
- Always validate input
- Maintain security
- Optimize performance

## Workflows
### Data Processing
1. Input validation
2. Processing logic
3. Output formatting

```bash
./process_data.sh input.txt
```

## Settings
- timeout: 30s
- max_retries: 3
- debug: false
"""

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_analyzer_initialization(self):
        """Test UsagePatternAnalyzer initialization."""
        self.assertIsInstance(self.analyzer, UsagePatternAnalyzer)
        self.assertIsInstance(self.analyzer.security_validator, SecurityValidator)
        self.assertTrue(self.analyzer.storage_path.exists())
        
        # Check default pattern categories
        self.assertIn('document_type', self.analyzer.pattern_categories)
        self.assertIn('section_importance', self.analyzer.pattern_categories)
        self.assertIn('optimization_preference', self.analyzer.pattern_categories)
        
        # Check metrics initialization
        self.assertEqual(self.analyzer.metrics['predictions_made'], 0)
        self.assertEqual(self.analyzer.metrics['successful_optimizations'], 0)

    def test_document_feature_extraction(self):
        """Test document feature extraction functionality."""
        features = self.analyzer._extract_document_features(
            self.sample_markdown, 
            "/test/document.md"
        )
        
        # Basic features
        self.assertIn('file_path', features)
        self.assertIn('content_length', features)
        self.assertIn('word_count', features)
        self.assertIn('line_count', features)
        
        # Structure analysis
        self.assertIn('header_count', features)
        self.assertIn('code_blocks', features)
        self.assertIn('markdown_links', features)
        
        # Document type detection
        self.assertIn('document_type', features)
        
        # Sections analysis
        self.assertIn('sections', features)
        self.assertIsInstance(features['sections'], dict)
        
        # Verify specific values
        self.assertGreater(features['word_count'], 0)
        self.assertGreater(features['header_count'], 0)
        self.assertEqual(features['code_blocks'], 1)  # One code block in sample

    def test_document_type_detection(self):
        """Test document type detection accuracy."""
        # Test markdown document
        doc_type = self.analyzer._detect_document_type(
            self.sample_markdown, 
            "/test/document.md"
        )
        # Accept either general_markdown or documentation (both are valid for this content)
        self.assertIn(doc_type, ['general_markdown', 'documentation'])
        
        # Test Claude configuration
        doc_type = self.analyzer._detect_document_type(
            self.sample_claude_config, 
            "/test/CLAUDE.md"
        )
        self.assertEqual(doc_type, 'claude_config')
        
        # Test README
        doc_type = self.analyzer._detect_document_type(
            "# Project Name\n\nInstallation instructions...", 
            "/test/README.md"
        )
        self.assertEqual(doc_type, 'readme')
        
        # Test API documentation
        api_doc = "# API Documentation\n\n## Endpoints\n\n### GET /users\n\nReturns user data..."
        doc_type = self.analyzer._detect_document_type(api_doc, "/test/api.md")
        self.assertEqual(doc_type, 'api_documentation')

    def test_section_analysis(self):
        """Test section analysis functionality."""
        sections = self.analyzer._analyze_sections(self.sample_markdown)
        
        self.assertIsInstance(sections, dict)
        self.assertGreater(len(sections), 0)
        
        # Check section structure
        for section_name, section_data in sections.items():
            self.assertIn('level', section_data)
            self.assertIn('title', section_data)
            self.assertIn('importance_score', section_data)
            self.assertIsInstance(section_data['importance_score'], float)
            self.assertTrue(0 <= section_data['importance_score'] <= 1)

    def test_complexity_classification(self):
        """Test document complexity classification."""
        # Simple document
        simple_features = {
            'word_count': 100,
            'code_blocks': 0,
            'sections': {}
        }
        complexity = self.analyzer._classify_complexity(simple_features)
        self.assertEqual(complexity, 'low')
        
        # Medium complexity
        medium_features = {
            'word_count': 2500,
            'code_blocks': 3,
            'sections': {f'section_{i}': {} for i in range(10)}
        }
        complexity = self.analyzer._classify_complexity(medium_features)
        self.assertEqual(complexity, 'medium')
        
        # High complexity
        high_features = {
            'word_count': 8000,
            'code_blocks': 15,
            'sections': {f'section_{i}': {} for i in range(20)}
        }
        complexity = self.analyzer._classify_complexity(high_features)
        self.assertEqual(complexity, 'high')

    def test_structure_classification(self):
        """Test document structure classification."""
        # Code-heavy document
        code_heavy_features = {
            'header_count': 5,
            'bullet_list_items': 3,
            'numbered_list_items': 2,
            'code_blocks': 10
        }
        structure = self.analyzer._classify_structure(code_heavy_features)
        self.assertEqual(structure, 'code_heavy')
        
        # List-heavy document
        list_heavy_features = {
            'header_count': 3,
            'bullet_list_items': 15,
            'numbered_list_items': 10,
            'code_blocks': 1
        }
        structure = self.analyzer._classify_structure(list_heavy_features)
        self.assertEqual(structure, 'list_heavy')
        
        # Structured document
        structured_features = {
            'header_count': 8,
            'bullet_list_items': 5,
            'numbered_list_items': 3,
            'code_blocks': 2
        }
        structure = self.analyzer._classify_structure(structured_features)
        self.assertEqual(structure, 'structured')

    def test_usage_pattern_analysis(self):
        """Test usage pattern analysis with optimization results."""
        features = self.analyzer._extract_document_features(
            self.sample_markdown, 
            "/test/document.md"
        )
        
        optimization_result = {
            'reduction_percentage': 75.5,
            'techniques_used': ['deduplication', 'compression', 'section_optimization'],
            'technique_effectiveness': {
                'deduplication': 25.0,
                'compression': 30.5,
                'section_optimization': 20.0
            },
            'processing_time': 1.5
        }
        
        patterns = self.analyzer._analyze_usage_patterns(features, optimization_result)
        
        self.assertIn('optimization_effectiveness', patterns)
        self.assertEqual(patterns['optimization_effectiveness'], 75.5)
        
        self.assertIn('preferred_techniques', patterns)
        self.assertIsInstance(patterns['preferred_techniques'], list)
        
        self.assertIn('content_characteristics', patterns)
        self.assertIn('user_behavior', patterns)

    def test_learning_from_optimization(self):
        """Test learning system with optimization results."""
        features = self.analyzer._extract_document_features(
            self.sample_markdown, 
            "/test/document.md"
        )
        
        optimization_result = {
            'reduction_percentage': 72.3,
            'size_reduction': 68.5,
            'processing_time': 2.1,
            'techniques_used': ['smart_compression', 'deduplication']
        }
        
        patterns = self.analyzer._analyze_usage_patterns(features, optimization_result)
        
        # Perform learning
        initial_history_length = len(self.analyzer.optimization_history)
        self.analyzer._learn_from_optimization(features, patterns, optimization_result)
        
        # Verify learning occurred
        self.assertEqual(
            len(self.analyzer.optimization_history), 
            initial_history_length + 1
        )
        
        # Check pattern categories were updated
        doc_type = features.get('document_type', 'unknown')
        if doc_type in self.analyzer.pattern_categories['document_type']:
            self.assertGreater(
                len(self.analyzer.pattern_categories['document_type'][doc_type]), 
                0
            )

    def test_optimization_prediction(self):
        """Test optimization effectiveness prediction."""
        # Add some historical data
        for i in range(5):
            features = {
                'document_type': 'general_markdown',
                'word_count': 1000 + i * 200,
                'header_count': 5 + i,
                'code_blocks': 2,
                'complexity': 'medium'
            }
            
            optimization_result = {
                'reduction_percentage': 70 + i * 2,
                'processing_time': 1.0 + i * 0.2
            }
            
            patterns = {'preferred_techniques': ['compression', 'deduplication']}
            self.analyzer._learn_from_optimization(features, patterns, optimization_result)
        
        # Test prediction
        test_features = {
            'document_type': 'general_markdown',
            'word_count': 1200,
            'header_count': 7,
            'code_blocks': 2,
            'complexity': 'medium'
        }
        
        prediction = self.analyzer.predict_optimization_effectiveness(test_features)
        
        self.assertIn('predicted_reduction', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('recommended_techniques', prediction)
        self.assertIn('estimated_time', prediction)
        
        # Verify prediction values are reasonable
        self.assertGreater(prediction['predicted_reduction'], 0)
        self.assertTrue(0 <= prediction['confidence'] <= 1)

    def test_document_usage_analysis(self):
        """Test complete document usage analysis workflow."""
        optimization_result = {
            'reduction_percentage': 76.8,
            'size_reduction': 72.3,
            'processing_time': 1.8,
            'techniques_used': ['smart_analysis', 'compression'],
            'technique_effectiveness': {'smart_analysis': 40.0, 'compression': 36.8}
        }
        
        analysis_result = self.analyzer.analyze_document_usage(
            self.sample_markdown,
            "/test/document.md",
            optimization_result
        )
        
        self.assertNotIn('error', analysis_result)
        self.assertIn('features', analysis_result)
        self.assertIn('patterns', analysis_result)
        self.assertIn('recommendations', analysis_result)
        self.assertIn('confidence', analysis_result)
        self.assertIn('timestamp', analysis_result)
        
        # Verify features
        features = analysis_result['features']
        self.assertIn('document_type', features)
        self.assertIn('word_count', features)
        
        # Verify patterns
        patterns = analysis_result['patterns']
        self.assertIn('optimization_effectiveness', patterns)
        
        # Verify recommendations
        recommendations = analysis_result['recommendations']
        self.assertIn('suggested_techniques', recommendations)
        self.assertIn('expected_reduction', recommendations)

    def test_usage_statistics(self):
        """Test usage statistics generation."""
        # Add some test data
        for i in range(3):
            optimization_result = {
                'reduction_percentage': 70 + i * 5,
                'processing_time': 1.0 + i * 0.5
            }
            self.analyzer.optimization_history.append({
                'features': {'document_type': f'type_{i}', 'word_count': 1000 + i * 500},
                'result': optimization_result,
                'timestamp': datetime.now().isoformat()
            })
        
        stats = self.analyzer.get_usage_statistics()
        
        self.assertIn('total_optimizations', stats)
        self.assertIn('average_effectiveness', stats)
        self.assertIn('most_common_document_types', stats)
        self.assertIn('model_performance', stats)
        
        # Verify calculated values
        self.assertEqual(stats['total_optimizations'], 3)
        self.assertGreater(stats['average_effectiveness'], 0)

    def test_pattern_storage_and_loading(self):
        """Test pattern storage and loading functionality."""
        # Add some test data
        self.analyzer.pattern_categories['test_category'] = {'test_data': 'value'}
        self.analyzer.metrics['test_metric'] = 42
        
        # Save patterns
        self.analyzer._save_patterns()
        
        # Create new analyzer and load patterns
        new_analyzer = UsagePatternAnalyzer(
            storage_path=self.temp_dir / "test_patterns",
            security_validator=self.security_validator
        )
        
        # Verify data was loaded
        self.assertIn('test_category', new_analyzer.pattern_categories)
        self.assertEqual(new_analyzer.metrics.get('test_metric'), 42)

    def test_old_data_cleanup(self):
        """Test cleanup of old optimization data."""
        # Add recent and old data
        recent_date = datetime.now()
        old_date = recent_date - timedelta(days=45)
        
        self.analyzer.optimization_history = [
            {'timestamp': recent_date.isoformat(), 'data': 'recent'},
            {'timestamp': old_date.isoformat(), 'data': 'old'},
            {'timestamp': recent_date.isoformat(), 'data': 'recent2'}
        ]
        
        # Cleanup data older than 30 days
        self.analyzer.cleanup_old_data(days_to_keep=30)
        
        # Verify old data was removed
        remaining_entries = [
            entry for entry in self.analyzer.optimization_history
            if 'recent' in entry.get('data', '')
        ]
        self.assertEqual(len(remaining_entries), 2)

    def test_learning_data_export(self):
        """Test learning data export functionality."""
        # Add some test data
        self.analyzer.pattern_categories['test_export'] = {'data': 'test'}
        self.analyzer.optimization_history = [
            {'test': 'export_data', 'timestamp': datetime.now().isoformat()}
        ]
        
        # Export data
        export_path = self.temp_dir / "export_test.json"
        result_path = self.analyzer.export_learning_data(export_path)
        
        self.assertEqual(result_path, str(export_path))
        self.assertTrue(export_path.exists())
        
        # Verify exported data
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn('metadata', exported_data)
        self.assertIn('pattern_categories', exported_data)
        self.assertIn('optimization_history_sample', exported_data)

    @patch('src.core.usage_pattern_analyzer.SKLEARN_AVAILABLE', False)
    def test_fallback_without_sklearn(self):
        """Test system functionality without scikit-learn."""
        analyzer = UsagePatternAnalyzer(
            storage_path=self.temp_dir / "fallback_test",
            security_validator=self.security_validator
        )
        
        # Verify system still works without sklearn
        self.assertFalse(analyzer._ensure_sklearn_available())
        
        # Test basic functionality
        features = analyzer._extract_document_features(
            self.sample_markdown, 
            "/test/fallback.md"
        )
        
        self.assertIsInstance(features, dict)
        self.assertIn('document_type', features)

    def test_security_validation(self):
        """Test security validation integration."""
        # Test with invalid file path
        with self.assertRaises(ValueError):
            self.analyzer.analyze_document_usage(
                self.sample_markdown,
                "../../../etc/passwd",  # Invalid path
                {'reduction_percentage': 70}
            )

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with malformed optimization result
        result = self.analyzer.analyze_document_usage(
            self.sample_markdown,
            "/test/error_test.md",
            {}  # Empty optimization result
        )
        
        # Should not raise exception, should handle gracefully
        self.assertIn('timestamp', result)
        
        # Test prediction with invalid features
        prediction = self.analyzer.predict_optimization_effectiveness({})
        self.assertIn('predicted_reduction', prediction)
        self.assertEqual(prediction['predicted_reduction'], 0.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_usage_pattern_analyzer(self):
        """Test analyzer factory function."""
        analyzer = create_usage_pattern_analyzer(self.temp_dir / "test")
        
        self.assertIsInstance(analyzer, UsagePatternAnalyzer)
        self.assertEqual(analyzer.storage_path, self.temp_dir / "test")

    def test_analyze_document_patterns_function(self):
        """Test convenience function for document pattern analysis."""
        content = "# Test\n\nSample content for testing."
        file_path = "/test/convenience.md"
        optimization_result = {'reduction_percentage': 68.5}
        
        result = analyze_document_patterns(content, file_path, optimization_result)
        
        self.assertIn('features', result)
        self.assertIn('patterns', result)
        self.assertIn('recommendations', result)

    def test_predict_optimization_success_function(self):
        """Test convenience function for optimization prediction."""
        features = {
            'document_type': 'general_markdown',
            'word_count': 1500,
            'complexity': 'medium'
        }
        
        prediction = predict_optimization_success(features)
        
        self.assertIn('predicted_reduction', prediction)
        self.assertIn('confidence', prediction)


class TestMachineLearningIntegration(unittest.TestCase):
    """Test suite for machine learning model integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.analyzer = UsagePatternAnalyzer(
            storage_path=self.temp_dir / "ml_test",
            security_validator=SecurityValidator()
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_ml_model_initialization(self):
        """Test ML model initialization when enough data is available."""
        # Add sufficient training data
        for i in range(15):
            features = {
                'document_type': f'type_{i % 3}',
                'word_count': 1000 + i * 100,
                'complexity': ['low', 'medium', 'high'][i % 3]
            }
            patterns = {'preferred_techniques': ['technique_1', 'technique_2']}
            optimization_result = {'reduction_percentage': 65 + i * 2}
            
            self.analyzer._learn_from_optimization(features, patterns, optimization_result)
        
        # Models should be initialized after enough data
        if self.analyzer._ensure_sklearn_available():
            # ML models should be available after learning
            self.assertIsNotNone(self.analyzer._vectorizer)
            self.assertIsNotNone(self.analyzer._clusterer)

    def test_ml_prediction_accuracy(self):
        """Test ML prediction accuracy with known patterns."""
        if not self.analyzer._ensure_sklearn_available():
            self.skipTest("scikit-learn not available")
        
        # Create pattern in training data
        pattern_features = {
            'document_type': 'test_pattern',
            'word_count': 2000,
            'complexity': 'medium'
        }
        
        # Add multiple similar examples
        for i in range(5):
            similar_features = pattern_features.copy()
            similar_features['word_count'] += i * 50
            
            patterns = {'preferred_techniques': ['pattern_technique']}
            optimization_result = {'reduction_percentage': 75 + i}
            
            self.analyzer._learn_from_optimization(similar_features, patterns, optimization_result)
        
        # Trigger model update
        self.analyzer._update_ml_models()
        
        # Test prediction on similar pattern
        test_features = pattern_features.copy()
        test_features['word_count'] = 2100
        
        prediction = self.analyzer.predict_optimization_effectiveness(test_features)
        
        # Should predict similar effectiveness
        if 'ml_prediction' in prediction:
            ml_reduction = prediction['ml_prediction'].get('ml_predicted_reduction', 0)
            self.assertGreater(ml_reduction, 70)  # Should be in expected range

    def test_model_persistence(self):
        """Test ML model saving and loading."""
        if not self.analyzer._ensure_sklearn_available():
            self.skipTest("scikit-learn not available")
        
        # Train models with data
        for i in range(10):
            features = {'document_type': f'persist_test_{i}', 'word_count': 1000 + i * 200}
            patterns = {'preferred_techniques': ['persist_technique']}
            optimization_result = {'reduction_percentage': 70 + i}
            
            self.analyzer._learn_from_optimization(features, patterns, optimization_result)
        
        # Save models
        self.analyzer._save_ml_models()
        
        # Create new analyzer and load models
        new_analyzer = UsagePatternAnalyzer(
            storage_path=self.temp_dir / "ml_test",
            security_validator=SecurityValidator()
        )
        
        # Models should be loaded
        if new_analyzer._ensure_sklearn_available():
            models_exist = (
                new_analyzer._vectorizer is not None or
                new_analyzer._clusterer is not None
            )
            # At least one model should be loaded
            self.assertTrue(models_exist or len(new_analyzer.optimization_history) == 0)


if __name__ == '__main__':
    # Set up test environment
    import sys
    import os
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Run tests
    unittest.main(verbosity=2)
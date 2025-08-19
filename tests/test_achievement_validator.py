"""
Test suite for the Achievement Validation System.

This module contains comprehensive tests for validating that the Claude.md
token reduction system achieves and exceeds the 70% reduction target.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.achievement_validator import (
    AchievementValidator,
    AchievementMetrics,
    BenchmarkResult,
    CertificationReport,
    validate_70_percent_achievement,
    generate_achievement_certificate
)


class TestAchievementValidator:
    """Test cases for the AchievementValidator class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = AchievementValidator()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.TARGET_REDUCTION == 0.70
        assert self.validator.EXCELLENCE_THRESHOLD == 0.75
        assert self.validator.SEMANTIC_THRESHOLD == 0.95
        assert len(self.validator.CERTIFICATION_LEVELS) == 4
        
    def test_create_comprehensive_test_suite(self):
        """Test creation of comprehensive test suite."""
        test_docs = self.validator.create_comprehensive_test_suite(self.temp_dir)
        
        assert len(test_docs) == 5
        assert all(Path(doc).exists() for doc in test_docs)
        
        # Check document types
        doc_names = [Path(doc).name for doc in test_docs]
        expected_names = [
            "technical_documentation.md",
            "api_reference.md",
            "configuration_guide.md",
            "tutorial_content.md",
            "faq_document.md"
        ]
        
        assert all(name in doc_names for name in expected_names)
        
    def test_validate_single_document(self):
        """Test validation of a single document."""
        # Create a test document
        test_doc = Path(self.temp_dir) / "test_doc.md"
        test_content = """# Test Document

## Overview
This is a test document with redundant content.
This section contains duplicate information.
This part has similar content to test optimization.

## Examples
Example 1: Basic example
Example 2: Another similar example
Example 3: Yet another example

## Conclusion
This concludes the test document.
"""
        
        with open(test_doc, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Mock the tokenizer optimization
        with patch.object(self.validator.tokenizer, 'analyze_file') as mock_analyze, \
             patch.object(self.validator.tokenizer, 'optimize_file') as mock_optimize, \
             patch.object(self.validator.tokenizer, 'estimate_tokens') as mock_estimate:
            
            mock_analyze.return_value = {'section_count': 3}
            mock_optimize.return_value = {
                'optimized_content': test_content[:len(test_content)//2],
                'final_tokens': 100,
                'semantic_preservation_score': 0.98,
                'structure_preservation': 0.99,
                'reference_integrity': 1.0,
                'content_coherence': 0.97,
                'formatting_preservation': 0.99
            }
            mock_estimate.return_value = 400
            
            result = self.validator._validate_single_document(str(test_doc))
            
            assert isinstance(result, BenchmarkResult)
            assert result.reduction_percentage == 0.75  # (400-100)/400
            assert result.semantic_score == 0.98
            assert result.original_tokens == 400
            assert result.optimized_tokens == 100
            
    def test_determine_certification_level(self):
        """Test determination of certification levels."""
        assert self.validator._determine_certification_level(0.69) == "BELOW_TARGET"
        assert self.validator._determine_certification_level(0.70) == "TARGET_ACHIEVED"
        assert self.validator._determine_certification_level(0.75) == "EXCELLENCE"
        assert self.validator._determine_certification_level(0.80) == "OUTSTANDING"
        assert self.validator._determine_certification_level(0.85) == "EXCEPTIONAL"
        
    def test_generate_certification_hash(self):
        """Test certification hash generation."""
        # Create mock results
        results = [
            BenchmarkResult(
                document_type="technical",
                file_path="/test1.md",
                original_size=1000,
                optimized_size=250,
                original_tokens=400,
                optimized_tokens=100,
                reduction_percentage=0.75,
                semantic_score=0.98,
                processing_time=1.0,
                quality_metrics={}
            ),
            BenchmarkResult(
                document_type="api",
                file_path="/test2.md",
                original_size=800,
                optimized_size=200,
                original_tokens=300,
                optimized_tokens=75,
                reduction_percentage=0.75,
                semantic_score=0.97,
                processing_time=0.8,
                quality_metrics={}
            )
        ]
        
        hash1 = self.validator._generate_certification_hash(results)
        hash2 = self.validator._generate_certification_hash(results)
        
        assert len(hash1) == 16
        assert hash1 == hash2  # Should be deterministic
        
    def test_analyze_by_document_type(self):
        """Test analysis by document type."""
        results = [
            BenchmarkResult(
                document_type="technical",
                file_path="/test1.md",
                original_size=1000,
                optimized_size=250,
                original_tokens=400,
                optimized_tokens=100,
                reduction_percentage=0.75,
                semantic_score=0.98,
                processing_time=1.0,
                quality_metrics={}
            ),
            BenchmarkResult(
                document_type="technical",
                file_path="/test2.md",
                original_size=800,
                optimized_size=240,
                original_tokens=300,
                optimized_tokens=90,
                reduction_percentage=0.70,
                semantic_score=0.97,
                processing_time=0.8,
                quality_metrics={}
            ),
            BenchmarkResult(
                document_type="api",
                file_path="/test3.md",
                original_size=600,
                optimized_size=120,
                original_tokens=250,
                optimized_tokens=50,
                reduction_percentage=0.80,
                semantic_score=0.99,
                processing_time=0.6,
                quality_metrics={}
            )
        ]
        
        analysis = self.validator._analyze_by_document_type(results)
        
        assert "technical" in analysis
        assert "api" in analysis
        
        # Check technical document analysis
        technical_stats = analysis["technical"]
        assert technical_stats["count"] == 2
        assert technical_stats["average_reduction"] == 0.725  # (0.75 + 0.70) / 2
        assert technical_stats["target_achievement_rate"] == 1.0  # Both >= 0.70
        
        # Check API document analysis
        api_stats = analysis["api"]
        assert api_stats["count"] == 1
        assert api_stats["average_reduction"] == 0.80
        assert api_stats["target_achievement_rate"] == 1.0
        
    @patch('builtins.open', create=True)
    def test_save_certification_report(self, mock_open):
        """Test saving certification report."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Create a mock report
        report = CertificationReport(
            project_name="Test Project",
            target_reduction=0.70,
            achieved_reduction=0.75,
            target_achievement=True,
            certification_level="EXCELLENCE",
            total_files_tested=5,
            average_reduction=0.75,
            min_reduction=0.70,
            max_reduction=0.80,
            semantic_preservation_average=0.98,
            benchmark_results=[],
            validation_timestamp="2024-01-01T00:00:00",
            certification_hash="abc123"
        )
        
        self.validator._save_certification_report(report, self.temp_dir)
        
        mock_open.assert_called_once()
        mock_file.write.assert_called_once()
        
    @patch('builtins.open', create=True)
    def test_generate_achievement_certificate(self, mock_open):
        """Test generation of achievement certificate."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Create a mock report
        report = CertificationReport(
            project_name="Test Project",
            target_reduction=0.70,
            achieved_reduction=0.75,
            target_achievement=True,
            certification_level="EXCELLENCE",
            total_files_tested=5,
            average_reduction=0.75,
            min_reduction=0.70,
            max_reduction=0.80,
            semantic_preservation_average=0.98,
            benchmark_results=[],
            validation_timestamp="2024-01-01T00:00:00",
            certification_hash="abc123"
        )
        
        self.validator._generate_achievement_certificate(report, self.temp_dir)
        
        mock_open.assert_called_once()
        written_content = mock_file.write.call_args[0][0]
        
        # Check certificate content
        assert "ACHIEVEMENT CERTIFICATE" in written_content
        assert "TARGET ACHIEVED" in written_content
        assert "75.00%" in written_content
        assert "EXCELLENCE" in written_content
        assert "abc123" in written_content
        
    @patch('builtins.open', create=True)  
    def test_create_benchmark_summary(self, mock_open):
        """Test creation of benchmark summary."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        results = [
            BenchmarkResult(
                document_type="technical",
                file_path="/test1.md",
                original_size=1000,
                optimized_size=250,
                original_tokens=400,
                optimized_tokens=100,
                reduction_percentage=0.75,
                semantic_score=0.98,
                processing_time=1.0,
                quality_metrics={}
            ),
            BenchmarkResult(
                document_type="api",
                file_path="/test2.md",
                original_size=800,
                optimized_size=200,
                original_tokens=300,
                optimized_tokens=75,
                reduction_percentage=0.75,
                semantic_score=0.97,
                processing_time=0.8,
                quality_metrics={}
            )
        ]
        
        self.validator._create_benchmark_summary(results, self.temp_dir)
        
        mock_open.assert_called_once()
        written_content = mock_file.write.call_args[0][0]
        summary_data = json.loads(written_content)
        
        assert "benchmark_summary" in summary_data
        assert "document_type_analysis" in summary_data
        assert "achievement_validation" in summary_data
        
        # Check statistics
        stats = summary_data["benchmark_summary"]
        assert stats["total_documents"] == 2
        assert stats["reduction_statistics"]["mean"] == 0.75
        assert stats["reduction_statistics"]["min"] == 0.75
        assert stats["reduction_statistics"]["max"] == 0.75
        
    def test_validate_achievement_integration(self):
        """Test the full achievement validation process."""
        # Create test documents
        test_docs = self.validator.create_comprehensive_test_suite(self.temp_dir)
        
        # Mock the validation process
        with patch.object(self.validator, '_validate_single_document') as mock_validate:
            # Mock successful validation results
            mock_results = []
            for i, doc in enumerate(test_docs):
                result = BenchmarkResult(
                    document_type=f"type_{i}",
                    file_path=doc,
                    original_size=1000,
                    optimized_size=250,
                    original_tokens=400,
                    optimized_tokens=100 - i*5,  # Vary results
                    reduction_percentage=0.75 + i*0.01,  # 75%, 76%, 77%, etc.
                    semantic_score=0.98,
                    processing_time=1.0,
                    quality_metrics={}
                )
                mock_results.append(result)
            
            mock_validate.side_effect = mock_results
            
            # Run validation
            report = self.validator.validate_achievement(test_docs, self.temp_dir)
            
            # Check results
            assert isinstance(report, CertificationReport)
            assert report.target_achievement == True
            assert report.total_files_tested == 5
            assert report.achieved_reduction >= 0.70
            assert report.certification_level in ["TARGET_ACHIEVED", "EXCELLENCE", "OUTSTANDING", "EXCEPTIONAL"]
            
    def test_file_security_validation(self):
        """Test file security validation."""
        # Test with invalid file path
        invalid_path = "/invalid/path/test.md"
        
        with pytest.raises((FileNotFoundError, ValueError)):
            self.validator._validate_single_document(invalid_path)
            
    def test_error_handling_in_validation(self):
        """Test error handling during validation."""
        # Create a test document
        test_doc = Path(self.temp_dir) / "error_test.md"
        with open(test_doc, 'w', encoding='utf-8') as f:
            f.write("# Test")
        
        # Mock tokenizer to raise an exception
        with patch.object(self.validator.tokenizer, 'analyze_file', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                self.validator._validate_single_document(str(test_doc))


class TestConvenienceFunctions:
    """Test convenience functions for achievement validation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('src.core.achievement_validator.AchievementValidator')
    def test_validate_70_percent_achievement_success(self, mock_validator_class):
        """Test successful 70% achievement validation."""
        # Mock validator and report
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_report = Mock()
        mock_report.target_achievement = True
        mock_validator.run_comprehensive_achievement_validation.return_value = mock_report
        
        result = validate_70_percent_achievement(output_dir=self.temp_dir)
        
        assert result == True
        mock_validator.run_comprehensive_achievement_validation.assert_called_once_with(self.temp_dir)
        
    @patch('src.core.achievement_validator.AchievementValidator')
    def test_validate_70_percent_achievement_failure(self, mock_validator_class):
        """Test failed 70% achievement validation."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_report = Mock()
        mock_report.target_achievement = False
        mock_validator.run_comprehensive_achievement_validation.return_value = mock_report
        
        result = validate_70_percent_achievement(output_dir=self.temp_dir)
        
        assert result == False
        
    @patch('src.core.achievement_validator.AchievementValidator')
    def test_generate_achievement_certificate_success(self, mock_validator_class):
        """Test successful certificate generation."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_report = Mock()
        mock_validator.run_comprehensive_achievement_validation.return_value = mock_report
        
        result = generate_achievement_certificate(output_dir=self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / "ACHIEVEMENT_CERTIFICATE.md")
        assert result == expected_path
        

class TestDataClasses:
    """Test the dataclasses used in achievement validation."""
    
    def test_achievement_metrics_creation(self):
        """Test AchievementMetrics dataclass."""
        metrics = AchievementMetrics(
            original_tokens=1000,
            optimized_tokens=250,
            reduction_percentage=0.75,
            compression_ratio=4.0,
            processing_time=1.5,
            file_size_reduction=0.80,
            semantic_preservation_score=0.98,
            target_achievement=True,
            certification_grade="EXCELLENCE"
        )
        
        assert metrics.original_tokens == 1000
        assert metrics.optimized_tokens == 250
        assert metrics.reduction_percentage == 0.75
        assert metrics.target_achievement == True
        assert metrics.certification_grade == "EXCELLENCE"
        
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult dataclass."""
        result = BenchmarkResult(
            document_type="technical",
            file_path="/test.md",
            original_size=2000,
            optimized_size=500,
            original_tokens=800,
            optimized_tokens=200,
            reduction_percentage=0.75,
            semantic_score=0.98,
            processing_time=2.0,
            quality_metrics={"structure": 0.99, "coherence": 0.97}
        )
        
        assert result.document_type == "technical"
        assert result.original_tokens == 800
        assert result.optimized_tokens == 200
        assert result.reduction_percentage == 0.75
        assert result.quality_metrics["structure"] == 0.99
        
    def test_certification_report_creation(self):
        """Test CertificationReport dataclass."""
        report = CertificationReport(
            project_name="Claude.md Token Reduction",
            target_reduction=0.70,
            achieved_reduction=0.76,
            target_achievement=True,
            certification_level="EXCELLENCE",
            total_files_tested=10,
            average_reduction=0.76,
            min_reduction=0.70,
            max_reduction=0.82,
            semantic_preservation_average=0.98,
            benchmark_results=[],
            validation_timestamp="2024-01-01T00:00:00Z",
            certification_hash="abc123def456"
        )
        
        assert report.project_name == "Claude.md Token Reduction"
        assert report.target_achievement == True
        assert report.certification_level == "EXCELLENCE"
        assert report.achieved_reduction == 0.76
        assert report.certification_hash == "abc123def456"


class TestPerformanceBenchmarking:
    """Test performance benchmarking capabilities."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = AchievementValidator()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_benchmark_different_document_types(self):
        """Test benchmarking across different document types."""
        # This would test real performance if we had actual documents
        test_docs = self.validator.create_comprehensive_test_suite(self.temp_dir)
        
        # Verify different document types are created
        doc_names = [Path(doc).name for doc in test_docs]
        
        # Each document should have different characteristics for testing
        for doc_path in test_docs:
            content = Path(doc_path).read_text(encoding='utf-8')
            assert len(content) > 1000  # Ensure substantial content
            assert '##' in content  # Ensure structured content
            
    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics."""
        # Create mock results with varying performance
        results = [
            BenchmarkResult(
                document_type="technical",
                file_path="/test1.md",
                original_size=1000,
                optimized_size=200,
                original_tokens=400,
                optimized_tokens=80,
                reduction_percentage=0.80,
                semantic_score=0.98,
                processing_time=0.5,
                quality_metrics={}
            ),
            BenchmarkResult(
                document_type="api",
                file_path="/test2.md",
                original_size=1500,
                optimized_size=450,
                original_tokens=600,
                optimized_tokens=180,
                reduction_percentage=0.70,
                semantic_score=0.95,
                processing_time=0.8,
                quality_metrics={}
            ),
            BenchmarkResult(
                document_type="tutorial",
                file_path="/test3.md",
                original_size=2000,
                optimized_size=400,
                original_tokens=800,
                optimized_tokens=160,
                reduction_percentage=0.80,
                semantic_score=0.99,
                processing_time=1.2,
                quality_metrics={}
            )
        ]
        
        # Calculate aggregate metrics manually to verify
        reductions = [r.reduction_percentage for r in results]
        semantic_scores = [r.semantic_score for r in results]
        processing_times = [r.processing_time for r in results]
        
        # Check average calculations
        avg_reduction = sum(reductions) / len(reductions)
        avg_semantic = sum(semantic_scores) / len(semantic_scores)
        avg_processing = sum(processing_times) / len(processing_times)
        
        assert avg_reduction == (0.80 + 0.70 + 0.80) / 3
        assert avg_semantic == (0.98 + 0.95 + 0.99) / 3
        assert avg_processing == (0.5 + 0.8 + 1.2) / 3
        
        # Verify target achievement
        target_achievers = sum(1 for r in reductions if r >= 0.70)
        assert target_achievers == 3  # All should achieve target
        
    def test_certification_package_creation(self):
        """Test creation of certification package."""
        # Create mock report
        report = CertificationReport(
            project_name="Test Project",
            target_reduction=0.70,
            achieved_reduction=0.75,
            target_achievement=True,
            certification_level="EXCELLENCE",
            total_files_tested=5,
            average_reduction=0.75,
            min_reduction=0.70,
            max_reduction=0.80,
            semantic_preservation_average=0.98,
            benchmark_results=[],
            validation_timestamp="2024-01-01T00:00:00",
            certification_hash="abc123"
        )
        
        # Create required files for packaging
        report_file = Path(self.temp_dir) / "achievement_certification_report.json"
        cert_file = Path(self.temp_dir) / "ACHIEVEMENT_CERTIFICATE.md" 
        summary_file = Path(self.temp_dir) / "benchmark_summary.json"
        
        with open(report_file, 'w') as f:
            json.dump({'test': 'data'}, f)
        with open(cert_file, 'w') as f:
            f.write("# Test Certificate")
        with open(summary_file, 'w') as f:
            json.dump({'summary': 'data'}, f)
        
        # Test package creation
        self.validator._create_certification_package(report, self.temp_dir)
        
        package_path = Path(self.temp_dir) / "ACHIEVEMENT_CERTIFICATION_PACKAGE.zip"
        assert package_path.exists()
        assert package_path.stat().st_size > 0
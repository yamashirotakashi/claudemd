"""
Usage Pattern Learning System for Claude.md Token Reduction

This module implements machine learning-based pattern recognition to analyze user
behavior, document types, and optimization preferences to improve token reduction
effectiveness through adaptive optimization strategies.

Author: Claude Code Enhanced
Version: 1.0.0
Phase: 1C-5 (Week 3)
Target: 3-8% additional token reduction through learned optimizations
"""

import json
import logging
import pickle
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations when scikit-learn is not available
    
from ..security.validator import SecurityValidator


class UsagePatternAnalyzer:
    """
    Intelligent system that learns from user's document editing and usage patterns
    to optimize token reduction strategies through machine learning algorithms.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, security_validator: Optional[SecurityValidator] = None):
        """Initialize the Usage Pattern Analyzer with ML capabilities."""
        self.logger = logging.getLogger(__name__)
        self.security_validator = security_validator or SecurityValidator()
        
        # Storage configuration
        self.storage_path = storage_path or Path.cwd() / ".claudemd_patterns"
        self.storage_path.mkdir(exist_ok=True)
        
        # Pattern tracking
        self.usage_patterns = defaultdict(list)
        self.document_features = {}
        self.optimization_history = []
        self.user_preferences = {}
        
        # ML Models (lazy initialization)
        self._vectorizer = None
        self._clusterer = None
        self._scaler = None
        
        # Pattern categories
        self.pattern_categories = {
            'document_type': {},
            'section_importance': {},
            'optimization_preference': {},
            'content_pattern': {},
            'editing_behavior': {}
        }
        
        # Performance metrics
        self.metrics = {
            'predictions_made': 0,
            'successful_optimizations': 0,
            'pattern_accuracy': 0.0,
            'learning_iterations': 0
        }
        
        # Load existing patterns
        self._load_patterns()
        
        self.logger.info("UsagePatternAnalyzer initialized with ML capabilities")

    def _ensure_sklearn_available(self) -> bool:
        """Ensure scikit-learn is available for ML operations."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available, using fallback implementations")
            return False
        return True

    def analyze_document_usage(self, 
                             content: str, 
                             file_path: str,
                             optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze document usage patterns and learn from optimization results.
        
        Args:
            content: Original document content
            file_path: Path to the document
            optimization_result: Results from token optimization
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not self.security_validator.validate_file_path(file_path):
            raise ValueError(f"Invalid file path: {file_path}")
            
        try:
            # Extract document features
            features = self._extract_document_features(content, file_path)
            
            # Analyze usage patterns
            patterns = self._analyze_usage_patterns(features, optimization_result)
            
            # Learn from the results
            self._learn_from_optimization(features, patterns, optimization_result)
            
            # Update pattern categories
            self._update_pattern_categories(features, patterns)
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(features)
            
            analysis_result = {
                'features': features,
                'patterns': patterns,
                'recommendations': recommendations,
                'confidence': self._calculate_confidence(features),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store the analysis
            self._store_analysis_result(file_path, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing document usage: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _extract_document_features(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive features from document content."""
        features = {
            'file_path': file_path,
            'content_length': len(content),
            'line_count': len(content.splitlines()),
            'word_count': len(content.split()),
            'char_count': len(content),
            'timestamp': datetime.now().isoformat()
        }
        
        # Content structure analysis
        features.update(self._analyze_content_structure(content))
        
        # Document type detection
        features['document_type'] = self._detect_document_type(content, file_path)
        
        # Section analysis
        features['sections'] = self._analyze_sections(content)
        
        # Code and formatting analysis
        features.update(self._analyze_code_and_formatting(content))
        
        # Complexity metrics
        features.update(self._calculate_complexity_metrics(content))
        
        return features

    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structural elements of the content."""
        structure = {}
        
        # Headers analysis
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        structure['header_count'] = len(headers)
        structure['header_levels'] = [len(h.split()[0]) for h in re.findall(r'^(#+)', content, re.MULTILINE)]
        
        # Lists analysis
        bullet_lists = re.findall(r'^[-*+]\s+', content, re.MULTILINE)
        numbered_lists = re.findall(r'^\d+\.\s+', content, re.MULTILINE)
        structure['bullet_list_items'] = len(bullet_lists)
        structure['numbered_list_items'] = len(numbered_lists)
        
        # Links analysis
        markdown_links = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
        url_links = re.findall(r'https?://[^\s]+', content)
        structure['markdown_links'] = len(markdown_links)
        structure['url_links'] = len(url_links)
        
        # Code blocks
        code_blocks = re.findall(r'```[^`]*```', content, re.DOTALL)
        inline_code = re.findall(r'`[^`]+`', content)
        structure['code_blocks'] = len(code_blocks)
        structure['inline_code'] = len(inline_code)
        
        # Emphasis
        bold_text = re.findall(r'\*\*[^*]+\*\*', content)
        italic_text = re.findall(r'\*[^*]+\*', content)
        structure['bold_elements'] = len(bold_text)
        structure['italic_elements'] = len(italic_text)
        
        return structure

    def _detect_document_type(self, content: str, file_path: str) -> str:
        """Detect the type of document based on content and file path."""
        file_name = Path(file_path).name.lower()
        
        # Check file name patterns
        if 'readme' in file_name:
            return 'readme'
        elif 'claude' in file_name:
            return 'claude_config'
        elif 'config' in file_name:
            return 'configuration'
        elif 'spec' in file_name or 'specification' in file_name:
            return 'specification'
        elif 'doc' in file_name or 'documentation' in file_name:
            return 'documentation'
        
        # Check content patterns
        content_lower = content.lower()
        
        if 'api' in content_lower and ('endpoint' in content_lower or 'request' in content_lower):
            return 'api_documentation'
        elif 'test' in content_lower and ('assert' in content_lower or 'expect' in content_lower):
            return 'test_documentation'
        elif 'install' in content_lower and ('pip' in content_lower or 'npm' in content_lower):
            return 'installation_guide'
        elif '```' in content and 'python' in content_lower:
            return 'technical_guide'
        elif 'rule' in content_lower and 'must' in content_lower:
            return 'policy_document'
        
        return 'general_markdown'

    def _analyze_sections(self, content: str) -> Dict[str, Any]:
        """Analyze sections and their importance."""
        sections = {}
        current_section = None
        
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Extract section title and level
                level = len(line.split()[0]) if line.split() else 1
                title = line.strip('#').strip()
                
                if current_section:
                    sections[current_section]['end_line'] = i - 1
                
                current_section = title
                sections[current_section] = {
                    'level': level,
                    'start_line': i,
                    'title': title,
                    'importance_score': self._calculate_section_importance(title, content)
                }
        
        # Close the last section
        if current_section:
            sections[current_section]['end_line'] = len(lines) - 1
        
        return sections

    def _calculate_section_importance(self, title: str, content: str) -> float:
        """Calculate the importance score of a section based on title and content."""
        important_keywords = [
            'important', 'critical', 'essential', 'required', 'mandatory',
            'security', 'safety', 'warning', 'caution', 'note', 'attention'
        ]
        
        title_lower = title.lower()
        importance_score = 0.5  # Base score
        
        # Check for important keywords in title
        for keyword in important_keywords:
            if keyword in title_lower:
                importance_score += 0.2
        
        # Check section length (longer sections might be more important)
        section_content = self._extract_section_content(title, content)
        if section_content:
            word_count = len(section_content.split())
            if word_count > 100:
                importance_score += 0.1
            elif word_count > 50:
                importance_score += 0.05
        
        return min(1.0, importance_score)

    def _extract_section_content(self, section_title: str, content: str) -> str:
        """Extract content of a specific section."""
        lines = content.splitlines()
        section_lines = []
        in_section = False
        
        for line in lines:
            if line.strip().startswith('#'):
                if section_title.lower() in line.lower():
                    in_section = True
                    continue
                elif in_section:
                    break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines)

    def _analyze_code_and_formatting(self, content: str) -> Dict[str, Any]:
        """Analyze code elements and formatting patterns."""
        analysis = {}
        
        # Code language detection
        code_blocks = re.findall(r'```(\w+)?\n([^`]*)\n```', content, re.DOTALL)
        languages = [lang for lang, _ in code_blocks if lang]
        analysis['code_languages'] = list(set(languages))
        analysis['code_block_sizes'] = [len(code) for _, code in code_blocks]
        
        # Formatting density
        total_chars = len(content)
        formatting_chars = len(re.findall(r'[*_`#\[\]()]', content))
        analysis['formatting_density'] = formatting_chars / total_chars if total_chars > 0 else 0
        
        # Table analysis
        tables = re.findall(r'\|.*\|', content)
        analysis['table_rows'] = len(tables)
        
        return analysis

    def _calculate_complexity_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate various complexity metrics for the document."""
        metrics = {}
        
        lines = content.splitlines()
        words = content.split()
        
        # Basic metrics
        metrics['avg_line_length'] = sum(len(line) for line in lines) / len(lines) if lines else 0
        metrics['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
        
        # Nesting complexity (based on indentation)
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
        
        metrics['max_indent_level'] = max(indent_levels) if indent_levels else 0
        metrics['avg_indent_level'] = sum(indent_levels) / len(indent_levels) if indent_levels else 0
        
        # Reference complexity
        references = re.findall(r'\[([^\]]+)\]', content)
        metrics['reference_count'] = len(references)
        metrics['unique_references'] = len(set(references))
        
        return metrics

    def _analyze_usage_patterns(self, features: Dict[str, Any], optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in document usage and optimization."""
        patterns = {
            'optimization_effectiveness': optimization_result.get('reduction_percentage', 0),
            'preferred_techniques': [],
            'content_characteristics': {},
            'user_behavior': {}
        }
        
        # Analyze which optimization techniques were most effective
        if 'techniques_used' in optimization_result:
            techniques = optimization_result['techniques_used']
            effectiveness = optimization_result.get('technique_effectiveness', {})
            
            # Sort techniques by effectiveness
            sorted_techniques = sorted(
                techniques,
                key=lambda t: effectiveness.get(t, 0),
                reverse=True
            )
            patterns['preferred_techniques'] = sorted_techniques[:3]
        
        # Analyze content characteristics that led to good optimization
        if optimization_result.get('reduction_percentage', 0) > 70:  # Good optimization
            patterns['content_characteristics'] = {
                'document_type': features.get('document_type'),
                'complexity_level': self._classify_complexity(features),
                'structure_type': self._classify_structure(features)
            }
        
        # Track user behavior patterns
        patterns['user_behavior'] = {
            'optimization_frequency': self._calculate_optimization_frequency(),
            'preferred_file_types': self._get_preferred_file_types(),
            'common_optimization_times': self._get_common_optimization_times()
        }
        
        return patterns

    def _classify_complexity(self, features: Dict[str, Any]) -> str:
        """Classify document complexity based on features."""
        word_count = features.get('word_count', 0)
        code_blocks = features.get('code_blocks', 0)
        sections = len(features.get('sections', {}))
        
        complexity_score = 0
        
        if word_count > 5000:
            complexity_score += 2
        elif word_count > 2000:
            complexity_score += 1
        
        if code_blocks > 10:
            complexity_score += 2
        elif code_blocks > 5:
            complexity_score += 1
        
        if sections > 15:
            complexity_score += 2
        elif sections > 8:
            complexity_score += 1
        
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _classify_structure(self, features: Dict[str, Any]) -> str:
        """Classify document structure type."""
        header_count = features.get('header_count', 0)
        list_items = features.get('bullet_list_items', 0) + features.get('numbered_list_items', 0)
        code_blocks = features.get('code_blocks', 0)
        
        if code_blocks > header_count:
            return 'code_heavy'
        elif list_items > header_count * 2:
            return 'list_heavy'
        elif header_count > 5:
            return 'structured'
        else:
            return 'narrative'

    def _learn_from_optimization(self, 
                                features: Dict[str, Any],
                                patterns: Dict[str, Any], 
                                optimization_result: Dict[str, Any]) -> None:
        """Learn from optimization results to improve future predictions."""
        learning_data = {
            'features': features,
            'patterns': patterns,
            'result': optimization_result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(learning_data)
        
        # Update ML models if enough data is available
        if len(self.optimization_history) >= 10 and self._ensure_sklearn_available():
            self._update_ml_models()
        
        # Update pattern categories
        doc_type = features.get('document_type', 'unknown')
        effectiveness = optimization_result.get('reduction_percentage', 0)
        
        if doc_type not in self.pattern_categories['document_type']:
            self.pattern_categories['document_type'][doc_type] = []
        
        self.pattern_categories['document_type'][doc_type].append({
            'effectiveness': effectiveness,
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
        
        self.metrics['learning_iterations'] += 1
        self._save_patterns()

    def _update_ml_models(self) -> None:
        """Update machine learning models with new data."""
        if not self._ensure_sklearn_available():
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 5:  # Need minimum samples
                return
            
            # Update vectorizer
            if self._vectorizer is None:
                self._vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            
            # Update clustering model
            if self._clusterer is None:
                n_clusters = min(5, len(X) // 2)  # Adaptive cluster count
                self._clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            # Fit models
            X_vectorized = self._vectorizer.fit_transform([str(x) for x in X])
            self._clusterer.fit(X_vectorized.toarray())
            
            # Update scaler
            if self._scaler is None:
                self._scaler = StandardScaler()
            
            numeric_features = self._extract_numeric_features(X)
            if len(numeric_features) > 0:
                self._scaler.fit(numeric_features)
            
            self.logger.info("ML models updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating ML models: {e}")

    def _prepare_training_data(self) -> Tuple[List[Dict], List[float]]:
        """Prepare training data from optimization history."""
        X = []  # Features
        y = []  # Target (optimization effectiveness)
        
        for entry in self.optimization_history:
            features = entry['features']
            effectiveness = entry['result'].get('reduction_percentage', 0)
            
            # Create feature vector
            feature_vector = {
                'word_count': features.get('word_count', 0),
                'header_count': features.get('header_count', 0),
                'code_blocks': features.get('code_blocks', 0),
                'complexity': self._classify_complexity(features),
                'document_type': features.get('document_type', 'unknown'),
                'formatting_density': features.get('formatting_density', 0)
            }
            
            X.append(feature_vector)
            y.append(effectiveness)
        
        return X, y

    def _extract_numeric_features(self, feature_dicts: List[Dict]) -> List[List[float]]:
        """Extract numeric features for scaling."""
        numeric_features = []
        
        for features in feature_dicts:
            numeric_row = [
                features.get('word_count', 0),
                features.get('header_count', 0),
                features.get('code_blocks', 0),
                features.get('formatting_density', 0)
            ]
            numeric_features.append(numeric_row)
        
        return numeric_features

    def _update_pattern_categories(self, features: Dict[str, Any], patterns: Dict[str, Any]) -> None:
        """Update pattern categories with new observations."""
        # Update section importance patterns
        sections = features.get('sections', {})
        for section_name, section_data in sections.items():
            importance = section_data.get('importance_score', 0.5)
            
            if section_name not in self.pattern_categories['section_importance']:
                self.pattern_categories['section_importance'][section_name] = []
            
            self.pattern_categories['section_importance'][section_name].append(importance)
        
        # Update optimization preference patterns
        preferred_techniques = patterns.get('preferred_techniques', [])
        for technique in preferred_techniques:
            if technique not in self.pattern_categories['optimization_preference']:
                self.pattern_categories['optimization_preference'][technique] = 0
            self.pattern_categories['optimization_preference'][technique] += 1

    def _generate_optimization_recommendations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations based on learned patterns."""
        recommendations = {
            'suggested_techniques': [],
            'priority_sections': [],
            'expected_reduction': 0.0,
            'confidence_level': 0.0
        }
        
        try:
            doc_type = features.get('document_type', 'unknown')
            
            # Get historical data for similar documents
            similar_docs = self._find_similar_documents(features)
            
            if similar_docs:
                # Recommend techniques that worked well for similar documents
                technique_scores = defaultdict(float)
                total_effectiveness = 0
                
                for doc in similar_docs:
                    patterns = doc.get('patterns', {})
                    effectiveness = doc.get('result', {}).get('reduction_percentage', 0)
                    techniques = patterns.get('preferred_techniques', [])
                    
                    total_effectiveness += effectiveness
                    
                    for i, technique in enumerate(techniques):
                        # Weight techniques by their position (first is most effective)
                        weight = 1.0 / (i + 1)
                        technique_scores[technique] += weight * effectiveness
                
                # Sort techniques by score
                sorted_techniques = sorted(
                    technique_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                recommendations['suggested_techniques'] = [t[0] for t in sorted_techniques[:3]]
                recommendations['expected_reduction'] = total_effectiveness / len(similar_docs)
            
            # Use ML predictions if available
            if self._vectorizer and self._clusterer and self._ensure_sklearn_available():
                ml_recommendations = self._get_ml_recommendations(features)
                recommendations.update(ml_recommendations)
            
            # Calculate confidence based on amount of similar data
            recommendations['confidence_level'] = min(1.0, len(similar_docs) * 0.2)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations['error'] = str(e)
        
        return recommendations

    def _find_similar_documents(self, features: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to the given features."""
        similar_docs = []
        target_doc_type = features.get('document_type', 'unknown')
        target_complexity = self._classify_complexity(features)
        target_word_count = features.get('word_count', 0)
        
        for entry in self.optimization_history:
            entry_features = entry['features']
            
            # Calculate similarity score
            similarity_score = 0.0
            
            # Document type similarity
            if entry_features.get('document_type') == target_doc_type:
                similarity_score += 0.4
            
            # Complexity similarity
            if self._classify_complexity(entry_features) == target_complexity:
                similarity_score += 0.3
            
            # Word count similarity
            entry_word_count = entry_features.get('word_count', 0)
            word_count_diff = abs(target_word_count - entry_word_count) / max(target_word_count, 1)
            word_count_similarity = max(0, 1 - word_count_diff)
            similarity_score += 0.3 * word_count_similarity
            
            if similarity_score > 0.5:  # Threshold for similarity
                entry['similarity_score'] = similarity_score
                similar_docs.append(entry)
        
        # Sort by similarity and return top matches
        similar_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_docs[:limit]

    def _get_ml_recommendations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations using machine learning models."""
        ml_recommendations = {}
        
        try:
            # Prepare feature vector
            feature_text = str(features)
            feature_vector = self._vectorizer.transform([feature_text])
            
            # Get cluster prediction
            cluster = self._clusterer.predict(feature_vector.toarray())[0]
            
            # Find documents in the same cluster
            cluster_docs = []
            for entry in self.optimization_history:
                entry_vector = self._vectorizer.transform([str(entry['features'])])
                entry_cluster = self._clusterer.predict(entry_vector.toarray())[0]
                
                if entry_cluster == cluster:
                    cluster_docs.append(entry)
            
            if cluster_docs:
                # Calculate average effectiveness for this cluster
                avg_effectiveness = sum(
                    doc['result'].get('reduction_percentage', 0)
                    for doc in cluster_docs
                ) / len(cluster_docs)
                
                ml_recommendations['ml_expected_reduction'] = avg_effectiveness
                ml_recommendations['ml_confidence'] = min(1.0, len(cluster_docs) * 0.1)
                ml_recommendations['cluster_id'] = int(cluster)
        
        except Exception as e:
            self.logger.error(f"Error getting ML recommendations: {e}")
            ml_recommendations['ml_error'] = str(e)
        
        return ml_recommendations

    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence level for predictions."""
        base_confidence = 0.5
        
        # Increase confidence based on amount of historical data
        doc_type = features.get('document_type', 'unknown')
        if doc_type in self.pattern_categories['document_type']:
            historical_count = len(self.pattern_categories['document_type'][doc_type])
            confidence_boost = min(0.4, historical_count * 0.05)
            base_confidence += confidence_boost
        
        # Increase confidence if ML models are available and trained
        if (self._vectorizer and self._clusterer and 
            len(self.optimization_history) >= 10):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _calculate_optimization_frequency(self) -> float:
        """Calculate how frequently the user optimizes documents."""
        if len(self.optimization_history) < 2:
            return 0.0
        
        # Calculate average time between optimizations
        timestamps = [
            datetime.fromisoformat(entry['timestamp'])
            for entry in self.optimization_history
            if 'timestamp' in entry
        ]
        
        if len(timestamps) < 2:
            return 0.0
        
        timestamps.sort()
        time_diffs = []
        
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
            time_diffs.append(diff)
        
        avg_hours_between = sum(time_diffs) / len(time_diffs)
        
        # Convert to frequency (optimizations per day)
        frequency = 24 / avg_hours_between if avg_hours_between > 0 else 0
        
        return min(10.0, frequency)  # Cap at 10 per day

    def _get_preferred_file_types(self) -> List[str]:
        """Get the most commonly optimized file types."""
        file_type_counts = defaultdict(int)
        
        for entry in self.optimization_history:
            file_path = entry.get('features', {}).get('file_path', '')
            file_type = Path(file_path).suffix.lower() if file_path else 'unknown'
            file_type_counts[file_type] += 1
        
        # Sort by count and return top types
        sorted_types = sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True)
        return [file_type for file_type, _ in sorted_types[:3]]

    def _get_common_optimization_times(self) -> List[str]:
        """Get the most common times when optimizations occur."""
        hour_counts = defaultdict(int)
        
        for entry in self.optimization_history:
            timestamp_str = entry.get('timestamp', '')
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    hour = dt.hour
                    hour_counts[hour] += 1
                except Exception:
                    continue
        
        # Convert to time ranges
        common_times = []
        for hour, count in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            time_range = f"{hour:02d}:00-{(hour+1):02d}:00"
            common_times.append(time_range)
        
        return common_times

    def predict_optimization_effectiveness(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the effectiveness of optimization for given document features.
        
        Args:
            features: Document features extracted from content analysis
            
        Returns:
            Dictionary with prediction results and confidence levels
        """
        try:
            prediction_result = {
                'predicted_reduction': 0.0,
                'confidence': 0.0,
                'recommended_techniques': [],
                'estimated_time': 0,
                'risk_factors': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Get similar documents for baseline prediction
            similar_docs = self._find_similar_documents(features, limit=10)
            
            if similar_docs:
                # Calculate weighted average effectiveness
                total_weight = 0
                weighted_reduction = 0
                
                for doc in similar_docs:
                    weight = doc.get('similarity_score', 0.5)
                    reduction = doc.get('result', {}).get('reduction_percentage', 0)
                    weighted_reduction += weight * reduction
                    total_weight += weight
                
                if total_weight > 0:
                    prediction_result['predicted_reduction'] = weighted_reduction / total_weight
                
                # Extract common techniques
                technique_votes = defaultdict(int)
                for doc in similar_docs:
                    techniques = doc.get('patterns', {}).get('preferred_techniques', [])
                    for technique in techniques[:2]:  # Top 2 techniques
                        technique_votes[technique] += 1
                
                # Sort techniques by votes
                sorted_techniques = sorted(technique_votes.items(), key=lambda x: x[1], reverse=True)
                prediction_result['recommended_techniques'] = [t[0] for t in sorted_techniques[:3]]
                
                # Calculate confidence based on similarity and amount of data
                avg_similarity = sum(doc.get('similarity_score', 0) for doc in similar_docs) / len(similar_docs)
                data_confidence = min(1.0, len(similar_docs) * 0.1)
                prediction_result['confidence'] = (avg_similarity + data_confidence) / 2
            
            # Use ML prediction if available
            if (self._vectorizer and self._clusterer and 
                self._ensure_sklearn_available() and 
                len(self.optimization_history) >= 10):
                
                ml_prediction = self._get_ml_prediction(features)
                
                # Combine statistical and ML predictions
                if ml_prediction.get('ml_predicted_reduction', 0) > 0:
                    stat_reduction = prediction_result['predicted_reduction']
                    ml_reduction = ml_prediction['ml_predicted_reduction']
                    ml_confidence = ml_prediction.get('ml_confidence', 0)
                    
                    # Weighted average of predictions
                    combined_reduction = (
                        stat_reduction * (1 - ml_confidence) + 
                        ml_reduction * ml_confidence
                    )
                    prediction_result['predicted_reduction'] = combined_reduction
                    prediction_result['ml_prediction'] = ml_prediction
            
            # Estimate optimization time based on document complexity
            complexity = self._classify_complexity(features)
            word_count = features.get('word_count', 0)
            
            if complexity == 'high':
                base_time = 10
            elif complexity == 'medium':
                base_time = 5
            else:
                base_time = 2
            
            # Adjust for document size
            size_multiplier = min(3.0, word_count / 1000)
            prediction_result['estimated_time'] = int(base_time * size_multiplier)
            
            # Identify risk factors
            risk_factors = []
            if word_count > 10000:
                risk_factors.append("Very large document - may require multiple optimization passes")
            if features.get('code_blocks', 0) > 20:
                risk_factors.append("High code content - preserve functionality carefully")
            if len(features.get('sections', {})) > 20:
                risk_factors.append("Complex structure - manual review recommended")
            
            prediction_result['risk_factors'] = risk_factors
            
            # Update metrics
            self.metrics['predictions_made'] += 1
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting optimization effectiveness: {e}")
            return {
                'error': str(e),
                'predicted_reduction': 0.0,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _get_ml_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction using trained ML models."""
        ml_prediction = {}
        
        try:
            # Prepare feature vector
            feature_text = str(features)
            feature_vector = self._vectorizer.transform([feature_text])
            
            # Get cluster assignment
            cluster = self._clusterer.predict(feature_vector.toarray())[0]
            
            # Get effectiveness statistics for this cluster
            cluster_effectiveness = []
            for entry in self.optimization_history:
                entry_vector = self._vectorizer.transform([str(entry['features'])])
                entry_cluster = self._clusterer.predict(entry_vector.toarray())[0]
                
                if entry_cluster == cluster:
                    effectiveness = entry['result'].get('reduction_percentage', 0)
                    cluster_effectiveness.append(effectiveness)
            
            if cluster_effectiveness:
                ml_prediction['ml_predicted_reduction'] = sum(cluster_effectiveness) / len(cluster_effectiveness)
                ml_prediction['ml_confidence'] = min(1.0, len(cluster_effectiveness) * 0.1)
                ml_prediction['cluster_size'] = len(cluster_effectiveness)
                ml_prediction['cluster_id'] = int(cluster)
        
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            ml_prediction['ml_error'] = str(e)
        
        return ml_prediction

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics and patterns."""
        stats = {
            'total_optimizations': len(self.optimization_history),
            'average_effectiveness': 0.0,
            'most_common_document_types': {},
            'optimization_trends': {},
            'model_performance': dict(self.metrics),
            'pattern_categories': dict(self.pattern_categories),
            'last_updated': datetime.now().isoformat()
        }
        
        if self.optimization_history:
            # Calculate average effectiveness
            effectiveness_values = [
                entry['result'].get('reduction_percentage', 0)
                for entry in self.optimization_history
            ]
            stats['average_effectiveness'] = sum(effectiveness_values) / len(effectiveness_values)
            
            # Most common document types
            doc_type_counts = defaultdict(int)
            for entry in self.optimization_history:
                doc_type = entry.get('features', {}).get('document_type', 'unknown')
                doc_type_counts[doc_type] += 1
            
            stats['most_common_document_types'] = dict(sorted(
                doc_type_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])
            
            # Optimization trends (by time)
            if len(self.optimization_history) >= 5:
                recent_effectiveness = effectiveness_values[-5:]
                older_effectiveness = effectiveness_values[:-5] if len(effectiveness_values) > 5 else []
                
                if older_effectiveness:
                    recent_avg = sum(recent_effectiveness) / len(recent_effectiveness)
                    older_avg = sum(older_effectiveness) / len(older_effectiveness)
                    trend = recent_avg - older_avg
                    
                    stats['optimization_trends'] = {
                        'recent_average': recent_avg,
                        'historical_average': older_avg,
                        'improvement_trend': trend,
                        'trend_description': 'improving' if trend > 1 else 'stable' if trend > -1 else 'declining'
                    }
        
        return stats

    def _store_analysis_result(self, file_path: str, analysis_result: Dict[str, Any]) -> None:
        """Store analysis result for future reference."""
        try:
            # Create a hash of the file path for secure storage
            path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
            
            storage_file = self.storage_path / f"analysis_{path_hash}.json"
            
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error storing analysis result: {e}")

    def _save_patterns(self) -> None:
        """Save learned patterns to persistent storage."""
        try:
            patterns_data = {
                'usage_patterns': dict(self.usage_patterns),
                'pattern_categories': dict(self.pattern_categories),
                'metrics': dict(self.metrics),
                'user_preferences': dict(self.user_preferences),
                'last_updated': datetime.now().isoformat()
            }
            
            patterns_file = self.storage_path / "learned_patterns.json"
            
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, ensure_ascii=False)
            
            # Save optimization history (limited to recent entries)
            history_to_save = self.optimization_history[-100:]  # Keep last 100 entries
            history_file = self.storage_path / "optimization_history.json"
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)
            
            # Save ML models if available
            if self._vectorizer and self._clusterer and self._ensure_sklearn_available():
                self._save_ml_models()
                
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")

    def _save_ml_models(self) -> None:
        """Save trained ML models to disk."""
        try:
            models_dir = self.storage_path / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Save vectorizer
            if self._vectorizer:
                vectorizer_file = models_dir / "vectorizer.pkl"
                with open(vectorizer_file, 'wb') as f:
                    pickle.dump(self._vectorizer, f)
            
            # Save clusterer
            if self._clusterer:
                clusterer_file = models_dir / "clusterer.pkl"
                with open(clusterer_file, 'wb') as f:
                    pickle.dump(self._clusterer, f)
            
            # Save scaler
            if self._scaler:
                scaler_file = models_dir / "scaler.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self._scaler, f)
                    
        except Exception as e:
            self.logger.error(f"Error saving ML models: {e}")

    def _load_patterns(self) -> None:
        """Load previously learned patterns from storage."""
        try:
            patterns_file = self.storage_path / "learned_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                
                self.usage_patterns.update(patterns_data.get('usage_patterns', {}))
                self.pattern_categories.update(patterns_data.get('pattern_categories', {}))
                self.metrics.update(patterns_data.get('metrics', {}))
                self.user_preferences.update(patterns_data.get('user_preferences', {}))
            
            # Load optimization history
            history_file = self.storage_path / "optimization_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.optimization_history = json.load(f)
            
            # Load ML models
            self._load_ml_models()
            
            self.logger.info("Patterns and models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")

    def _load_ml_models(self) -> None:
        """Load previously trained ML models."""
        if not self._ensure_sklearn_available():
            return
            
        try:
            models_dir = self.storage_path / "models"
            
            # Load vectorizer
            vectorizer_file = models_dir / "vectorizer.pkl"
            if vectorizer_file.exists():
                with open(vectorizer_file, 'rb') as f:
                    self._vectorizer = pickle.load(f)
            
            # Load clusterer
            clusterer_file = models_dir / "clusterer.pkl"
            if clusterer_file.exists():
                with open(clusterer_file, 'rb') as f:
                    self._clusterer = pickle.load(f)
            
            # Load scaler
            scaler_file = models_dir / "scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self._scaler = pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading ML models: {e}")
            # Reset models on error
            self._vectorizer = None
            self._clusterer = None
            self._scaler = None

    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old analysis data and optimization history."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Filter optimization history
            filtered_history = []
            for entry in self.optimization_history:
                timestamp_str = entry.get('timestamp', '')
                if timestamp_str:
                    try:
                        entry_date = datetime.fromisoformat(timestamp_str)
                        if entry_date >= cutoff_date:
                            filtered_history.append(entry)
                    except Exception:
                        # Keep entries with invalid timestamps for safety
                        filtered_history.append(entry)
                else:
                    filtered_history.append(entry)
            
            self.optimization_history = filtered_history
            
            # Clean up old analysis files
            for analysis_file in self.storage_path.glob("analysis_*.json"):
                try:
                    file_time = datetime.fromtimestamp(analysis_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        analysis_file.unlink()
                except Exception as e:
                    self.logger.error(f"Error cleaning up file {analysis_file}: {e}")
            
            # Save cleaned data
            self._save_patterns()
            
            self.logger.info(f"Cleanup completed. Kept {len(filtered_history)} optimization history entries")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def export_learning_data(self, export_path: Optional[Path] = None) -> str:
        """Export learning data for analysis or backup."""
        try:
            export_path = export_path or Path.cwd() / "claudemd_learning_export.json"
            
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_optimizations': len(self.optimization_history),
                    'pattern_categories_count': len(self.pattern_categories),
                    'export_version': '1.0.0'
                },
                'usage_patterns': dict(self.usage_patterns),
                'pattern_categories': dict(self.pattern_categories),
                'metrics': dict(self.metrics),
                'user_preferences': dict(self.user_preferences),
                'optimization_history_sample': self.optimization_history[-10:]  # Last 10 entries
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Learning data exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            error_msg = f"Error exporting learning data: {e}"
            self.logger.error(error_msg)
            return error_msg


# Utility functions for integration
def create_usage_pattern_analyzer(storage_path: Optional[Path] = None) -> UsagePatternAnalyzer:
    """Factory function to create a UsagePatternAnalyzer instance."""
    return UsagePatternAnalyzer(storage_path=storage_path)


def analyze_document_patterns(content: str, 
                            file_path: str,
                            optimization_result: Dict[str, Any],
                            analyzer: Optional[UsagePatternAnalyzer] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze document patterns with automatic analyzer creation.
    
    Args:
        content: Document content to analyze
        file_path: Path to the document
        optimization_result: Results from token optimization
        analyzer: Optional pre-created analyzer instance
        
    Returns:
        Pattern analysis results
    """
    if analyzer is None:
        analyzer = create_usage_pattern_analyzer()
    
    return analyzer.analyze_document_usage(content, file_path, optimization_result)


def predict_optimization_success(features: Dict[str, Any],
                               analyzer: Optional[UsagePatternAnalyzer] = None) -> Dict[str, Any]:
    """
    Convenience function to predict optimization effectiveness.
    
    Args:
        features: Document features
        analyzer: Optional pre-created analyzer instance
        
    Returns:
        Prediction results
    """
    if analyzer is None:
        analyzer = create_usage_pattern_analyzer()
    
    return analyzer.predict_optimization_effectiveness(features)
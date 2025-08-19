"""
Claude.md Token Reduction Project

A secure, clean implementation for reducing Claude.md token overhead.
"""

__version__ = "0.1.0"
__author__ = "Takashi Yamashiro"
__email__ = "takashi@example.com"

# Version info
VERSION_INFO = (0, 1, 0)

# AI module components - FIXED import path
from .core.tokenizer import SmartAnalysisEngine

__all__ = [
    'SmartAnalysisEngine',
    '__version__',
    '__author__',
    '__email__',
    'VERSION_INFO'
]
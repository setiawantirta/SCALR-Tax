"""
Benchmark Package for Bioinformatics Analysis
============================================

Package ini berisi modul-modul untuk:
- Data loading dan preprocessing
- Feature extraction dan reduction  
- Machine learning training
- Utility functions
"""

# Import semua fungsi utama
try:
    from .loader import *
except ImportError:
    print("Warning: loader module not found")

try:
    from .extract_split import *
except ImportError:
    print("Warning: extract_split module not found")

try:
    from .feature_reduction import *
except ImportError:
    print("Warning: feature_reduction module not found")

try:
    from .training import *
except ImportError:
    print("Warning: training module not found")

try:
    from .create_folder import *
except ImportError:
    print("Warning: create_folder module not found")

__version__ = "1.0.0"
__author__ = "Tirta Setiawan"
"""
Monkey patches for import-time behavior
"""
import warnings

warnings.filterwarnings("ignore")  # mainly to filter out lzma warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

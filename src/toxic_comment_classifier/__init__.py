"""toxic_comment_classifier.

Toxic comment classifier using DistilBERT and HuggingFace Transformers with MLOps pipeline
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("toxic_comment_classifier")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "team_toxic"
__email__ = "apate424@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]

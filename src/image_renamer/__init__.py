"""AI-powered image renaming tool with structured outputs."""

from .cli import app
from .models import ImageAnalysis, ModelPerformance
from .ollama_client import OllamaClient

__version__ = "0.1.0"

__all__ = [
    "app",
    "ImageAnalysis",
    "ModelPerformance",
    "OllamaClient",
]

"""Ollama API client for structured image analysis."""

import json
from typing import Optional

import requests

from .models import ImageAnalysis

# Ollama configuration defaults
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llava:latest"
ALTERNATIVE_MODEL = "gemma3:latest"

# Analysis prompt template
ANALYSIS_PROMPT = """Analyze this wallpaper image and provide structured analysis.

Analyze the image focusing on:
1. Main subject/theme (be specific but concise)
2. Visual style and mood
3. Dominant colors (1-3 maximum)
4. Setting/environment if applicable
5. Generate a filename that captures the essence

Guidelines for filename_suggestion:
- Use exactly 4 words separated by underscores
- Lowercase only
- Maximum 40 characters
- Be descriptive and capture key visual elements
- Examples: "neon_city_night_skyline", "abstract_blue_flowing_waves", "mountain_sunset_lake_reflection"

Provide a confidence score based on how clear and distinctive the image features are.

Respond with valid JSON matching the required schema."""


class OllamaClient:
    """Client for communicating with Ollama API."""

    def __init__(self, model_name: str = DEFAULT_MODEL, ollama_host: str = DEFAULT_OLLAMA_HOST):
        """
        Initialize Ollama client.

        Args:
            model_name: Name of the Ollama vision model to use
            ollama_host: Base URL for Ollama API
        """
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip('/')  # Remove trailing slash if present

    async def analyze_image(self, image_b64: str) -> Optional[ImageAnalysis]:
        """
        Analyze image using Ollama vision model with structured output.

        Args:
            image_b64: Base64 encoded image string

        Returns:
            ImageAnalysis object or None if analysis fails

        Raises:
            OllamaAPIError: If API request fails
        """
        try:
            # Create the JSON schema for structured output
            schema = ImageAnalysis.model_json_schema()

            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": ANALYSIS_PROMPT,
                "images": [image_b64],
                "format": schema,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }

            # Make API request
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                raise OllamaAPIError(f"Ollama API error: {response.status_code}")

            result = response.json()
            response_text = result.get('response', '').strip()

            # Parse the JSON response
            analysis_data = json.loads(response_text)

            # Normalize confidence to be between 0.1 and 1.0
            if 'confidence' in analysis_data:
                conf = analysis_data['confidence']
                if conf > 1.0:
                    analysis_data['confidence'] = min(conf / 10.0, 1.0)
                elif conf < 0.1:
                    analysis_data['confidence'] = 0.1

            return ImageAnalysis(**analysis_data)

        except json.JSONDecodeError as e:
            raise OllamaParseError(f"Failed to parse JSON response: {e}") from e
        except requests.RequestException as e:
            raise OllamaAPIError(f"API request failed: {e}") from e
        except Exception as e:
            raise OllamaClientError(f"Unexpected error during analysis: {e}") from e

    def test_connection(self) -> bool:
        """
        Test connection to Ollama API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaAPIError(OllamaClientError):
    """Raised when Ollama API request fails."""
    pass


class OllamaParseError(OllamaClientError):
    """Raised when response parsing fails."""
    pass

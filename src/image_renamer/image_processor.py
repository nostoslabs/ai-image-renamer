"""Image processing utilities for resizing and encoding."""

import base64
import hashlib
import io
from pathlib import Path
from typing import Optional

from PIL import Image

# Image processing settings
MAX_SIZE = 1024  # Resize images to max 1024px for analysis


class ImageProcessor:
    """Handles image resizing and base64 encoding."""

    @staticmethod
    def calculate_checksum(image_path: Path) -> str:
        """
        Calculate SHA-256 checksum of an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Hexadecimal SHA-256 checksum string
        """
        sha256 = hashlib.sha256()
        with open(image_path, 'rb') as f:
            # Read file in chunks to handle large images efficiently
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def resize_and_encode(image_path: Path, max_size: int = MAX_SIZE) -> Optional[str]:
        """
        Resize image to max dimensions and convert to base64 JPEG.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height)

        Returns:
            Base64 encoded image string, or None if processing fails
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Calculate new size maintaining aspect ratio
                width, height = img.size
                if width > height:
                    new_width = min(width, max_size)
                    new_height = int(height * (new_width / width))
                else:
                    new_height = min(height, max_size)
                    new_width = int(width * (new_height / height))

                # Resize image
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                resized_img.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
                return base64.b64encode(image_bytes).decode('utf-8')

        except Exception as e:
            raise ImageProcessingError(f"Failed to process image {image_path}: {e}") from e


class ImageProcessingError(Exception):
    """Raised when image processing fails."""
    pass

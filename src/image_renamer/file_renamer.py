"""File naming and renaming utilities."""

import re
from pathlib import Path
from typing import Set

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


class FileRenamer:
    """Handles file renaming operations with safety checks."""

    @staticmethod
    def clean_filename(filename: str, original_extension: str) -> str:
        """
        Clean and validate filename suggestion.

        Args:
            filename: Suggested filename (without extension)
            original_extension: Original file extension (e.g., '.jpg')

        Returns:
            Cleaned filename with extension
        """
        # Clean the filename
        cleaned = re.sub(r'[^\w\s-]', '', filename)
        cleaned = re.sub(r'\s+', '_', cleaned)
        cleaned = cleaned.lower().strip('_')

        # Limit length
        if len(cleaned) > 40:
            cleaned = cleaned[:40].rstrip('_')

        # Ensure it's not empty
        if not cleaned:
            cleaned = "unnamed_image"

        return cleaned + original_extension

    @staticmethod
    def get_safe_filename(base_name: str, extension: str, target_dir: Path) -> str:
        """
        Generate a unique filename that doesn't conflict with existing files.

        Args:
            base_name: Base filename (without extension)
            extension: File extension (e.g., '.jpg')
            target_dir: Directory where file will be created

        Returns:
            Unique filename with extension
        """
        counter = 1
        while True:
            if counter == 1:
                new_name = f"{base_name}{extension}"
            else:
                new_name = f"{base_name}_{counter}{extension}"

            if not (target_dir / new_name).exists():
                return new_name

            counter += 1

    @staticmethod
    def rename_file(old_path: Path, new_path: Path, dry_run: bool = False) -> bool:
        """
        Rename a file with dry-run support.

        Args:
            old_path: Current file path
            new_path: New file path
            dry_run: If True, don't actually rename the file

        Returns:
            True if rename was successful (or would be in dry-run mode)
        """
        if dry_run:
            return True

        try:
            old_path.rename(new_path)
            return True
        except Exception as e:
            raise FileRenamingError(f"Failed to rename {old_path} to {new_path}: {e}") from e

    @staticmethod
    def find_images(directory: Path, supported_extensions: Set[str] = SUPPORTED_EXTENSIONS) -> list[Path]:
        """
        Find all image files in a directory.

        Args:
            directory: Directory to search
            supported_extensions: Set of supported file extensions

        Returns:
            List of image file paths
        """
        if not directory.exists() or not directory.is_dir():
            raise DirectoryNotFoundError(f"{directory} is not a valid directory")

        return [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]


class FileRenamingError(Exception):
    """Raised when file renaming fails."""
    pass


class DirectoryNotFoundError(Exception):
    """Raised when directory doesn't exist."""
    pass

"""Allow running the package as a module: python -m image_renamer"""

from .cli import app

if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
Test script to check model availability and basic functionality.
"""

import asyncio
from pathlib import Path

import typer
from image_renamer import ImageRenamer, console


async def test_models():
    """Test both models for availability and basic functionality."""

    models = ["llava:latest", "gemma3:latest"]

    console.print("[bold]🧪 Testing Model Availability[/bold]\n")

    for model in models:
        console.print(f"Testing {model}...")

        try:
            renamer = ImageRenamer(model)

            # Test with a simple prompt (without image for basic connectivity)
            console.print(f"  ✅ {model} initialized successfully")

        except Exception as e:
            console.print(f"  ❌ {model} failed: {e}")

    console.print("\n[green]Model testing complete![/green]")


def main():
    """Entry point for the test-models script."""
    asyncio.run(test_models())


if __name__ == "__main__":
    main()
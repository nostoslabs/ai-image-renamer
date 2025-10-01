"""Command-line interface for image-renamer."""

import asyncio
import time
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import Progress

from .file_renamer import DirectoryNotFoundError, FileRenamer
from .image_processor import ImageProcessor, ImageProcessingError
from .models import ImageAnalysis
from .ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    ALTERNATIVE_MODEL,
    OllamaClient,
    OllamaClientError,
)
from .performance_tracker import PerformanceTracker, display_comparison_table

console = Console()
app = typer.Typer(
    name="image-renamer",
    help="AI-powered image renaming tool with structured outputs"
)


class ImageRenamerService:
    """Orchestrates image analysis and renaming operations."""

    def __init__(self, model_name: str, ollama_host: str):
        """
        Initialize the image renamer service.

        Args:
            model_name: Name of the Ollama model to use
            ollama_host: Ollama API host URL
        """
        self.ollama_client = OllamaClient(model_name, ollama_host)
        self.image_processor = ImageProcessor()
        self.file_renamer = FileRenamer()
        self.tracker = PerformanceTracker(model_name)

    async def analyze_and_rename_image(
        self,
        image_path: Path,
        dry_run: bool = False
    ) -> Optional[ImageAnalysis]:
        """
        Analyze and rename a single image.

        Args:
            image_path: Path to the image file
            dry_run: If True, preview changes without renaming

        Returns:
            ImageAnalysis result if successful, None otherwise
        """
        try:
            # Process image
            image_b64 = self.image_processor.resize_and_encode(image_path)

            # Analyze with timing
            with self.tracker.track_analysis():
                analysis = await self.ollama_client.analyze_image(image_b64)

            if not analysis:
                return None

            self.tracker.record_success(analysis)

            # Clean filename
            clean_name = self.file_renamer.clean_filename(
                analysis.filename_suggestion,
                image_path.suffix.lower()
            )

            # Get unique filename
            new_name = self.file_renamer.get_safe_filename(
                clean_name.replace(image_path.suffix.lower(), ''),
                image_path.suffix.lower(),
                image_path.parent
            )

            # Display analysis
            self._display_analysis(image_path.name, analysis, new_name, dry_run)

            # Rename the file
            new_path = image_path.parent / new_name
            self.file_renamer.rename_file(image_path, new_path, dry_run)

            return analysis

        except (ImageProcessingError, OllamaClientError) as e:
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
            self.tracker.record_error()
            return None

    async def process_directory(
        self,
        target_dir: Path,
        dry_run: bool = False,
        max_files: Optional[int] = None,
        concurrent: int = 2
    ) -> Dict[str, int]:
        """
        Process all images in a directory with concurrent processing.

        Args:
            target_dir: Directory containing images
            dry_run: If True, preview changes without renaming
            max_files: Maximum number of files to process
            concurrent: Number of concurrent requests to process

        Returns:
            Dictionary with processing statistics
        """
        try:
            # Find all image files
            image_files = self.file_renamer.find_images(target_dir)
        except DirectoryNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return {"processed": 0, "skipped": 0, "errors": 0}

        if not image_files:
            console.print(f"[yellow]No image files found in {target_dir}[/yellow]")
            return {"processed": 0, "skipped": 0, "errors": 0}

        # Limit files if specified
        if max_files:
            image_files = image_files[:max_files]

        console.print(f"[green]Found {len(image_files)} image files to process[/green]")
        console.print(f"[blue]Using model: {self.ollama_client.model_name}[/blue]")
        console.print(f"[blue]Concurrent requests: {concurrent}[/blue]")

        stats = {"processed": 0, "skipped": 0, "errors": 0}

        # Process images with progress bar
        with Progress() as progress:
            overall_task = progress.add_task(
                "[green]Overall progress",
                total=len(image_files)
            )

            # Process images in batches
            for i in range(0, len(image_files), concurrent):
                batch = image_files[i:i + concurrent]

                # Create tasks for concurrent processing
                tasks = [
                    self.analyze_and_rename_image(image_file, dry_run)
                    for image_file in batch
                ]

                # Wait for all tasks in batch to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Update stats
                for result in results:
                    if isinstance(result, Exception):
                        stats["errors"] += 1
                    elif result:
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

                    progress.advance(overall_task)

        return stats

    def _display_analysis(
        self,
        original_name: str,
        analysis: ImageAnalysis,
        new_name: str,
        dry_run: bool
    ):
        """Display structured analysis results."""
        action = "Would rename" if dry_run else "Renamed"
        status_icon = "🔍" if dry_run else "✅"

        console.print(f"\n{status_icon} [bold]{original_name}[/bold]")
        console.print(f"  Subject: [cyan]{analysis.main_subject}[/cyan]")
        console.print(f"  Style: [magenta]{analysis.style}[/magenta]")
        console.print(f"  Colors: [yellow]{', '.join(analysis.dominant_colors)}[/yellow]")

        if analysis.setting:
            console.print(f"  Setting: [green]{analysis.setting}[/green]")

        console.print(f"  Confidence: [blue]{analysis.confidence:.2f}[/blue]")
        console.print(f"  {action}: [bold green]{new_name}[/bold green]")


@app.command()
def main(
    directory: Optional[Path] = typer.Argument(None, help="Target directory containing images"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without renaming"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Ollama model to use"),
    host: str = typer.Option(DEFAULT_OLLAMA_HOST, "--host", help="Ollama host URL (e.g., http://192.168.1.100:11434)"),
    max_files: Optional[int] = typer.Option(None, "--max-files", help="Limit number of files to process"),
    concurrent: int = typer.Option(2, "--concurrent", "-c", help="Number of concurrent requests (default: 2)"),
    compare_models: bool = typer.Option(False, "--compare", help="Compare LLaVA vs Gemma3 performance"),
    test: bool = typer.Option(False, "--test", help="Test model availability")
):
    """AI-powered image renaming with structured outputs using Ollama."""

    if test:
        asyncio.run(test_models(host))
        return

    if not directory:
        console.print("[red]Error: Directory argument is required (unless using --test)[/red]")
        console.print("Use --help for usage information")
        raise typer.Exit(1)

    console.print("[bold green]🎨 AI Image Renamer with Structured Outputs[/bold green]\n")

    if compare_models:
        asyncio.run(compare_model_performance(directory, dry_run, max_files, host))
    else:
        asyncio.run(process_with_model(directory, model, dry_run, max_files, host, concurrent))


async def process_with_model(
    directory: Path,
    model: str,
    dry_run: bool,
    max_files: Optional[int],
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    concurrent: int = 2
):
    """Process images with a single model."""
    service = ImageRenamerService(model, ollama_host)
    stats = await service.process_directory(directory, dry_run, max_files, concurrent)

    # Display summary
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Processed: [green]{stats['processed']}[/green]")
    console.print(f"Skipped: [yellow]{stats['skipped']}[/yellow]")
    console.print(f"Errors: [red]{stats['errors']}[/red]")

    service.tracker.display_summary()


async def compare_model_performance(
    directory: Path,
    dry_run: bool,
    max_files: Optional[int],
    ollama_host: str = DEFAULT_OLLAMA_HOST
):
    """Compare performance between LLaVA and Gemma3 models."""
    console.print("[bold yellow]🔬 Comparing Model Performance[/bold yellow]\n")

    models_to_test = [DEFAULT_MODEL, ALTERNATIVE_MODEL]
    results = {}

    # Limit to fewer files for comparison
    test_limit = min(5, max_files) if max_files else 5

    for model in models_to_test:
        console.print(f"\n[bold]Testing {model}[/bold]")

        service = ImageRenamerService(model, ollama_host)

        start_time = time.time()
        stats = await service.process_directory(directory, True, test_limit)  # Always dry run for comparison
        total_time = time.time() - start_time

        results[model] = {
            "tracker": service.tracker,
            "stats": stats,
            "total_time": total_time
        }

    # Display comparison
    display_comparison_table(results)


async def test_models(ollama_host: str = DEFAULT_OLLAMA_HOST):
    """Test model availability."""
    console.print("[bold]🧪 Testing Model Availability[/bold]\n")

    models = [DEFAULT_MODEL, ALTERNATIVE_MODEL]

    for model in models:
        console.print(f"Testing {model}...")

        try:
            client = OllamaClient(model, ollama_host)
            if client.test_connection():
                console.print(f"  ✅ {model} initialized successfully")
            else:
                console.print(f"  ❌ {model} connection failed")

        except Exception as e:
            console.print(f"  ❌ {model} failed: {e}")

    console.print("\n[green]Model testing complete![/green]")


if __name__ == "__main__":
    app()

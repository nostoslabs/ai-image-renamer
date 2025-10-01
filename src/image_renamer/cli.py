"""Command-line interface for image-renamer."""

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn

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

    def __init__(self, model_name: str, ollama_host: str, dest_dir: Optional[Path] = None):
        """
        Initialize the image renamer service.

        Args:
            model_name: Name of the Ollama model to use
            ollama_host: Ollama API host URL
            dest_dir: Optional destination directory for renamed files
        """
        self.ollama_client = OllamaClient(model_name, ollama_host)
        self.image_processor = ImageProcessor()
        self.file_renamer = FileRenamer()
        self.tracker = PerformanceTracker(model_name)
        self.dest_dir = dest_dir

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

            # Determine target directory
            target_dir = self.dest_dir if self.dest_dir else image_path.parent

            # Get unique filename
            new_name = self.file_renamer.get_safe_filename(
                clean_name.replace(image_path.suffix.lower(), ''),
                image_path.suffix.lower(),
                target_dir
            )

            # Display analysis
            self._display_analysis(image_path.name, analysis, new_name, dry_run, self.dest_dir)

            # Rename/move the file
            new_path = target_dir / new_name
            self.file_renamer.rename_file(image_path, new_path, dry_run, move_to_dest=bool(self.dest_dir))

            return analysis

        except (ImageProcessingError, OllamaClientError) as e:
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
            self.tracker.record_error()
            return None

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format ETA in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    @staticmethod
    def _format_rate(rate: float, elapsed: float, completed: int) -> str:
        """Format processing rate (img/s or s/img)."""
        if rate >= 1.0:
            return f"{rate:.2f} img/s"
        else:
            seconds_per_image = elapsed / completed
            return f"{seconds_per_image:.2f} s/img"

    def _find_unique_images(
        self,
        image_files: List[Path],
        delete_duplicates: bool = False
    ) -> tuple[List[Path], Dict[str, List[Path]], int]:
        """
        Find unique images by calculating SHA-256 checksums.

        Args:
            image_files: List of image file paths
            delete_duplicates: If True, delete duplicate files

        Returns:
            Tuple of (unique_files, duplicates_dict, deleted_count) where duplicates_dict
            maps checksums to lists of duplicate file paths, and deleted_count is the
            number of files deleted
        """
        checksums: Dict[str, Path] = {}
        duplicates: Dict[str, List[Path]] = defaultdict(list)
        unique_files: List[Path] = []
        deleted_count = 0

        console.print("[cyan]Calculating checksums to detect duplicates...[/cyan]")

        with Progress() as progress:
            checksum_task = progress.add_task(
                "[cyan]Calculating checksums",
                total=len(image_files)
            )

            for image_path in image_files:
                try:
                    checksum = self.image_processor.calculate_checksum(image_path)

                    if checksum in checksums:
                        # Found duplicate
                        duplicates[checksum].append(image_path)

                        # Delete if requested
                        if delete_duplicates:
                            try:
                                image_path.unlink()
                                deleted_count += 1
                                console.print(f"[red]Deleted duplicate: {image_path.name}[/red]")
                            except Exception as e:
                                console.print(f"[yellow]Failed to delete {image_path.name}: {e}[/yellow]")
                    else:
                        # First occurrence of this checksum
                        checksums[checksum] = image_path
                        unique_files.append(image_path)

                except Exception as e:
                    console.print(f"[yellow]Warning: Could not checksum {image_path.name}: {e}[/yellow]")
                    # Include files that failed checksum (better safe than sorry)
                    unique_files.append(image_path)

                progress.advance(checksum_task)

        return unique_files, duplicates, deleted_count

    async def process_directory(
        self,
        target_dir: Path,
        dry_run: bool = False,
        max_files: Optional[int] = None,
        concurrent: int = 2,
        delete_duplicates: bool = False
    ) -> Dict[str, int]:
        """
        Process all images in a directory with concurrent processing.

        Args:
            target_dir: Directory containing images
            dry_run: If True, preview changes without renaming
            max_files: Maximum number of files to process
            concurrent: Number of concurrent requests to process
            delete_duplicates: If True, delete duplicate files

        Returns:
            Dictionary with processing statistics
        """
        try:
            # Find all image files
            image_files = self.file_renamer.find_images(target_dir)
        except DirectoryNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return {"processed": 0, "skipped": 0, "errors": 0, "duplicates": 0, "deleted": 0}

        if not image_files:
            console.print(f"[yellow]No image files found in {target_dir}[/yellow]")
            return {"processed": 0, "skipped": 0, "errors": 0, "duplicates": 0, "deleted": 0}

        console.print(f"[green]Found {len(image_files)} image files[/green]")

        # Detect and filter duplicates
        unique_files, duplicates, deleted_count = self._find_unique_images(image_files, delete_duplicates)
        duplicate_count = sum(len(dups) for dups in duplicates.values())

        if duplicate_count > 0:
            if delete_duplicates:
                console.print(f"[red]Deleted {deleted_count} duplicate files[/red]")
            else:
                console.print(f"[yellow]Found {duplicate_count} duplicate files (will be skipped)[/yellow]")

            for checksum, dup_files in duplicates.items():
                console.print(f"  [dim]Checksum {checksum[:16]}...: {', '.join(f.name for f in dup_files)}[/dim]")

        console.print(f"[green]{len(unique_files)} unique images to process[/green]")

        # Limit files if specified
        if max_files:
            unique_files = unique_files[:max_files]

        console.print(f"[blue]Using model: {self.ollama_client.model_name}[/blue]")
        console.print(f"[blue]Concurrent requests: {concurrent}[/blue]")

        stats = {"processed": 0, "skipped": 0, "errors": 0, "duplicates": duplicate_count, "deleted": deleted_count}

        # Process images with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("[yellow]ETA: {task.fields[eta]}"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[rate]}"),
            console=console
        ) as progress:
            overall_task = progress.add_task(
                "[green]Processing images",
                total=len(unique_files),
                rate="--",
                eta="--"
            )

            start_time = time.time()

            # Process images in batches
            for i in range(0, len(unique_files), concurrent):
                batch = unique_files[i:i + concurrent]

                # Create tasks for concurrent processing
                tasks = [
                    self.analyze_and_rename_image(image_file, dry_run)
                    for image_file in batch
                ]

                # Wait for all tasks in batch to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Update stats and count completed in batch
                batch_completed = 0
                for result in results:
                    if isinstance(result, Exception):
                        stats["errors"] += 1
                    elif result:
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1
                    batch_completed += 1

                # Advance progress by batch size (more efficient and better for ETA calculation)
                progress.advance(overall_task, batch_completed)

                # Update rate and ETA
                elapsed = time.time() - start_time
                completed = stats["processed"] + stats["skipped"] + stats["errors"]
                if elapsed > 0 and completed > 0:
                    rate = completed / elapsed
                    remaining = len(unique_files) - completed
                    eta_seconds = remaining / rate

                    eta_str = self._format_eta(eta_seconds)
                    rate_str = self._format_rate(rate, elapsed, completed)

                    progress.update(overall_task, rate=rate_str, eta=eta_str)

        return stats

    def _display_analysis(
        self,
        original_name: str,
        analysis: ImageAnalysis,
        new_name: str,
        dry_run: bool,
        dest_dir: Optional[Path] = None
    ):
        """Display structured analysis results."""
        if dest_dir:
            action = "Would move to" if dry_run else "Moved to"
        else:
            action = "Would rename" if dry_run else "Renamed"

        status_icon = "🔍" if dry_run else "✅"

        console.print(f"\n{status_icon} [bold]{original_name}[/bold]")
        console.print(f"  Subject: [cyan]{analysis.main_subject}[/cyan]")
        console.print(f"  Style: [magenta]{analysis.style}[/magenta]")
        console.print(f"  Colors: [yellow]{', '.join(analysis.dominant_colors)}[/yellow]")

        if analysis.setting:
            console.print(f"  Setting: [green]{analysis.setting}[/green]")

        console.print(f"  Confidence: [blue]{analysis.confidence:.2f}[/blue]")

        if dest_dir:
            console.print(f"  {action}: [bold green]{dest_dir}/{new_name}[/bold green]")
        else:
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
    test: bool = typer.Option(False, "--test", help="Test model availability"),
    dest: Optional[Path] = typer.Option(None, "--dest", help="Destination directory for renamed files (will be created if it doesn't exist)"),
    delete_duplicates: bool = typer.Option(False, "--delete-duplicates", help="Delete duplicate files instead of just skipping them")
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
        asyncio.run(compare_model_performance(directory, dry_run, max_files, host, dest, delete_duplicates))
    else:
        asyncio.run(process_with_model(directory, model, dry_run, max_files, host, concurrent, dest, delete_duplicates))


async def process_with_model(
    directory: Path,
    model: str,
    dry_run: bool,
    max_files: Optional[int],
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    concurrent: int = 2,
    dest_dir: Optional[Path] = None,
    delete_duplicates: bool = False
):
    """Process images with a single model."""
    service = ImageRenamerService(model, ollama_host, dest_dir)
    stats = await service.process_directory(directory, dry_run, max_files, concurrent, delete_duplicates)

    # Display summary
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Processed: [green]{stats['processed']}[/green]")
    console.print(f"Duplicates: [yellow]{stats['duplicates']}[/yellow]")
    if delete_duplicates and stats['deleted'] > 0:
        console.print(f"Deleted: [red]{stats['deleted']}[/red]")
    console.print(f"Skipped: [yellow]{stats['skipped']}[/yellow]")
    console.print(f"Errors: [red]{stats['errors']}[/red]")

    service.tracker.display_summary()


async def compare_model_performance(
    directory: Path,
    dry_run: bool,
    max_files: Optional[int],
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    dest_dir: Optional[Path] = None,
    delete_duplicates: bool = False
):
    """Compare performance between LLaVA and Gemma3 models."""
    console.print("[bold yellow]🔬 Comparing Model Performance[/bold yellow]\n")

    models_to_test = [DEFAULT_MODEL, ALTERNATIVE_MODEL]
    results = {}

    # Limit to fewer files for comparison
    test_limit = min(5, max_files) if max_files else 5

    for model in models_to_test:
        console.print(f"\n[bold]Testing {model}[/bold]")

        service = ImageRenamerService(model, ollama_host, dest_dir)

        start_time = time.time()
        stats = await service.process_directory(directory, True, test_limit, delete_duplicates=delete_duplicates)  # Always dry run for comparison
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

#!/usr/bin/env python3
"""
AI-powered Image Renaming Tool with Structured Outputs

This script analyzes images using Ollama models with structured responses via Pydantic
and renames them based on their content. It supports both LLaVA and Gemma3 vision models
for optimal performance.

Usage: python image_renamer.py <target_directory>
"""

import asyncio
import base64
import io
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Initialize rich console for pretty printing
console = Console()

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llava:latest"
ALTERNATIVE_MODEL = "gemma3:latest"

# Image processing settings
MAX_SIZE = 1024  # Resize images to max 1024px for analysis


class ImageAnalysis(BaseModel):
    """Structured response model for image analysis."""

    main_subject: str = Field(
        description="Primary subject of the image (e.g., 'mountain', 'cityscape', 'abstract', 'nature')"
    )
    style: str = Field(
        description="Visual style or mood (e.g., 'minimalist', 'vibrant', 'dark', 'sunset', 'digital_art')"
    )
    dominant_colors: List[str] = Field(
        description="1-3 dominant colors in the image (e.g., ['blue', 'orange'], ['green'])",
        max_length=3
    )
    setting: Optional[str] = Field(
        default=None,
        description="Setting or environment if applicable (e.g., 'urban', 'forest', 'space', 'underwater')"
    )
    filename_suggestion: str = Field(
        description="Suggested filename using underscores, lowercase, max 40 chars (e.g., 'sunset_mountain_landscape')"
    )
    confidence: float = Field(
        description="Confidence score from 0.1 to 1.0 for the analysis",
        ge=0.1,
        le=1.0
    )


class ModelPerformance(BaseModel):
    """Track model performance metrics."""

    model_name: str
    success_count: int = 0
    error_count: int = 0
    total_time: float = 0.0
    avg_confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_time_per_image(self) -> float:
        return self.total_time / self.success_count if self.success_count > 0 else 0.0


class ImageRenamer:
    """Main class for AI-powered image renaming with structured outputs."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.performance = ModelPerformance(model_name=model_name)
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[None, ImageAnalysis]:
        """Create Pydantic AI agent with Ollama model."""

        model = OpenAIChatModel(
            model_name=self.model_name,
            provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
        )

        # Create agent with structured output
        agent = Agent(
            model=model,
            output_type=ImageAnalysis,
            system_prompt="""You are an expert image analyst that provides structured,
            consistent descriptions for wallpapers and artistic images.

            Analyze the image focusing on:
            1. Main subject/theme (be specific but concise)
            2. Visual style and mood
            3. Dominant colors (1-3 maximum)
            4. Setting/environment if applicable
            5. Generate a filename that captures the essence

            Guidelines for filename_suggestion:
            - Use underscores instead of spaces
            - Lowercase only
            - Maximum 40 characters
            - Be descriptive but concise
            - Examples: "neon_city_night", "abstract_blue_waves", "mountain_sunset_reflection"

            Provide a confidence score based on how clear and distinctive the image features are."""
        )

        return agent

    def resize_image_for_analysis(self, image_path: Path) -> Optional[str]:
        """Resize image and convert to base64 for analysis."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Calculate new size maintaining aspect ratio
                width, height = img.size
                if width > height:
                    new_width = min(width, MAX_SIZE)
                    new_height = int(height * (new_width / width))
                else:
                    new_height = min(height, MAX_SIZE)
                    new_width = int(width * (new_height / height))

                # Resize image
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                resized_img.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
                return base64.b64encode(image_bytes).decode('utf-8')

        except Exception as e:
            console.print(f"[red]Error processing image {image_path}: {e}[/red]")
            return None

    async def analyze_image(self, image_path: Path) -> Optional[ImageAnalysis]:
        """Analyze image using direct Ollama API with structured output."""

        # Resize and encode image
        image_b64 = self.resize_image_for_analysis(image_path)
        if not image_b64:
            return None

        start_time = time.time()

        try:
            # Use direct Ollama API call with structured output
            import requests
            import json

            # Create the JSON schema for structured output
            schema = ImageAnalysis.model_json_schema()

            # Prepare the request
            prompt = """Analyze this wallpaper image and provide structured analysis.

Analyze the image focusing on:
1. Main subject/theme (be specific but concise)
2. Visual style and mood
3. Dominant colors (1-3 maximum)
4. Setting/environment if applicable
5. Generate a filename that captures the essence

Guidelines for filename_suggestion:
- Use underscores instead of spaces
- Lowercase only
- Maximum 40 characters
- Be descriptive but concise
- Examples: "neon_city_night", "abstract_blue_waves", "mountain_sunset_reflection"

Provide a confidence score based on how clear and distinctive the image features are.

Respond with valid JSON matching the required schema."""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "format": schema,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()

                # Parse the JSON response
                try:
                    analysis_data = json.loads(response_text)

                    # Debug: print the raw data (uncomment for debugging)
                    # console.print(f"[blue]Raw analysis data: {analysis_data}[/blue]")

                    # Normalize confidence to be between 0.1 and 1.0
                    if 'confidence' in analysis_data:
                        conf = analysis_data['confidence']
                        if conf > 1.0:
                            analysis_data['confidence'] = min(conf / 10.0, 1.0)
                        elif conf < 0.1:
                            analysis_data['confidence'] = 0.1

                    analysis = ImageAnalysis(**analysis_data)

                    analysis_time = time.time() - start_time

                    # Update performance metrics
                    self.performance.success_count += 1
                    self.performance.total_time += analysis_time

                    # Update average confidence
                    total_confidence = (
                        self.performance.avg_confidence * (self.performance.success_count - 1) +
                        analysis.confidence
                    )
                    self.performance.avg_confidence = total_confidence / self.performance.success_count

                    return analysis

                except json.JSONDecodeError as e:
                    console.print(f"[red]Failed to parse JSON response: {e}[/red]")
                    console.print(f"[yellow]Raw response: {response_text[:200]}...[/yellow]")
                    self.performance.error_count += 1
                    return None

            else:
                console.print(f"[red]Ollama API error: {response.status_code}[/red]")
                self.performance.error_count += 1
                return None

        except Exception as e:
            self.performance.error_count += 1
            console.print(f"[red]Error analyzing {image_path}: {e}[/red]")
            return None

    def clean_filename(self, filename: str, original_extension: str) -> str:
        """Clean and validate filename suggestion."""

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

    def get_safe_filename(self, base_name: str, extension: str, target_dir: Path) -> str:
        """Generate a unique filename that doesn't conflict with existing files."""

        counter = 1
        while True:
            if counter == 1:
                new_name = f"{base_name}{extension}"
            else:
                new_name = f"{base_name}_{counter}{extension}"

            if not (target_dir / new_name).exists():
                return new_name

            counter += 1

    async def process_directory(
        self,
        target_dir: Path,
        dry_run: bool = False,
        max_files: Optional[int] = None
    ) -> Dict[str, int]:
        """Process all images in the target directory."""

        if not target_dir.exists() or not target_dir.is_dir():
            console.print(f"[red]Error: {target_dir} is not a valid directory[/red]")
            return {"processed": 0, "skipped": 0, "errors": 0}

        # Find all image files
        image_files = [
            f for f in target_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            console.print(f"[yellow]No image files found in {target_dir}[/yellow]")
            return {"processed": 0, "skipped": 0, "errors": 0}

        # Limit files if specified
        if max_files:
            image_files = image_files[:max_files]

        console.print(f"[green]Found {len(image_files)} image files to process[/green]")
        console.print(f"[blue]Using model: {self.model_name}[/blue]")

        stats = {"processed": 0, "skipped": 0, "errors": 0}

        # Process images with progress bar
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Analyzing images with {self.model_name}...",
                total=len(image_files)
            )

            for image_file in image_files:
                progress.update(task, description=f"[cyan]Processing: {image_file.name}")

                try:
                    # Analyze image
                    analysis = await self.analyze_image(image_file)

                    if analysis:
                        # Clean filename
                        clean_name = self.clean_filename(
                            analysis.filename_suggestion,
                            image_file.suffix.lower()
                        )

                        # Get unique filename
                        new_name = self.get_safe_filename(
                            clean_name.replace(image_file.suffix.lower(), ''),
                            image_file.suffix.lower(),
                            target_dir
                        )

                        # Display analysis
                        self._display_analysis(image_file.name, analysis, new_name, dry_run)

                        if not dry_run:
                            # Rename the file
                            new_path = target_dir / new_name
                            image_file.rename(new_path)

                        stats["processed"] += 1
                    else:
                        console.print(f"[yellow]  ⚠️  Could not analyze {image_file.name}[/yellow]")
                        stats["skipped"] += 1

                except Exception as e:
                    console.print(f"[red]  ❌ Error processing {image_file.name}: {e}[/red]")
                    stats["errors"] += 1

                progress.advance(task)

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

    def display_performance_summary(self):
        """Display performance metrics."""

        console.print("\n[bold]Performance Summary[/bold]")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model", self.performance.model_name)
        table.add_row("Success Rate", f"{self.performance.success_rate:.1%}")
        table.add_row("Images Processed", str(self.performance.success_count))
        table.add_row("Errors", str(self.performance.error_count))
        table.add_row("Average Time/Image", f"{self.performance.avg_time_per_image:.2f}s")
        table.add_row("Average Confidence", f"{self.performance.avg_confidence:.2f}")

        console.print(table)


# CLI Application
app = typer.Typer(
    name="image-renamer",
    help="AI-powered image renaming tool with structured outputs"
)


@app.command()
def main(
    directory: Optional[Path] = typer.Argument(None, help="Target directory containing images"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without renaming"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Ollama model to use"),
    max_files: Optional[int] = typer.Option(None, "--max-files", help="Limit number of files to process"),
    compare_models: bool = typer.Option(False, "--compare", help="Compare LLaVA vs Gemma3 performance"),
    test: bool = typer.Option(False, "--test", help="Test model availability")
):
    """AI-powered image renaming with structured outputs using Ollama."""

    if test:
        try:
            from test_models import test_models
        except ImportError:
            # Inline test function for standalone execution
            async def test_models():
                console.print("[bold]🧪 Testing Model Availability[/bold]\n")

                models = ["llava:latest", "gemma3:latest"]

                for model in models:
                    console.print(f"Testing {model}...")

                    try:
                        renamer = ImageRenamer(model)
                        console.print(f"  ✅ {model} initialized successfully")

                    except Exception as e:
                        console.print(f"  ❌ {model} failed: {e}")

                console.print("\n[green]Model testing complete![/green]")

        asyncio.run(test_models())
        return

    if not directory:
        console.print("[red]Error: Directory argument is required (unless using --test)[/red]")
        console.print("Use --help for usage information")
        raise typer.Exit(1)

    console.print("[bold green]🎨 AI Image Renamer with Structured Outputs[/bold green]\n")

    if compare_models:
        asyncio.run(compare_model_performance(directory, dry_run, max_files))
    else:
        asyncio.run(process_with_model(directory, model, dry_run, max_files))


if __name__ == "__main__":
    app()




async def process_with_model(
    directory: Path,
    model: str,
    dry_run: bool,
    max_files: Optional[int]
):
    """Process images with a single model."""

    renamer = ImageRenamer(model)

    stats = await renamer.process_directory(directory, dry_run, max_files)

    # Display summary
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Processed: [green]{stats['processed']}[/green]")
    console.print(f"Skipped: [yellow]{stats['skipped']}[/yellow]")
    console.print(f"Errors: [red]{stats['errors']}[/red]")

    renamer.display_performance_summary()


async def compare_model_performance(
    directory: Path,
    dry_run: bool,
    max_files: Optional[int]
):
    """Compare performance between LLaVA and Gemma3 models."""

    console.print("[bold yellow]🔬 Comparing Model Performance[/bold yellow]\n")

    models_to_test = [DEFAULT_MODEL, ALTERNATIVE_MODEL]
    results = {}

    # Limit to fewer files for comparison
    test_limit = min(5, max_files) if max_files else 5

    for model in models_to_test:
        console.print(f"\n[bold]Testing {model}[/bold]")

        renamer = ImageRenamer(model)

        start_time = time.time()
        stats = await renamer.process_directory(directory, True, test_limit)  # Always dry run for comparison
        total_time = time.time() - start_time

        results[model] = {
            "renamer": renamer,
            "stats": stats,
            "total_time": total_time
        }

    # Display comparison
    console.print("\n[bold]Model Comparison Results[/bold]")

    comparison_table = Table()
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Success Rate", style="green")
    comparison_table.add_column("Avg Time/Image", style="yellow")
    comparison_table.add_column("Avg Confidence", style="magenta")
    comparison_table.add_column("Total Time", style="blue")

    for model, result in results.items():
        renamer = result["renamer"]
        comparison_table.add_row(
            model,
            f"{renamer.performance.success_rate:.1%}",
            f"{renamer.performance.avg_time_per_image:.2f}s",
            f"{renamer.performance.avg_confidence:.2f}",
            f"{result['total_time']:.1f}s"
        )

    console.print(comparison_table)

    # Recommend best model
    best_model = min(
        results.keys(),
        key=lambda m: results[m]["renamer"].performance.avg_time_per_image
    )

    console.print(f"\n[bold green]💡 Recommendation: {best_model} appears to be faster[/bold green]")


if __name__ == "__main__":
    app()
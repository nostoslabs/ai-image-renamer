"""Performance tracking and display utilities."""

import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.table import Table

from .models import ImageAnalysis, ModelPerformance

console = Console()


class PerformanceTracker:
    """Tracks and displays performance metrics for image analysis."""

    def __init__(self, model_name: str):
        """
        Initialize performance tracker.

        Args:
            model_name: Name of the model being tracked
        """
        self.performance = ModelPerformance(model_name=model_name)

    @contextmanager
    def track_analysis(self) -> Generator[None, None, None]:
        """
        Context manager to track timing of an analysis operation.

        Usage:
            with tracker.track_analysis():
                # perform analysis
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            self.performance.total_time += elapsed_time

    def record_success(self, analysis: ImageAnalysis) -> None:
        """
        Record a successful analysis.

        Args:
            analysis: The successful ImageAnalysis result
        """
        self.performance.success_count += 1

        # Update average confidence
        total_confidence = (
            self.performance.avg_confidence * (self.performance.success_count - 1) +
            analysis.confidence
        )
        self.performance.avg_confidence = total_confidence / self.performance.success_count

    def record_error(self) -> None:
        """Record an analysis error."""
        self.performance.error_count += 1

    def display_summary(self) -> None:
        """Display performance metrics in a formatted table."""
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

    @property
    def stats(self) -> ModelPerformance:
        """Get the current performance statistics."""
        return self.performance


def display_comparison_table(results: dict) -> None:
    """
    Display comparison table for multiple models.

    Args:
        results: Dictionary mapping model names to their results
                 Each result should have 'tracker' and 'total_time' keys
    """
    console.print("\n[bold]Model Comparison Results[/bold]")

    comparison_table = Table()
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Success Rate", style="green")
    comparison_table.add_column("Avg Time/Image", style="yellow")
    comparison_table.add_column("Avg Confidence", style="magenta")
    comparison_table.add_column("Total Time", style="blue")

    for model, result in results.items():
        tracker = result["tracker"]
        comparison_table.add_row(
            model,
            f"{tracker.stats.success_rate:.1%}",
            f"{tracker.stats.avg_time_per_image:.2f}s",
            f"{tracker.stats.avg_confidence:.2f}",
            f"{result['total_time']:.1f}s"
        )

    console.print(comparison_table)

    # Recommend best model
    best_model = min(
        results.keys(),
        key=lambda m: results[m]["tracker"].stats.avg_time_per_image
    )

    console.print(f"\n[bold green]💡 Recommendation: {best_model} appears to be faster[/bold green]")

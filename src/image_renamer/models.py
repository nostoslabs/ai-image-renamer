"""Pydantic models for structured data validation."""

from typing import List, Optional

from pydantic import BaseModel, Field


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
        """Calculate success rate as a percentage."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_time_per_image(self) -> float:
        """Calculate average processing time per successful image."""
        return self.total_time / self.success_count if self.success_count > 0 else 0.0

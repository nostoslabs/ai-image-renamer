# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered image renaming tool using local vision models via Ollama. Analyzes images and generates descriptive filenames with structured outputs validated by Pydantic.

## Development Commands

**IMPORTANT: Always use `uv` for all operations. Do not use `python` directly.**

```bash
# Install dependencies
uv sync

# Run the tool (dry-run recommended first)
uv run image-renamer /path/to/images --dry-run

# Test model connectivity
uv run image-renamer --test

# Compare model performance
uv run image-renamer /path/to/images --compare

# Use remote Ollama instance
uv run image-renamer /path/to/images --host http://192.168.1.100:11434

# Run with specific model
uv run image-renamer /path/to/images --model gemma3:latest

# Run with concurrent processing (faster)
uv run image-renamer /path/to/images --concurrent 4

# Move renamed files to destination directory
uv run image-renamer /path/to/images --dest /path/to/output

# Delete duplicate files during processing
uv run image-renamer /path/to/images --delete-duplicates
```

## Architecture

### Core Components

**Modular architecture** - Organized into focused modules:
- `models.py`: Pydantic models (`ImageAnalysis`, `ModelPerformance`)
- `image_processor.py`: Image resizing, encoding, and SHA-256 checksum calculation
- `ollama_client.py`: Ollama API client with structured outputs
- `file_renamer.py`: File operations, naming, and moving to destination directories
- `performance_tracker.py`: Metrics tracking and display
- `cli.py`: CLI interface with Typer, orchestration, and Rich progress bars

### Key Design Patterns

**Direct Ollama API Integration**: Uses native Ollama `/api/generate` endpoint with JSON schema validation instead of higher-level wrappers. This provides structured outputs with Pydantic validation.

**Dual Model Support**:
- Primary: LLaVA (faster, ~4.8s/image)
- Alternative: Gemma3 (slower, ~25.5s/image, better for abstract content)

**Concurrent Processing**: Uses async/await with batch processing. Default is 2 concurrent requests; configurable via `--concurrent` flag. Images are processed in batches using `asyncio.gather()` for parallel API calls.

**Duplicate Detection**: SHA-256 checksums are calculated before processing to identify and skip duplicate files. Saves processing time by avoiding redundant analysis. Optionally delete duplicates with `--delete-duplicates`.

**Destination Directory Support**: The `--dest` option allows moving renamed files to a different directory. The destination is created automatically if it doesn't exist.

**Host Flexibility**: The `--host` option allows connecting to remote Ollama instances. All API calls (both `/v1` and `/api/generate` endpoints) use the configurable `ollama_host` parameter.

**Progress Tracking**: Rich progress bars show spinner, completion count, elapsed time, ETA, and dynamic rate display (img/s or s/img depending on speed).

### Image Processing Flow

1. Calculate SHA-256 checksums to detect and filter duplicates
2. Resize images to max 1024px (maintains aspect ratio)
3. Convert to base64 JPEG
4. Send to Ollama vision model with JSON schema
5. Parse structured response into `ImageAnalysis`
6. Generate 4-word descriptive filename with conflict resolution
7. Rename or move file to destination (or dry-run preview)

### Important Implementation Details

- SHA-256 checksums calculated in 8KB chunks for memory efficiency
- Images are resized before analysis to improve performance and reduce token usage
- Confidence scores are normalized to 0.1-1.0 range (models sometimes return out-of-range values)
- Filename conflicts are resolved with automatic numbering (`_1`, `_2`, etc.)
- Filenames are 4 words: descriptive, lowercase, underscores, max 40 chars
- Original file extensions are preserved
- Destination directories are created automatically if they don't exist
- Progress bar updates with each batch, showing real-time ETA and rate calculations

## Prerequisites

- Python 3.11+
- Ollama installed and running (`ollama serve`)
- Vision models downloaded: `ollama pull llava:latest` or `ollama pull gemma3:latest`

## Testing

Use `--test` flag to verify Ollama connectivity and model availability without processing images.

## Ollama Configuration

Default host: `http://localhost:11434`

To use a remote Ollama instance, pass `--host` with the full URL. The code automatically handles path construction for both OpenAI-compatible (`/v1`) and native (`/api/generate`) endpoints.

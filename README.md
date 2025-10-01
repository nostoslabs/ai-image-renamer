# AI Image Renamer

A command-line tool for intelligent image renaming using local vision models. Analyzes image content via Ollama and generates descriptive filenames with structured outputs and type safety.

## Features

- Vision-based content analysis using LLaVA or Gemma3 models
- Structured JSON outputs with Pydantic validation
- Built-in model performance comparison and benchmarking
- Filesystem-safe filename generation with conflict resolution
- Comprehensive content analysis (subject, style, colors, setting)
- Dry-run mode for safe preview before making changes
- Multiple execution methods: uv/uvx, direct Python
- Progress tracking and error handling for batch operations

## Installation

### Prerequisites

Install [Ollama](https://ollama.ai) and download vision models:

```bash
ollama pull llava:latest    # recommended for most use cases
ollama pull gemma3:latest   # alternative multimodal model
ollama serve
```

### Usage

Clone and run with uv:

```bash
git clone https://github.com/nostoslabs/ai-image-renamer
cd ai-image-renamer
uv sync

# Test model availability
uv run image-renamer --test

# Preview changes (recommended)
uv run image-renamer /path/to/images --dry-run

# Rename images
uv run image-renamer /path/to/images
```

Alternative execution methods:

```bash
# With uvx (from anywhere)
uvx --from git+https://github.com/nostoslabs/ai-image-renamer image-renamer /path/to/images --dry-run

# Direct Python execution
python image_renamer.py /path/to/images --dry-run
```

## Example Output

```
IMG_1234.jpg → sunset_mountain_landscape.jpg
DSC_5678.png → neon_city_night_skyline.png
photo.webp → abstract_blue_waves.webp
```

## Command Reference

```bash
uv run image-renamer [DIRECTORY] [OPTIONS]

Options:
  --test                  Test model availability
  --dry-run              Preview changes without renaming
  --model TEXT           Ollama model [default: llava:latest]
  --host TEXT            Ollama host URL [default: http://localhost:11434]
  --max-files INTEGER    Limit number of files to process
  --concurrent, -c INT   Number of concurrent requests [default: 2]
  --compare              Compare model performance
  --help                 Show help message
```

Examples:

```bash
# Test model connectivity
uv run image-renamer --test

# Preview changes
uv run image-renamer ~/Pictures --dry-run

# Process with limits
uv run image-renamer ~/Pictures --max-files 10

# Process with concurrent requests (faster)
uv run image-renamer ~/Pictures --concurrent 4
uv run image-renamer ~/Pictures -c 4  # short form

# Compare models
uv run image-renamer ~/Pictures --compare

# Use alternative model
uv run image-renamer ~/Pictures --model gemma3:latest

# Use remote Ollama instance
uv run image-renamer ~/Pictures --host http://192.168.1.100:11434

# Combine options for maximum speed
uv run image-renamer ~/Pictures --host http://192.168.1.100:11434 --concurrent 5
```

## Performance

Benchmark results on 4K wallpapers:

| Model | Speed | Accuracy | Confidence | Notes |
|-------|-------|----------|------------|-------|
| LLaVA | 4.8s/image | High | 95% | Recommended for most use cases |
| Gemma3 | 25.5s/image | Good | 85% | Better for abstract content |

LLaVA is recommended due to 5x faster processing and higher accuracy.

**Concurrent Processing:**
- Default: 2 concurrent requests (safe for most systems)
- Use `--concurrent 4-5` for faster processing on powerful machines
- Higher concurrency = faster overall processing but more CPU/memory usage
- Remote Ollama servers can typically handle higher concurrency

## Output Schema

The tool generates structured JSON output validated with Pydantic:

```python
class ImageAnalysis(BaseModel):
    main_subject: str           # "mountain", "cityscape", "abstract"
    style: str                  # "minimalist", "vibrant", "sunset"
    dominant_colors: List[str]  # ["blue", "orange"] (max 3)
    setting: Optional[str]      # "urban", "forest", "space"
    filename_suggestion: str    # "sunset_mountain_landscape"
    confidence: float           # 0.1 to 1.0
```

## Supported Formats

Input: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`

The tool automatically resizes large images (4K→1024px) for analysis while maintaining original format and quality. Filename conflicts are avoided through automatic numbering.

## Technical Details

**Architecture:**
- Modular design following SOLID principles
- Pydantic AI with direct Ollama integration
- JSON schema validation for structured outputs
- Async processing with progress tracking
- Type-safe validation and error handling

**Module Structure:**
```
src/image_renamer/
├── models.py              # Pydantic data models
├── image_processor.py     # Image resizing & encoding
├── ollama_client.py       # Ollama API abstraction
├── file_renamer.py        # File operations & naming
├── performance_tracker.py # Metrics & display
└── cli.py                 # CLI interface & orchestration
```

**Features:**
- Metrics tracking (success rates, timing, confidence)
- Graceful error handling and recovery
- Memory-efficient image processing
- Batch processing for large directories
- Conflict-free filename generation

## Requirements

- Python 3.11+
- 4-8GB RAM (depending on model)
- ~5GB storage for models
- Internet connection for initial model download

## Privacy

All processing happens locally. Images never leave your machine, no API keys required, no cloud services used.

## Advanced Usage

Custom model configuration:
```bash
# Use specific model versions
uv run image-renamer ~/Pictures --model llava:13b

# Connect to remote Ollama server
uv run image-renamer ~/Pictures --host http://192.168.1.100:11434

# Use remote with custom model
uv run image-renamer ~/Pictures --host http://myserver:11434 --model llava:13b

# Batch processing with system tools
find ~/Pictures -name "*.jpg" -type f | head -20 | \
  xargs -I {} dirname {} | sort -u | \
  xargs -I {} uv run image-renamer {} --dry-run
```

Process multiple directories:
```bash
for dir in ~/Pictures/*/; do
    uv run image-renamer "$dir" --dry-run
done
```

## Development

```bash
git clone https://github.com/nostoslabs/ai-image-renamer
cd ai-image-renamer
uv sync
uv run image-renamer --test
```

## License

MIT - see [LICENSE](LICENSE) for details.
---
title: depth-view
emoji: 🏔
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# depth-view

Upload a photo, get an interactive 3D point cloud. Uses monocular depth estimation to infer depth from a single image, then projects pixels into 3D space with synthetic camera intrinsics.

## Features

- **Monocular depth estimation** using Depth Anything V2 (24.8M params, runs on CPU)
- **Interactive 3D point cloud** via Plotly — rotate, zoom, and pan the reconstructed scene
- **Depth colormap** visualization with configurable colormaps (turbo, viridis, plasma, etc.)
- **Gradio UI** for interactive demos with adjustable FOV and point density
- **FastAPI** with Pydantic-validated endpoints for programmatic access
- **Hydra config** with model variants and serving modes
- Runs on CPU — no GPU required

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)

### Install

```bash
just install
```

### Launch Demo

```bash
just demo                          # Default: Depth Anything V2 Small
just demo model=base               # Higher quality: Depth Anything V2 Base
```

### Launch API Server

```bash
just serve serving=api
```

### Run Tests

```bash
just test-fast                     # Skip model downloads
just test                          # All tests (downloads model)
```

## How It Works

```
Image (JPEG/PNG)
       |
       v
Depth Anything V2 → relative depth map (H x W)
       |
       ├──→ Colormap → depth visualization
       |
       └──→ Backprojection (synthetic intrinsics, configurable FOV)
              |
              v
         3D point cloud (N x 3) + RGB colors
              |
              v
         Plotly Scatter3d → interactive viewer
```

The depth model produces relative inverse depth — it doesn't know absolute distances, but it accurately captures which parts of the scene are closer vs. farther. The backprojection uses an assumed field of view (default 60°) to convert 2D pixels + depth into 3D coordinates.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model info |
| `/depth` | POST | Returns depth colormap as base64 PNG |
| `/pointcloud` | POST | Returns 3D point cloud + depth map |

## Configuration

```bash
# Switch model variant
just demo model=base

# Change serving mode
just serve serving=api

# Adjust visualization defaults via Hydra overrides
just demo visualization.max_points=30000 visualization.colormap=viridis
```

## Project Structure

```
depth-view/
├── pyproject.toml
├── justfile
├── app.py                      # HuggingFace Spaces entrypoint
├── src/depthview/
│   ├── models/                 # Depth estimator + point cloud math
│   ├── schemas/                # Pydantic request/response models
│   ├── configs/                # Hydra YAML + structured configs
│   ├── serving/                # FastAPI + Gradio apps
│   └── visualization/          # Depth colormap + 3D plot rendering
└── tests/                      # 35 tests (fast + slow split)
```

## Development

```bash
just lint                          # Check code style (ruff)
just fix                           # Auto-fix lint + format
just check                         # lint + test-fast
```

## License

MIT

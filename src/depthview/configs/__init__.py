"""Structured Hydra configs with type safety via dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    name: str = "small"
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "cpu"


@dataclass
class VisualizationConfig:
    colormap: str = "turbo"
    max_points: int = 15000
    point_size: float = 2.0
    default_fov: float = 60.0


@dataclass
class ServingConfig:
    mode: str = "gradio"
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def register_configs() -> None:
    """Register all structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=AppConfig)
    cs.store(
        group="model",
        name="small",
        node=ModelConfig(
            name="small",
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
        ),
    )
    cs.store(
        group="model",
        name="base",
        node=ModelConfig(
            name="base",
            model_id="depth-anything/Depth-Anything-V2-Base-hf",
        ),
    )
    cs.store(
        group="serving",
        name="gradio",
        node=ServingConfig(mode="gradio", port=7860),
    )
    cs.store(
        group="serving",
        name="api",
        node=ServingConfig(mode="api", host="0.0.0.0", port=8000),
    )

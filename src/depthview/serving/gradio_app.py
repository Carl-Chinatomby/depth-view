"""Gradio Blocks UI for interactive depth estimation and 3D visualization."""

from __future__ import annotations

import numpy as np
from PIL import Image

from depthview.models.depth import DepthEstimator
from depthview.models.pointcloud import depth_to_pointcloud, subsample
from depthview.visualization.render import create_3d_plot, create_depth_colormap

# Module-level estimator, loaded lazily on first request
_estimator: DepthEstimator | None = None


def get_estimator(model_id: str = "depth-anything/Depth-Anything-V2-Base-hf") -> DepthEstimator:
    global _estimator
    if _estimator is None:
        _estimator = DepthEstimator(model_name=model_id)
    return _estimator


def process_image(
    image: Image.Image | np.ndarray | None,
    fov_deg: float,
    max_points: int,
    colormap: str,
):
    """Process an uploaded image through the depth pipeline.

    Returns (depth_colormap, plotly_figure) for Gradio outputs.
    """
    if image is None:
        return None, None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    estimator = get_estimator()
    depth = estimator.estimate(image)

    depth_img = create_depth_colormap(depth, colormap=colormap)

    rgb = np.array(image.convert("RGB"))
    points, colors = depth_to_pointcloud(depth, rgb, fov_deg=fov_deg)
    points, colors = subsample(points, colors, max_points=max_points)
    fig = create_3d_plot(points, colors)

    return depth_img, fig


def build_interface():
    """Build the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(
        title="depth-view",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# depth-view\nUpload a photo to see its depth map and interactive 3D point cloud."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                fov_slider = gr.Slider(
                    minimum=20, maximum=140, value=60, step=5, label="Field of View (degrees)"
                )
                points_slider = gr.Slider(
                    minimum=5000, maximum=100000, value=50000, step=5000, label="Max Points"
                )
                colormap_dropdown = gr.Dropdown(
                    choices=["turbo", "viridis", "plasma", "inferno", "magma"],
                    value="turbo",
                    label="Depth Colormap",
                )
                run_btn = gr.Button("Estimate Depth", variant="primary")

            with gr.Column(scale=2):
                depth_output = gr.Image(type="pil", label="Depth Map")
                plot_output = gr.Plot(label="3D Point Cloud")

        run_btn.click(
            fn=process_image,
            inputs=[image_input, fov_slider, points_slider, colormap_dropdown],
            outputs=[depth_output, plot_output],
        )

        image_input.change(
            fn=process_image,
            inputs=[image_input, fov_slider, points_slider, colormap_dropdown],
            outputs=[depth_output, plot_output],
        )

    return demo


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    from depthview.configs import register_configs

    register_configs()

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        global _estimator
        _estimator = DepthEstimator(
            model_name=cfg.model.model_id,
            device=cfg.model.device,
        )
        demo = build_interface()
        demo.launch(
            server_name=cfg.serving.host,
            server_port=cfg.serving.port,
            share=cfg.serving.share,
        )

    main()

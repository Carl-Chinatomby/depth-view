"""HuggingFace Spaces entrypoint."""

from depthview.serving.gradio_app import build_interface

demo = build_interface()

if __name__ == "__main__":
    demo.launch()

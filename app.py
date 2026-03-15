"""HuggingFace Spaces entrypoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from depthview.serving.gradio_app import build_interface

demo = build_interface()

if __name__ == "__main__":
    demo.launch()

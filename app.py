"""Marginalia Hugging Face Spaces Entrypoint.

Serves the Marginalia Flask review dashboard within a Hugging Face Gradio ZeroGPU space.
Mounts the complete Flask WSGI application into FastAPI/Gradio and satisfies ZeroGPU supervisor.
"""

# CRITICAL: `import spaces` MUST be the very first import before torch, sentence_transformers, or dashboard
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

import os
import gradio as gr
from a2wsgi import WSGIMiddleware
from dashboard import app as flask_app

PORT = int(os.environ.get("PORT", 7860))

# ZeroGPU worker function registered with Gradio graph
if has_spaces:
    @spaces.GPU
    def zero_gpu_worker(prompt: str) -> str:
        """Registered GPU task for Hugging Face ZeroGPU supervisor."""
        return f"GPU Active: {prompt[:30]}"
else:
    def zero_gpu_worker(prompt: str) -> str:
        return f"CPU: {prompt[:30]}"

# Convert Flask WSGI app to ASGI
wsgi_app = WSGIMiddleware(flask_app)

with gr.Blocks(
    title="Marginalia — Autonomous Research Digest Review Dashboard",
    theme=gr.themes.Base(),
    css="footer {visibility: hidden} .gradio-container {padding: 0 !important; max-width: 100% !important;}"
) as demo:
    # Hidden components to register with ZeroGPU supervisor
    with gr.Row(visible=False):
        dummy_input = gr.Textbox(value="Marginalia", visible=False)
        dummy_output = gr.Textbox(visible=False)
        dummy_btn = gr.Button("GPU Check", visible=False)
        dummy_btn.click(zero_gpu_worker, inputs=dummy_input, outputs=dummy_output)

    # Full-height embedded dashboard
    gr.HTML(
        '<div style="width: 100%; height: 96vh; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">'
        '<iframe src="/dashboard/" style="width: 100%; height: 100%; border: none;"></iframe>'
        '</div>'
    )

# Mount the Flask application onto Gradio's underlying Starlette/FastAPI server
demo.app.mount("/dashboard", wsgi_app)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, show_api=False)

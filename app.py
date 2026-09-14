"""Marginalia Hugging Face Spaces Entrypoint.

Serves the Marginalia Flask review dashboard natively inside Hugging Face Gradio ZeroGPU.
Renders the complete review UI directly in the DOM without nested iframes.
"""

# CRITICAL: `import spaces` MUST be the very first import before torch, sentence_transformers, or dashboard
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

import os
from fastapi import Request
from fastapi.responses import JSONResponse
import gradio as gr
from dashboard import (
    get_rendered_dashboard_html,
    save_decision_record,
    load_digest,
    index_approved_paper,
)

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

# Generate the full interactive dashboard HTML
dashboard_content = get_rendered_dashboard_html()

with gr.Blocks(
    title="Marginalia — Autonomous Research Digest Review Dashboard",
    css=".gradio-container { max-width: 100% !important; padding: 0 !important; }"
) as demo:
    # Hidden components to register with ZeroGPU supervisor
    with gr.Row(visible=False):
        dummy_input = gr.Textbox(value="Marginalia", visible=False)
        dummy_output = gr.Textbox(visible=False)
        dummy_btn = gr.Button("GPU Check", visible=False)
        dummy_btn.click(zero_gpu_worker, inputs=dummy_input, outputs=dummy_output)

    # Render complete dashboard UI directly into the Gradio DOM (no iframes!)
    gr.HTML(dashboard_content)


# Handle AJAX review decision posts directly on FastAPI
@demo.app.post("/decide")
async def handle_decision(request: Request):
    data = await request.json()
    arxiv_id = data.get("arxiv_id", "").strip()
    decision = data.get("decision", "").strip().lower()

    if not arxiv_id or decision not in ("approve", "skip"):
        return JSONResponse({"status": "error", "message": "Invalid arxiv_id or decision"}, status_code=400)

    record = save_decision_record(arxiv_id, decision)
    indexed_file = None
    if decision == "approve":
        papers = load_digest()
        target_paper = next((p for p in papers if p.get("arxiv_id") == arxiv_id), None)
        if not target_paper:
            target_paper = next(
                (p for p in papers if p.get("arxiv_id", "").split("v")[0] == arxiv_id.split("v")[0]),
                None,
            )
        if target_paper:
            indexed_file = index_approved_paper(target_paper)

    return JSONResponse({
        "status": "success",
        "decision": decision,
        "arxiv_id": arxiv_id,
        "timestamp": record.get("timestamp"),
        "corpus_file": indexed_file.name if indexed_file else None,
    })


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)

"""Marginalia Hugging Face Spaces Entrypoint.

Starts the Marginalia Flask review dashboard on port 7860 (Hugging Face standard).
Compatible with both Docker and Gradio Space SDKs.
"""

import os
from dashboard import app

# Hugging Face Spaces routes public traffic to port 7860
PORT = int(os.environ.get("PORT", 7860))

if __name__ == "__main__":
    print(f"Starting Marginalia Dashboard for Hugging Face Spaces on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)

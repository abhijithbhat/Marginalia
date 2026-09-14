"""Marginalia Hugging Face Spaces Entrypoint.

Starts the Marginalia Flask review dashboard on port 7860 (Hugging Face standard).
Includes ZeroGPU hook required by Hugging Face Spaces ZeroGPU runtime.
"""

import os

# Hugging Face ZeroGPU compatibility hook
try:
    import spaces

    @spaces.GPU
    def zero_gpu_marker():
        """Registers with Hugging Face ZeroGPU runtime supervisor."""
        return True

    # Invoke once at import time so ZeroGPU supervisor detects it
    zero_gpu_marker()
    print("[ZeroGPU] @spaces.GPU marker registered successfully.")
except ImportError:
    print("[ZeroGPU] spaces package not installed; skipping GPU registration.")
except Exception as e:
    print(f"[ZeroGPU] Registration info: {e}")

from dashboard import app

# Hugging Face Spaces routes public traffic to port 7860
PORT = int(os.environ.get("PORT", 7860))

if __name__ == "__main__":
    print(f"Starting Marginalia Dashboard for Hugging Face Spaces on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)

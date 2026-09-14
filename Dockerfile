FROM python:3.11-slim

# Create a non-root user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Install minimal web dependencies
COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy all application files (including digest.json, corpus, docs, templates)
COPY --chown=user . .

# Hugging Face Spaces requires listening on port 7860
EXPOSE 7860

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "120", "dashboard:app"]

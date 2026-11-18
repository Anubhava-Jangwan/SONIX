# SONIX detection backend for Hugging Face Spaces (SDK: docker).
#
# Vercel cannot host this: /ws needs a long-lived WebSocket server, the engine
# keeps call state in memory between requests, and torch + XLS-R blow the 250MB
# function cap. A Space is a normal container, so all three stop being problems.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; HF_HOME must be writable or the
# wav2vec2 download has nowhere to land.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Shallow clone — full history is ~253MB, and the trained heads we need are
# only ~9MB under outputs/models/ (committed, see .gitignore:9-11).
ARG REPO=https://github.com/Anubhava-Jangwan/SONIX
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} .

# CPU wheels. The cu126 index in CLAUDE.md is for local GPU boxes; a free
# Space has no GPU, and the CUDA wheels are ~2GB of dead weight here.
RUN pip install --no-cache-dir --user \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# server.py:55 reads this to let the Vercel-hosted page call us cross-origin.
# Comma-separated; override in Space Settings → Variables without a rebuild.
ENV SONIX_ALLOWED_ORIGINS=https://sonix-eosin.vercel.app

EXPOSE 7860

# -m, not a file path: server.py does `from realtime...` with no sys.path shim,
# so it only resolves from the repo root (README.md:41).
# webrtc mode skips the AudioSocket TCP listener (server.py:849) but still
# registers every /api route and /ws.
CMD ["python", "-m", "realtime.server", \
     "--mode", "webrtc", \
     "--host", "0.0.0.0", \
     "--ws-port", "7860", \
     "--output-dir", "/tmp/calls"]

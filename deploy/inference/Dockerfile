# Build from the repository root; weights are supplied at runtime.
FROM python:3.12-slim-bookworm
ARG MODEL_EXTRAS=vision-server
ARG TORCH_PACKAGES="torch==2.8.0 torchvision==0.23.0"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY jiuwensymbiosis ./jiuwensymbiosis
COPY jiuwensymbiosis_gui ./jiuwensymbiosis_gui
COPY examples ./examples
COPY scripts/calibrate ./scripts/calibrate
RUN python -m pip install ${TORCH_PACKAGES} --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install ".[${MODEL_EXTRAS}]"
COPY deploy/inference/healthcheck.py /app/healthcheck.py
RUN useradd --create-home inference
USER inference
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "-m", "jiuwensymbiosis.serving.grounding_dino_sam2_server", "--help"]

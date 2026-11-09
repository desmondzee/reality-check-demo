# Contract-first policy image. Base image is not mandated.
# Built on the DGX Spark (linux/aarch64). TORCH_INDEX selects the wheel:
# CUDA 12.8 matches GB10 (sm_121) PyTorch builds. Override for CPU.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.9.0 --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir numpy pillow

COPY . /app
RUN pip install --no-cache-dir ./rc_sdk

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "rc_sdk", "--manifest", "manifest.json", \
     "--policy", "policy.act_policy:ActPolicy"]

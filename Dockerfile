# Contract-first policy image. The base image is not mandated.
# Vision + IK only: no torch. MuJoCo is used for the kinematic model.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy pillow "mujoco==3.5.0"

COPY . /app
RUN pip install --no-cache-dir ./rc_sdk

ENV PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa
CMD ["python", "-m", "rc_sdk", "--manifest", "manifest.json", \
     "--policy", "policy.cup_policy:CupPolicy"]

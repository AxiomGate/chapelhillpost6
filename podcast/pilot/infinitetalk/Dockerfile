# Pilot image for MeiGen-AI/InfiniteTalk -- NOT part of the served cluster.
#
# This directory is deliberately outside docker/ and cluster/. It is not built
# by any compose file, not registered in cluster.yaml, and nothing in the
# production pipeline references it. The point is to answer one question --
# what fps does InfiniteTalk actually render at on this hardware, and does the
# output look better than MuseTalk's -- without risking the pipeline that is
# already working. Delete this whole directory freely if the pilot fails; it
# owns nothing outside itself.
#
# UNLIKE docker/Dockerfile.avatar, these version pins are NOT a measured
# `pip freeze` from a working container -- there is no working container yet.
# They are InfiniteTalk's own stated requirements. Expect this build to need at
# least one more iteration before it succeeds; the MuseTalk image took three
# for the same category of reason (mmcv/torch ABI mismatches). The most likely
# failure point is flash-attn: it compiles CUDA kernels against a specific
# torch+CUDA ABI and has no wheel for every combination, so it may need to
# build from source (slow, needs the CUDA toolkit present, which is why this
# uses a `devel` base image rather than `runtime`) or need a different prebuilt
# wheel URL substituted in.
#
# Python 3.10, not 3.11 -- InfiniteTalk's own requirements state 3.10
# specifically, and this pilot has no reason to fight that.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      ffmpeg git build-essential ninja-build libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pilot/infinitetalk/constraints.txt /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN python3.10 -m pip install --no-cache-dir "pip==24.2" "setuptools==69.5.1" "wheel"

# Versions from InfiniteTalk's own install instructions, cu121 throughout --
# there is no cu124 index for this repo's pinned torch/xformers pair, unlike
# the avatar image where cu121 was a deliberate mismatch. Here it is what the
# repo actually asks for.
RUN python3.10 -m pip install --no-cache-dir \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu121

RUN python3.10 -m pip install --no-cache-dir \
      xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# The likely failure point -- see the header. If this step fails or hangs for
# a very long time, that is the signal to stop and find a prebuilt wheel that
# matches torch 2.4.1 + cu121 + Python 3.10 exactly rather than let it compile.
RUN python3.10 -m pip install --no-cache-dir flash_attn==2.7.4.post1

ARG INFINITETALK_REF=main
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk /opt/InfiniteTalk \
    && cd /opt/InfiniteTalk \
    && git checkout "${INFINITETALK_REF}" \
    && echo "InfiniteTalk commit: $(git rev-parse HEAD)" | tee /opt/InfiniteTalk/COMMIT.txt

RUN python3.10 -m pip install --no-cache-dir librosa
RUN python3.10 -m pip install --no-cache-dir -r /opt/InfiniteTalk/requirements.txt

# Fail the build rather than the first render.
RUN python3.10 -c "\
import numpy, torch, torchvision, torchaudio, xformers, flash_attn; \
print('numpy', numpy.__version__); \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('xformers', xformers.__version__); \
print('flash_attn', flash_attn.__version__); \
print('ALL IMPORTS OK')"

COPY pilot/infinitetalk/run_pilot.py /opt/InfiniteTalk/run_pilot.py
WORKDIR /opt/InfiniteTalk

# No CMD. This is a one-shot pilot invoked manually with `docker run`, not a
# served worker with a health endpoint -- see README.md for the actual command.

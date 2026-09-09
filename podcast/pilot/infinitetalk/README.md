# InfiniteTalk pilot

Answers one question: what fps does InfiniteTalk actually render at on this
hardware, and does the output look better than MuseTalk's, on one real 60-second
chunk. Nothing here is wired into the served pipeline -- no compose file
references this directory, `cluster.yaml` does not know it exists, and the
production `podcast-avatar` containers on node-b and node-c are untouched by
any of this. Safe to run, safe to delete afterward if it doesn't pan out.

## Where this runs: amdtower

`amdtower` -- Unraid 7.3, Ryzen 9 9950X, RTX 3090, Tailscale `100.77.255.110`.
Not a cluster member: it has no NFS mount of the podcast share and isn't in
`cluster.yaml`. That's deliberate for a one-shot pilot -- it only needs two
small files (one audio chunk, the base loop video), so this copies them
directly with `scp` rather than exporting the share to a fourth machine before
knowing whether this model is even the right call.

Node-a is reachable from amdtower over Tailscale at `100.73.169.24`.

## 0. On amdtower — clone the repo

```bash
mkdir -p /mnt/user/infinitetalk-pilot
cd /mnt/user/infinitetalk-pilot
git clone https://github.com/AxiomGate/chapelhillpost6.git repo
cd repo
git checkout claude/podcast-video-pipeline-0rs4ns
```

## 1. On amdtower — confirm the GPU actually reaches a container

`nvidia-smi` working on the host (already confirmed) is not the same thing as
Docker being able to pass the GPU into a container. Cheap to check before
downloading tens of GB on the strength of an assumption:

```bash
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

`--device nvidia.com/gpu=all`, not `--gpus all` -- on this box, plain `--gpus
all` failed with "AMD CDI spec not found" despite an NVIDIA-only GPU, because
Docker's default GPU-vendor resolution guessed wrong. The toolkit and the real
CDI spec (`/etc/cdi/nvidia.yaml`) were both present and correct; naming the
device explicitly sidesteps whatever picked the wrong vendor.

If that doesn't print the 3090, check whether `nvidia-ctk` exists
(`which nvidia-ctk`) and whether `/etc/cdi/nvidia.yaml` exists
(`ls /etc/cdi/`). If both are missing, the Nvidia-Driver plugin needs
(re)installing from the Unraid web UI. If both are present but this still
fails, `nvidia-ctk runtime configure --runtime=docker` followed by
`/etc/rc.d/rc.docker restart` registers the runtime properly -- that was
enough to fix a Docker GPU integration that survived a full system rebuild
with the driver intact but nothing wiring it into Docker.

## 2. On amdtower — check free disk space

The weights are large: Wan2.1-I2V-14B-480P is a 14B-parameter model (tens of
GB at fp16), plus the wav2vec audio encoder and InfiniteTalk's own conditioning
weights on top. Budget **50GB+ free**:

```bash
df -h /mnt/user
```

## 3. On amdtower — fetch the weights, once

Unraid's host OS has no Python or pip by default -- deliberately minimal, same
as every other Unraid box in this project. Everything here runs in a
throwaway container instead, matching how the rest of the pipeline works:

```bash
WEIGHTS=/mnt/user/infinitetalk-pilot/weights
mkdir -p "$WEIGHTS"

docker run -d --name infinitetalk-weights -v "$WEIGHTS:/weights" python:3.10-slim sh -c "
    pip install -q -U huggingface_hub &&
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /weights/Wan2.1-I2V-14B-480P &&
    hf download TencentGameMate/chinese-wav2vec2-base --local-dir /weights/chinese-wav2vec2-base &&
    hf download MeiGen-AI/InfiniteTalk --local-dir /weights/InfiniteTalk
"
```

Check on it with short commands that connect, print, and exit -- not `docker
logs -f`, which is a long-lived stream that dies the moment a flaky
connection blips, and on a wireless-only site that's a real risk rather than
a theoretical one:

```bash
docker logs --tail 15 infinitetalk-weights
du -sh /mnt/user/infinitetalk-pilot/weights/* 2>/dev/null
```

Run that pair whenever you reconnect. **Don't `docker rm` the container until
`docker ps -a --filter name=infinitetalk-weights --format '{{.Status}}'`
reads `Exited (0)`** -- removing it before checking the exit status throws
away the only copy of the error if something went wrong.

`hf`, not `huggingface-cli` -- newer `huggingface_hub` releases renamed the CLI and dropped
the `[cli]` install extra. The old command prints a deprecation notice and exits
without downloading anything, which fails fast with no obvious error, so this
is worth getting right rather than discovering it after a wasted run.

Named and detached (`-d --name`) rather than `--rm` in the foreground: the
container keeps running even if this SSH session drops, and `docker logs -f`
can be reattached at any time without affecting it. Check on it, don't remove
it, until `docker ps -a --filter name=infinitetalk-weights --format
'{{.Status}}'` reads `Exited (0)` -- pulling it before confirming success
throws away the only copy of the error.

This step is the long pole -- let it run, it doesn't need attention.

## 4. On node-a — copy one real chunk to amdtower

Use a chunk from an episode that's already been through MuseTalk, so the
comparison is apples-to-apples: same audio, same base loop, watch both back
to back afterward.

```bash
mkdir -p /tmp/pilot-input
ffmpeg -y -i /mnt/user/podcast/output/firstrun/archive/2026-08-26_20260826-212251_published/master_podcast_2122.wav \
    -ss 0 -t 60 -c copy /tmp/pilot-input/chunk_60s.wav
cp /mnt/user/podcast/clients/firstrun/assets/avatar/base_loop.mp4 \
   /tmp/pilot-input/base_loop.mp4

scp /tmp/pilot-input/chunk_60s.wav /tmp/pilot-input/base_loop.mp4 \
    root@100.77.255.110:/mnt/user/infinitetalk-pilot/input/
```

(That's the finished episode from the most recent successful run. Path to
`base_loop.mp4` may differ -- check `clients/firstrun/show.yaml`'s
`avatar.base_loop` if that `cp` fails.)

`scp` will prompt for amdtower's root password the first time; say yes to the
host key prompt.

## 5. On amdtower — build the pilot image

```bash
cd /mnt/user/infinitetalk-pilot/repo/podcast
docker build -f pilot/infinitetalk/Dockerfile -t infinitetalk-pilot .
```

**Expect this to need at least one fix-and-rebuild cycle.** The version pins
here are InfiniteTalk's own stated requirements, not a measured working
`pip freeze` -- there's no working container yet to measure. The likely
failure point is the `flash_attn==2.7.4.post1` install: it compiles CUDA
kernels against a specific torch+CUDA ABI, and if pip falls back to a from-
source build it can take 30+ minutes or fail outright. If that happens, stop
and paste me the error rather than let it grind -- there is very likely a
prebuilt wheel for this exact torch 2.4.1 + cu121 + Python 3.10 combination
that avoids the compile entirely, but which one depends on the actual failure.

## 6. On amdtower — run it

```bash
mkdir -p /mnt/user/infinitetalk-pilot/output

docker run --rm --device nvidia.com/gpu=all \
    -v "$WEIGHTS:/weights:ro" \
    -v /mnt/user/infinitetalk-pilot/input:/in:ro \
    -v /mnt/user/infinitetalk-pilot/output:/out \
    infinitetalk-pilot \
    python3.10 run_pilot.py \
        --base-loop /in/base_loop.mp4 \
        --audio /in/chunk_60s.wav \
        --out /out/infinitetalk_pilot.mp4 \
        --ckpt-dir /weights/Wan2.1-I2V-14B-480P \
        --wav2vec-dir /weights/chinese-wav2vec2-base \
        --infinitetalk-dir /weights/InfiniteTalk/single/infinitetalk.safetensors
```

This prints a job config, runs the render, and ends with:

```
=== pilot result ===
output:          /out/infinitetalk_pilot.mp4
render_seconds:  ...
frames:          ...
output duration: ...
fps_effective:   ...   (MuseTalk baseline on the same hardware: 2.93)
```

**If it exits with code -9**, that's an OOM -- the same signature this project
has already hit once with MuseTalk. `--low-vram` is on by default in
`run_pilot.py`, so the next step is the fp8-quantized weights:

```bash
docker run --rm -v "$WEIGHTS:/weights" python:3.10-slim sh -c "
    pip install -q -U huggingface_hub &&
    hf download MeiGen-AI/InfiniteTalk --include 'quant_models/*' --local-dir /weights/InfiniteTalk
"
```

then re-run pointing `--infinitetalk-dir` at
`$WEIGHTS/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors` and
adding `--quant fp8 --quant_dir <same path>` to the command inside
`run_pilot.py` (not yet wired as a flag here -- add it if you hit this).

## 7. Judge it

`infinitetalk_pilot.mp4` lands at `/mnt/user/infinitetalk-pilot/output/` on
amdtower. Copy it to your desktop over amdtower's own share before watching --
SMB-over-Tailscale stutters even on small files, and that cost a whole
debugging round earlier in this project for a completely unrelated reason.

Watch it next to the MuseTalk output for the same chunk (the archive folder
copied from in step 4 also has `avatar_2122.mp4`, the silent MuseTalk render
for comparison). Report back:

- the `fps_effective` number
- whether it needed `--low-vram` / fp8 to fit, or ran clean
- how the mouth and face actually look next to MuseTalk's -- sharper teeth
  and mouth interior is the thing MuseTalk structurally can't do (it only
  inpaints a patch), so that's the specific thing to look for
- whether the head motion still looks like the real recorded footage, or
  whether it's visibly generating new motion not in the source

That's what turns this from a "should be better" into a decision.

# InfiniteTalk pilot

Answers one question: what fps does InfiniteTalk actually render at on this
hardware, and does the output look better than MuseTalk's, on one real 60-second
chunk. Nothing here is wired into the served pipeline -- no compose file
references this directory, `cluster.yaml` does not know it exists, and the
production `podcast-avatar` containers on node-b and node-c are untouched by
any of this. Safe to run, safe to delete afterward if it doesn't pan out.

## Where to run this

Any machine with a free RTX 3090 that is **not currently serving a live
episode render**. The new (4th) 3090 is the obvious choice if it's reachable
and idle -- otherwise node-b or node-c between episodes, but never while
`podcast-avatar` is mid-render on that box: this pilot and the production
container would fight over the same 24GB.

Everything below assumes you're on that machine, in the repo checkout
(`/mnt/user/podcast/repo` on node-a, `/mnt/remotes/podcast/repo` on node-b/c).

## 1. Check free disk space before downloading anything

The weights are large: Wan2.1-I2V-14B-480P is a 14B-parameter model (tens of
GB at fp16), plus the wav2vec audio encoder and InfiniteTalk's own conditioning
weights on top. Budget **50GB+ free** on whatever local disk you point the
download at. Check first:

```bash
df -h /mnt/cache    # or wherever you're about to point --local-dir below
```

## 2. Fetch the weights, once, to node-local storage

Not the NFS share -- same reason MuseTalk's weights live at
`/mnt/user/appdata/podcast/musetalk-models` per node instead of on `/pipeline`:
a render should never wait on NFS to page in a multi-GB model.

```bash
pip install -U "huggingface_hub[cli]"

WEIGHTS=/mnt/user/appdata/podcast/infinitetalk-pilot-models   # adjust per node
mkdir -p "$WEIGHTS"

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir "$WEIGHTS/Wan2.1-I2V-14B-480P"
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    --local-dir "$WEIGHTS/chinese-wav2vec2-base"
huggingface-cli download MeiGen-AI/InfiniteTalk \
    --local-dir "$WEIGHTS/InfiniteTalk"
```

## 3. Build the pilot image

From the repo root (`podcast/`):

```bash
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

## 4. Grab one real chunk to render

Use a chunk from an episode that's already been through MuseTalk, so the
comparison is apples-to-apples: same audio, same base loop, and you can watch
both outputs back to back.

On node-a, pull one 60-second slice from an archived episode's master audio
and copy the base loop over:

```bash
mkdir -p /mnt/user/podcast/pilot-input
ffmpeg -y -i /mnt/user/podcast/output/firstrun/archive/<pick-a-run>/master_podcast_*.wav \
    -ss 0 -t 60 -c copy /mnt/user/podcast/pilot-input/chunk_60s.wav
cp /mnt/user/podcast/clients/firstrun/assets/avatar/base_loop.mp4 \
   /mnt/user/podcast/pilot-input/base_loop.mp4
```

(Path to `base_loop.mp4` may differ -- check
`clients/firstrun/show.yaml`'s `avatar.base_loop` if that doesn't exist.)

If you're running the pilot on node-b or node-c, that's already on
`/mnt/remotes/podcast/pilot-input/` over NFS.

## 5. Run it

```bash
docker run --rm --gpus all \
    -v "$WEIGHTS:/weights:ro" \
    -v /mnt/user/podcast/pilot-input:/in:ro \
    -v /mnt/user/podcast/pilot-input:/out \
    infinitetalk-pilot \
    python3.10 run_pilot.py \
        --base-loop /in/base_loop.mp4 \
        --audio /in/chunk_60s.wav \
        --out /out/infinitetalk_pilot.mp4 \
        --ckpt-dir /weights/Wan2.1-I2V-14B-480P \
        --wav2vec-dir /weights/chinese-wav2vec2-base \
        --infinitetalk-dir /weights/InfiniteTalk/single/infinitetalk.safetensors
```

(Adjust the two `-v` mount sources to wherever step 4 actually put the files
on this machine.)

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
huggingface-cli download MeiGen-AI/InfiniteTalk \
    --include "quant_models/*" --local-dir "$WEIGHTS/InfiniteTalk"
```

then re-run pointing `--infinitetalk-dir` at
`$WEIGHTS/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors` and
adding `--quant fp8 --quant_dir <same path>` to the command inside
`run_pilot.py` (not yet wired as a flag here -- add it if you hit this).

## 6. Judge it

Copy `infinitetalk_pilot.mp4` local before watching it -- SMB-over-Tailscale
stutters even on small files, and that cost a whole debugging round earlier in
this project for a completely unrelated reason. Copy it to your desktop over
the `podcast` share, not a network stream.

Watch it next to the MuseTalk output for the same chunk. Report back:

- the `fps_effective` number
- whether it needed `--low-vram` / fp8 to fit, or ran clean
- how the mouth and face actually look next to MuseTalk's -- sharper teeth
  and mouth interior is the thing MuseTalk structurally can't do (it only
  inpaints a patch), so that's the specific thing to look for
- whether the head motion still looks like the real recorded footage, or
  whether it's visibly generating new motion not in the source

That's what turns this from a "should be better" into a decision.

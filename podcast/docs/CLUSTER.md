# Multi-node deployment

Three Unraid servers, five GPU workers. Every model loads once when its container
starts and stays in VRAM forever — no reload per episode, no config swapping
between stages.

```
┌──── node-a ── 10.10.5.15 ── 3090 ──────────────────┐
│  /mnt/user/podcast   ← shared export, NFS           │
│  /mnt/cache/podcast-scratch  ← local NVMe           │
│                                                     │
│  orchestrator   CPU     review UI  :8420            │
│  tts-worker     3090    Chatterbox :8080            │
└─────────────────────────────────────────────────────┘
                       │ 1 GbE
      ┌────────────────┴────────────────┐
┌──── node-b ── .16 ────────┐  ┌──── node-c ── .17 ────────┐
│  3090 + A1000             │  │  3090 + A1000             │
│  mounts the export (NFS)  │  │  mounts the export (NFS)  │
│                           │  │                           │
│  avatar-worker  3090 :8081│  │  avatar-worker  3090 :8081│
│  media-worker  A1000 :8082│  │  media-worker  A1000 :8082│
└───────────────────────────┘  └───────────────────────────┘
```

Node-b and node-c are identical in shape; only their GPU UUIDs differ. Node-a is
deliberately the odd one out — it is a 4-core i7-7700 on a BIOSTAR TB250-BTC+
mining board, the weakest machine here, so it holds the storage, the CPU-only
orchestrator, and exactly one GPU job.

### Card history

The A1000s both used to be node-a's problem. That board has one full-width slot;
anything else lands in an x1 chipset slot, and the card that was there ran behind
a USB riser at Gen1 x1 — about 250 MB/s, a thirty-second of what the card can do.
Whisper tolerated it (load once, small audio in, text out). NVENC would not have.

As of 2026-08-15 one A1000 moved to node-b, a second was installed in node-c, and
node-a is single-card. Both A1000s now have real PCIe lanes on Threadripper
boards, which is why the media worker runs captions *and* encode rather than
captions alone.

**A card's UUID follows the card, not the slot.** When you move one between
hosts, move its value in `.env` to the new variable name unchanged.

## The two ideas that make it work

**One path, everywhere.** Every worker container mounts the share at
`/pipeline`, no matter where it lives on its host. Node-a serves
`/mnt/user/podcast`; node-b mounts that at `/mnt/remotes/podcast`; both
containers see `/pipeline`. So a job payload names a file and every node resolves
it to the same bytes. `translate_path` enforces this and refuses to dispatch a
job referencing anything outside the share — better a clear error than a
`FileNotFoundError` forty minutes into a render.

**Nothing large crosses 1 GbE.** The driving video for a 25-minute episode is
several gigabytes. It is never transmitted. Each avatar worker keeps a local copy
of the base loop on its NVMe scratch, builds its own ping-pong clip once, and
slices windows locally. Over the wire: an audio window in (~3 MB) and a rendered
clip back (~150 MB, about 1.5 seconds). That is the entire network budget.

## Why the pipeline is faster than the sum of its parts

Voice and avatar work at different granularities — voice in ~20-second chunks,
avatar in 2-minute windows. The scheduler accumulates finished voice chunks until
a window is full, dispatches it to an avatar worker immediately, and keeps
synthesizing. Avatar rendering starts about 90 seconds into the episode instead
of waiting 6–8 minutes for the whole voice track.

| | Single box | Three nodes, as built |
|---|---|---|
| Model loads per episode | 3 | **0** |
| Voice | 4–8 min | ~13 min, overlapped (measured) |
| Avatar | 30–45 min | **15–23 min** (estimated) |
| Captions + encode | 7–13 min | 7–13 min |
| **Total after approval** | **45–70 min** | **23–37 min** |

Only the voice number is measured: **1.42× realtime**, from a 32-chunk run that
took 173 s of GPU time for 4.1 minutes of audio — better than the 1.19× a single
8-second probe suggested, because model warm-up amortizes across the batch. An
18-minute show is therefore about 13 minutes on node-a's one card. Everything in
the avatar column is still an estimate; no full render has been timed.

### Where the next card goes

A **3090 for a second TTS worker**, in node-b or node-c. Not a third avatar, and
not another A1000.

Voice at ~13 minutes and avatar at ~20 across two workers are close enough that
a third avatar worker would drop avatar to ~13 and make voice the new bottleneck
— the card would buy nothing. A second TTS card takes voice to ~6 minutes and
leaves avatar as the honest limit. An A1000 cannot take this job: 8 GB is tight
against Chatterbox's 6 GB floor, and it runs 4–6× slower than a 3090 even when
it fits.

---

## Setup

### 1 — Shared storage on node-a

In the Unraid web UI, **Shares → Add Share**:

- Name `podcast`
- Use cache: **Yes** or **Prefer** — episode I/O is heavy and small-file, and the
  parity array will make it crawl
- Export: **NFS = Yes**, Security: **Private**, Rule: `10.10.5.0/24(rw,sec=sys)`

Then create the directory layout and the local scratch pool:

```bash
mkdir -p /mnt/user/podcast/{work,output,assets/{voice,avatar,brand,broll},config}
mkdir -p /mnt/cache/podcast-scratch
cp /path/to/repo/podcast/config/*.yaml /mnt/user/podcast/config/
```

Scratch must be on a **cache/NVMe pool, not the array**. It holds the driving
video, which is written and read constantly during a render.

### 2 — Mount the share on node-b and node-c

Same commands on both:

```bash
mkdir -p /mnt/remotes/podcast
mount -t nfs -o rw,soft,timeo=100,retrans=3,rsize=131072,wsize=131072 \
      10.10.5.15:/mnt/user/podcast /mnt/remotes/podcast
mkdir -p /mnt/cache/podcast-scratch /mnt/cache/podcast-scratch-media
```

`soft` rather than `hard`: with a hard mount, node-a going away leaves every
worker process blocked in uninterruptible I/O and the containers cannot even be
killed. A soft mount fails the job with an error instead, which the scheduler
already knows how to retry on another node.

Verify both directions before going further — this is the single most common
failure and it surfaces as confusing errors much later:

```bash
touch /mnt/remotes/podcast/_probe && ls -l /mnt/user/podcast/_probe   # on node-a
```

Make it survive reboot by adding the mount to `/boot/config/go`, or use the
**Unassigned Devices** plugin, which handles remounts more gracefully.

### 3 — GPU drivers and UUIDs

Install the **Nvidia Driver** plugin from Community Applications on **all three**
servers, then reboot. On each host:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
```

**Use UUIDs, never indices.** Indices reorder across reboots and after any
hardware change; on node-b and node-c that eventually points the avatar render at
the A1000, where it runs out of memory or crawls. A UUID is burned into the card
and follows it between machines, so moving a card means moving its value to the
new variable name — not looking it up again expecting a new number.

Copy `.env.example` to `.env` **on each node**, in the same directory you run
`docker compose` from, and fill in the UUIDs that node needs. Confirm the
substitution actually happened before starting anything:

```bash
docker compose -f docker-compose.node-b.yml config | grep NVIDIA_VISIBLE
```

An unset variable expands to an empty string, which hands the container *every*
GPU on the box rather than failing.

### 4 — Check for VRAM already in use

Other AI/LLM containers run on these boxes. Before deploying, on each host:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

The pipeline needs roughly **6 GB free on node-a's 3090** (voice), **8 GB on each
avatar 3090**, and **2.5 GB on each A1000**.

Node-a is the one that bites: Ollama shares that card and holds ~17 GB with a 27B
model resident, leaving under the 6 GB the voice worker needs. Two fixes, in
order of preference:

1. Set `OLLAMA_KEEP_ALIVE=30m` (it defaults to 24h here) and
   `OLLAMA_MAX_LOADED_MODELS=1`, so the card frees up between uses.
2. Move voice to node-b or node-c's 3090 — swap the `role` values in
   `cluster.yaml` and the service in the compose files. This costs you avatar
   capacity, so it is a last resort.

Workers check free VRAM at startup and refuse to load with a specific message
rather than failing mid-render. Tune the threshold per worker with
`REQUIRED_VRAM_MB`.

### 5 — Build and start the workers

```bash
# node-a
cd /mnt/user/podcast/repo/podcast/docker
docker compose -f docker-compose.node-a.yml up -d --build

# node-b
cd /mnt/remotes/podcast/repo/podcast/docker
docker compose -f docker-compose.node-b.yml up -d --build

# node-c
cd /mnt/remotes/podcast/repo/podcast/docker
docker compose -f docker-compose.node-c.yml up -d --build
```

First build pulls CUDA base images and PyTorch — expect 20–40 minutes and ~25 GB
per node. Model weights mount from `appdata`, so they survive rebuilds.

Each avatar node then needs its MuseTalk weights, 9.3 GB, once:

```bash
docker exec podcast-avatar sh -c \
  'python3.11 $(find /pipeline/repo -name fetch_musetalk_weights.py | head -1)'
docker restart podcast-avatar
```

The `find` is deliberate: whether the checkout on the share puts this at
`/pipeline/repo/scripts/` or `/pipeline/repo/podcast/scripts/` depends on how it
was cloned, and a wrong path here fails in a way that looks like a missing script
rather than a missing checkout.

Do not use MuseTalk's own `download_weights.sh`. It prints "All weights have been
downloaded successfully!" unconditionally, including when nothing transferred —
which is how two avatar nodes reported healthy for days while holding zero bytes.

### 6 — Verify

```bash
podcastpipe cluster
```

```
NODE         ROLE     STATE    GPU                       VRAM FREE
tts-a        tts      up       NVIDIA GeForce RTX 3090      6912 MB
avatar-b     avatar   up       NVIDIA GeForce RTX 3090     14208 MB
media-b      media    up       NVIDIA RTX A1000             5376 MB
avatar-c     avatar   up       NVIDIA GeForce RTX 3090     14208 MB
media-c      media    up       NVIDIA RTX A1000             5376 MB
```

`DOWN` with a reason is diagnostic, not a failure to guess at:

| Reason | Meaning |
|---|---|
| `connection refused` | Container not running, or port not published |
| `reachable but model not loaded` | Still loading (MuseTalk takes 60–90 s), or the load failed — check `docker logs` |
| `timed out` | Firewall, or wrong IP in `cluster.yaml` |

### 7 — Assets onto the share

```
/mnt/user/podcast/assets/voice/reference.wav
/mnt/user/podcast/assets/avatar/base_loop.mp4
/mnt/user/podcast/assets/brand/background.png
```

The first avatar job copies the base loop to each worker's local scratch and
builds its ping-pong clip — a one-time cost of a minute or two per node. Replace
the base loop later and you must clear `/mnt/cache/podcast-scratch/pingpong_*`
on every avatar node, or they will keep using the old footage.

### 8 — First run

```bash
podcastpipe cluster && podcastpipe research && podcastpipe script
# review at http://10.10.5.15:8420
podcastpipe finish
```

---

## Adding a node, or a card

The pattern is the same either way: the scheduler dispatches to whichever node of
the right role is free, so capacity is a config fact, not a code change.

**A whole new server:**

1. Install Unraid, the Nvidia driver plugin, and mount node-a's export at
   `/mnt/remotes/podcast` (step 2 above).
2. Copy `docker-compose.node-c.yml` to `docker-compose.node-d.yml` and change the
   `${NODE_C_*}` variable names. Node-b and node-c are already identical apart
   from their UUIDs, which is the point.
3. Add the node's block to `nodes:` in `config/cluster.yaml`.
4. `podcastpipe cluster` — the new worker should appear `up`.

**A second card in an existing node:** add a service to that node's compose file
with its own port and UUID, and a matching block in `cluster.yaml`. Ports are the
convention that keeps this readable — `:8080` voice, `:8081` avatar, `:8082`
media — so a third role on one box gets the port its role already owns.

**Moving a card between machines:** move its UUID value in `.env` to the new
node's variable name, unchanged. UUIDs belong to cards, not slots.

At three or more avatar workers, drop `window_seconds` to 90 in `cluster.yaml`:
smaller windows spread more evenly across more workers, at the cost of slightly
more per-render warm-up. Check the bottleneck first — with voice on one card at
1.42× realtime, more avatar capacity stops helping at two workers.

## Operations

**Restarting a worker** costs one model load (60–90 s) and nothing else. Jobs in
flight fail, get retried on another node if one exists, and the episode
continues.

**A node dies mid-episode.** The client marks it unhealthy after three
consecutive failures and stops sending work. With a second node of that role the
episode completes on the survivor; without one it fails with a clear message
rather than hanging.

**Watch a render live:**

```bash
docker logs -f podcast-avatar
watch -n2 nvidia-smi
curl -s http://10.10.5.16:8081/health | python3 -m json.tool   # avatar-b
curl -s http://10.10.5.15:8080/health | python3 -m json.tool   # tts-a
```

`/health` reports jobs completed, mean job seconds and free VRAM — the fastest
way to tell whether a node is slow or simply idle.

**Storage.** The share grows by roughly 1–2 GB per episode before cleanup;
scratch stays flat once the driving videos exist. `podcastpipe gc --days 14`
prunes the share. Scratch cleans itself, but if you change the base loop, clear
it manually.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `path ... is not under the shared root` | An episode file was written outside the share. Check `work_dir` in `show.yaml` resolves under `/pipeline`. |
| Worker healthy but every job fails on missing files | The NFS mount is missing or read-only inside the container. `docker exec podcast-tts ls /pipeline`. |
| Avatar much slower than 30–45 min | Scratch is on the parity array, not a cache pool. Check the `/scratch` bind mount. |
| `need 8000 MB of VRAM, only 3200 MB free` | Another container holds the card. See step 4. |
| Renders fine, video and audio drift | Base loop fps differs from `avatar.fps`. Check with `ffprobe`. |
| Workers restart-loop on startup | Healthcheck `start_period` too short for the model load, or the load itself is failing. `docker logs` shows which. |

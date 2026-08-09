# Multi-node deployment

Two Unraid servers now, a third later. Every model loads once when its container
starts and stays in VRAM forever — no reload per episode, no config swapping
between stages.

```
┌──── node-a ── 10.10.5.15 ──────────────────────────┐
│  /mnt/user/podcast   ← shared export, NFS           │
│  /mnt/cache/podcast-scratch  ← local NVMe           │
│                                                     │
│  orchestrator   CPU     review UI  :8420            │
│  avatar-worker  3090    MuseTalk   :8081  ← slowest │
│  media-worker   A1000   Whisper + NVENC :8082       │
└─────────────────────────────────────────────────────┘
                       │ 1 GbE
┌──── node-b ── 10.10.5.16 ──────────────────────────┐
│  mounts node-a's export at /pipeline (NFS)          │
│  tts-worker     3090    Chatterbox :8080            │
└─────────────────────────────────────────────────────┘

later:  node-c ── 10.10.5.17 ── avatar-worker #2 :8081
```

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

| | Single box | Two nodes | Three nodes |
|---|---|---|---|
| Model loads per episode | 3 | **0** | **0** |
| Voice | 4–8 min | 4–8 min (overlapped) | overlapped |
| Avatar | 30–45 min | 30–45 min | **15–23 min** |
| Captions + encode | 7–13 min | 7–13 min | 7–13 min |
| **Total after approval** | **45–70 min** | **38–55 min** | **23–37 min** |

The third server should be a **second avatar worker**, not anything else. Avatar
is 30–45 minutes against 4–8 for voice; it is the only stage where another card
meaningfully shortens the episode. Two avatar workers roughly halve it.

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
mkdir -p /mnt/cache/podcast-scratch /mnt/cache/podcast-scratch-media
cp /path/to/repo/podcast/config/*.yaml /mnt/user/podcast/config/
```

Scratch must be on a **cache/NVMe pool, not the array**. It holds the driving
video, which is written and read constantly during a render.

### 2 — Mount the share on node-b

```bash
mkdir -p /mnt/remotes/podcast
mount -t nfs -o rw,hard,intr,rsize=131072,wsize=131072 \
      10.10.5.15:/mnt/user/podcast /mnt/remotes/podcast
```

Verify both directions before going further — this is the single most common
failure and it surfaces as confusing errors much later:

```bash
touch /mnt/remotes/podcast/_probe && ls -l /mnt/user/podcast/_probe   # on node-a
```

Make it survive reboot by adding the mount to `/boot/config/go`, or use the
**Unassigned Devices** plugin, which handles remounts more gracefully.

### 3 — GPU drivers and UUIDs

Install the **Nvidia Driver** plugin from Community Applications on **both**
servers, then reboot. Get the UUIDs:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
```

**Use UUIDs, never indices.** Indices reorder across reboots and after any
hardware change. On node-a that eventually puts the avatar render on the A1000,
where it will run out of memory or crawl. Paste the UUIDs into the
`NVIDIA_VISIBLE_DEVICES` values in both compose files.

### 4 — Check for VRAM already in use

You mentioned other AI/LLM containers on these boxes. Before deploying:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

The pipeline needs roughly **8 GB free on the avatar 3090**, **6 GB on the voice
3090**, and **2.5 GB on the A1000**. If Ollama or similar is holding a 3090, you
have three options, in order of preference:

1. Pin that container to the *other* 3090 and give this pipeline a clean card.
2. Set `OLLAMA_KEEP_ALIVE=0` so it releases VRAM when idle.
3. Put the avatar worker on node-b and the voice worker on node-a — swap the two
   `role` values in `cluster.yaml`; nothing else changes.

Workers check free VRAM at startup and refuse to load with a specific message
rather than failing mid-render. Tune the threshold per worker with
`REQUIRED_VRAM_MB`.

### 5 — Build and start the workers

```bash
# node-a
cd /mnt/user/podcast/repo/podcast/docker
docker compose -f docker-compose.node-a.yml up -d --build

# node-b
docker compose -f docker-compose.node-b.yml up -d --build
```

First build pulls CUDA base images and PyTorch — expect 20–40 minutes and ~25 GB
per node. Model weights mount from `appdata`, so they survive rebuilds.

### 6 — Verify

```bash
podcastpipe cluster
```

```
NODE         ROLE     STATE    GPU                       VRAM FREE
tts-b        tts      up       NVIDIA GeForce RTX 3090     18432 MB
avatar-a     avatar   up       NVIDIA GeForce RTX 3090     14208 MB
media-a      media    up       NVIDIA RTX A1000             5376 MB
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

## Adding the third server

1. Install Unraid, the Nvidia plugin, and mount node-a's export at
   `/mnt/remotes/podcast`.
2. Copy `docker-compose.node-b.yml`, swap the service for `avatar-worker`, point
   it at `Dockerfile.avatar`, and set that card's UUID.
3. Uncomment the `avatar-c` block in `config/cluster.yaml`.
4. `podcastpipe cluster` — it should show two avatar nodes and capacity 2.

Nothing else changes. The scheduler distributes windows to whichever avatar node
is free, so the episode gets shorter with no code or config change beyond that
one block.

With three nodes, consider dropping `window_seconds` to 90 in `cluster.yaml`:
smaller windows spread more evenly across more workers, at the cost of slightly
more per-render warm-up.

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
curl -s http://10.10.5.15:8081/health | python3 -m json.tool
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

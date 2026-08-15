#!/usr/bin/env python3
"""Fetch MuseTalk's model weights, and fail loudly if anything is missing.

MuseTalk ships download_weights.sh, which cannot be used:

  * It calls `huggingface-cli`, renamed to `hf` in huggingface_hub 1.x.
  * It passes `--include "a" "b"`. The CLI reads that as `--include=a` plus a
    stray positional argument, so every pattern after the first is dropped --
    silently. That is how you end up with multi-GB .pth files present and the
    small .json configs missing.
  * It calls `gdown --id`, removed in gdown 5.
  * It sets HF_ENDPOINT to a mirror, which is unnecessary outside China and
    frequently slower.
  * It has no exit-code checks anywhere, and ends with an unconditional
    "All weights have been downloaded successfully!" -- printed even when
    nothing transferred at all.

That last point is the reason this file exists. A downloader that reports
success on failure is worse than no downloader: it cost us two avatar nodes
that reported healthy for days while holding zero bytes of model weights.

Run it inside the avatar container so the weights land on the bind-mounted
host directory:

    docker exec podcast-avatar python3.11 /pipeline/repo/scripts/fetch_musetalk_weights.py

Exit code is 0 only when every required file is present and non-empty.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/MuseTalk/models")

# (repo_id, path within repo, subdirectory under DEST, required)
#
# The subdirectory matters: hf_hub_download with local_dir reproduces the
# repo's own layout beneath it. The MuseTalk entries already carry their
# musetalk/ and musetalkV15/ prefixes, so they take DEST directly; the others
# are bare filenames and need a subdirectory to land in the right place.
HF_FILES = [
    ("TMElyralab/MuseTalk", "musetalk/musetalk.json", "", True),
    ("TMElyralab/MuseTalk", "musetalk/pytorch_model.bin", "", True),
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "", True),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "", True),
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae", True),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "sd-vae", True),
    ("openai/whisper-tiny", "config.json", "whisper", True),
    ("openai/whisper-tiny", "pytorch_model.bin", "whisper", True),
    ("openai/whisper-tiny", "preprocessor_config.json", "whisper", True),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "dwpose", True),
    # Only used for sync evaluation, never on the inference path.
    ("ByteDance/LatentSync", "latentsync_syncnet.pt", "syncnet", False),
]

RESNET_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
BISENET_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"


def mb(path: Path) -> float:
    return path.stat().st_size / 1_000_000


def have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def fetch_hf(failures: list[str]) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  FAIL  huggingface_hub is not installed.")
        print("        Run this inside the avatar container, which has it,")
        print("        or: pip install 'huggingface_hub>=0.19.3,<1.0' gdown")
        failures.append("huggingface_hub not installed")
        return

    for repo, filename, sub, required in HF_FILES:
        target = DEST / sub / filename if sub else DEST / filename
        if have(target):
            print(f"  have  {target.relative_to(DEST)}  {mb(target):.1f} MB")
            continue
        local_dir = DEST / sub if sub else DEST
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(local_dir))
        except Exception as exc:
            label = "FAIL " if required else "skip "
            print(f"  {label} {filename}  ({type(exc).__name__}: {exc})")
            if required:
                failures.append(f"{repo}:{filename}")
            continue
        if have(target):
            print(f"  got   {target.relative_to(DEST)}  {mb(target):.1f} MB")
        elif required:
            print(f"  FAIL  {filename} downloaded but is missing or empty")
            failures.append(f"{repo}:{filename}")


def fetch_face_parse(failures: list[str]) -> None:
    out_dir = DEST / "face-parse-bisent"
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet = out_dir / "resnet18-5c106cde.pth"
    if have(resnet):
        print(f"  have  face-parse-bisent/resnet18-5c106cde.pth  {mb(resnet):.1f} MB")
    else:
        try:
            urllib.request.urlopen(RESNET_URL, timeout=60)  # fail fast on a bad URL
            urllib.request.urlretrieve(RESNET_URL, resnet)
            print(f"  got   face-parse-bisent/resnet18-5c106cde.pth  {mb(resnet):.1f} MB")
        except Exception as exc:
            print(f"  FAIL  resnet18-5c106cde.pth  ({type(exc).__name__}: {exc})")
            failures.append("pytorch:resnet18-5c106cde.pth")

    bisenet = out_dir / "79999_iter.pth"
    if have(bisenet):
        print(f"  have  face-parse-bisent/79999_iter.pth  {mb(bisenet):.1f} MB")
        return
    try:
        import gdown

        # gdown 5 removed --id; the Python API still takes one via id=.
        gdown.download(id=BISENET_GDRIVE_ID, output=str(bisenet), quiet=True)
    except Exception as exc:
        print(f"  FAIL  79999_iter.pth  ({type(exc).__name__}: {exc})")
        failures.append("gdrive:79999_iter.pth")
        return
    if have(bisenet):
        print(f"  got   face-parse-bisent/79999_iter.pth  {mb(bisenet):.1f} MB")
    else:
        # Google Drive serves an HTML interstitial for large files when it
        # feels like it, which lands as a small file rather than an error.
        print("  FAIL  79999_iter.pth is missing or empty (Drive quota page?)")
        failures.append("gdrive:79999_iter.pth")


def main() -> int:
    print(f"MuseTalk weights -> {DEST}\n")
    DEST.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    fetch_hf(failures)
    fetch_face_parse(failures)

    total = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"\ntotal on disk: {total / 1_000_000_000:.2f} GB")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        print("\nThe avatar worker will NOT load. Nothing above was faked.")
        return 1

    print("\nAll required weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

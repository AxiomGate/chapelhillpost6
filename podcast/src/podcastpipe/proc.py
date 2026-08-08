"""Subprocess and ffmpeg helpers.

Every ML tool in this stack (MuseTalk, Chatterbox, VibeVoice, LatentSync) pins
mutually incompatible versions of torch, diffusers and transformers. Installing
them into one environment does not work and will not work. So each lives in its
own virtualenv and the orchestrator drives it with a subprocess, pinning the GPU
through ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class CommandError(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: int, stderr: str):
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"command failed ({returncode}): {shlex.join(self.command)}\n{stderr[-4000:]}"
        )


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


def run(
    command: Sequence[str],
    env_overlay: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: int | None = None,
    check: bool = True,
    log_path: str | Path | None = None,
) -> Result:
    """Run a command, capturing output and optionally teeing it to a log file."""
    env = os.environ.copy()
    if env_overlay:
        env.update(env_overlay)

    completed = subprocess.run(
        list(command),
        env=env,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        capture_output=True,
        text=True,
    )

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {shlex.join(command)}\n\n--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}\n",
            encoding="utf-8",
        )

    if check and completed.returncode != 0:
        raise CommandError(command, completed.returncode, completed.stderr)

    return Result(completed.returncode, completed.stdout, completed.stderr)


def venv_python(venv: str | Path) -> str:
    """Absolute path to a virtualenv's python. Raises if it is not there."""
    path = Path(venv).expanduser() / "bin" / "python"
    if not path.exists():
        raise FileNotFoundError(
            f"no python at {path}. Create the environment first: "
            f"scripts/install_models.sh"
        )
    return str(path)


# ---- ffmpeg -------------------------------------------------------------


def ffprobe_duration(path: str | Path) -> float:
    """Duration of a media file in seconds."""
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    text = result.stdout.strip()
    try:
        return float(text)
    except ValueError as exc:
        raise CommandError(["ffprobe", str(path)], 1, f"unparseable duration {text!r}") from exc


def concat_audio(
    inputs: Sequence[str | Path],
    output: str | Path,
    crossfade_ms: int = 0,
    sample_rate: int = 24000,
) -> Path:
    """Join audio files into one.

    With ``crossfade_ms`` above zero this chains ``acrossfade`` filters, which
    hides the tiny level and room-tone discontinuities between separately
    synthesized TTS chunks. Without it, a plain concat demuxer pass is used,
    which is faster and lossless.
    """
    inputs = [str(p) for p in inputs]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not inputs:
        raise ValueError("concat_audio needs at least one input")

    if len(inputs) == 1:
        run(["ffmpeg", "-y", "-i", inputs[0], "-ar", str(sample_rate), str(output)])
        return output

    run(build_concat_command(inputs, output, crossfade_ms, sample_rate))
    return output


def build_concat_command(
    inputs: Sequence[str],
    output: str | Path,
    crossfade_ms: int,
    sample_rate: int,
) -> list[str]:
    """Build the ffmpeg argv for joining audio. Split out so it can be tested
    without ffmpeg installed."""
    command: list[str] = ["ffmpeg", "-y"]
    for path in inputs:
        command += ["-i", str(path)]

    if crossfade_ms <= 0:
        filtergraph = (
            "".join(f"[{i}:a]" for i in range(len(inputs)))
            + f"concat=n={len(inputs)}:v=0:a=1[out]"
        )
    else:
        duration = crossfade_ms / 1000.0
        parts = []
        previous = "[0:a]"
        for index in range(1, len(inputs)):
            label = "[out]" if index == len(inputs) - 1 else f"[x{index}]"
            parts.append(
                f"{previous}[{index}:a]acrossfade=d={duration}:c1=tri:c2=tri{label}"
            )
            previous = label
        filtergraph = ";".join(parts)

    command += [
        "-filter_complex",
        filtergraph,
        "-map",
        "[out]",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    return command


def loudnorm(
    input_path: str | Path,
    output_path: str | Path,
    target_lufs: float,
    true_peak: float = -1.5,
    loudness_range: float = 7.0,
    codec: str = "pcm_s16le",
    sample_rate: int = 48000,
) -> Path:
    """Single-pass EBU R128 normalization to a target integrated loudness."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(build_loudnorm_command(
        input_path, output_path, target_lufs, true_peak, loudness_range, codec, sample_rate
    ))
    return output_path


def build_loudnorm_command(
    input_path: str | Path,
    output_path: str | Path,
    target_lufs: float,
    true_peak: float,
    loudness_range: float,
    codec: str,
    sample_rate: int,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}",
        "-ar",
        str(sample_rate),
        "-c:a",
        codec,
        str(output_path),
    ]

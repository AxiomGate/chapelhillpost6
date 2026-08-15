"""LLM access with one interface over the Claude API and a local vLLM server.

Both providers take the same messages and return text, so switching between
"pay a few cents for a much better editorial brain" and "run everything on GPU1
with nothing leaving the house" is a one-line config change.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .config import Config, ConfigError

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LlmError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:800]
        raise LlmError(f"{url} returned {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LlmError(f"could not reach {url}: {exc.reason}") from exc


class LlmClient:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.llm.provider

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int = 300,
    ) -> str:
        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = self.config.llm.temperature if temperature is None else temperature
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens, temperature, timeout)
        if self.provider == "local":
            return self._openai_compatible(system, user, max_tokens, temperature, timeout)
        raise ConfigError(f"unknown llm.provider {self.provider!r} (want 'anthropic' or 'local')")

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Complete and parse JSON, tolerating fenced code blocks and preamble."""
        text = self.complete(system, user, **kwargs)
        return parse_json_response(text)

    # ---- providers ------------------------------------------------------

    def _anthropic(self, system: str, user: str, max_tokens: int, temperature: float, timeout: int) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Put it in podcast/.env, or set "
                "llm.provider: local in config/show.yaml to use the on-box model."
            )
        # No temperature. Anthropic removed the sampling parameters -- temperature,
        # top_p, top_k -- on Sonnet 5, Opus 5, Opus 4.7/4.8 and Fable 5; sending
        # any of them is a 400, not a warning. Steer these models by prompting
        # instead. `temperature` stays in the signature and in show.yaml because
        # the local provider below still honours it.
        payload = {
            "model": self.config.llm.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = _post_json(
            ANTHROPIC_URL,
            payload,
            {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            timeout,
        )
        # Check this BEFORE parsing. A response cut off at the token ceiling is
        # still valid text, so the failure surfaces downstream as an unhelpful
        # JSON error pointing at whatever character the truncation landed on --
        # which says nothing about the actual cause.
        if data.get("stop_reason") == "max_tokens":
            used = data.get("usage", {}).get("output_tokens", max_tokens)
            raise LlmError(
                f"{self.config.llm.model} hit the {max_tokens}-token output ceiling "
                f"after {used} tokens; the response is truncated. Raise llm.max_tokens "
                f"in show.yaml, or lower --max-stories to shorten the brief."
            )

        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        text = "".join(parts).strip()
        if not text:
            raise LlmError(f"empty response from {self.config.llm.model}")
        return text

    def _openai_compatible(self, system: str, user: str, max_tokens: int, temperature: float, timeout: int) -> str:
        url = self.config.llm.local_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.llm.local_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {}
        local_key = os.environ.get("LOCAL_LLM_API_KEY", "").strip()
        if local_key:
            headers["authorization"] = f"Bearer {local_key}"
        data = _post_json(url, payload, headers, timeout)
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise LlmError(f"unexpected response shape from {url}: {str(data)[:400]}") from exc
        if not text:
            raise LlmError("empty response from local LLM")
        return text


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> Any:
    """Parse JSON out of a model response.

    Models wrap JSON in fences or add a sentence of preamble often enough that
    handling it here is cheaper than re-prompting. Tries, in order: the raw
    text, the first fenced block, then the widest brace/bracket span.
    """
    text = text.strip()
    candidates = [text]

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    raise LlmError(f"no valid JSON in response. Tried {len(candidates)} candidates: {errors[:2]}")

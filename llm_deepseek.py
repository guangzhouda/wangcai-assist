import json
import os
from typing import Dict, Iterator, List, Optional

import httpx


DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def stream_chat(
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = DEFAULT_BASE_URL,
    timeout_sec: Optional[float] = None,
    trust_env: bool = True,
    include_reasoning: bool = False,
) -> Iterator[str]:
    """Stream assistant tokens from DeepSeek's OpenAI-compatible Chat Completions API.

    Note: "stream" refers to server-sent events (SSE) for OUTPUT deltas.
    The request INPUT is still a single JSON payload.
    """

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("Missing DeepSeek API key. Set env var DEEPSEEK_API_KEY.")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    timeout = None if timeout_sec is None else httpx.Timeout(timeout_sec)
    with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
        with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {})

                # In thinking mode, some providers stream reasoning separately.
                if include_reasoning and "reasoning_content" in delta:
                    chunk = delta.get("reasoning_content") or ""
                    if chunk:
                        yield chunk

                chunk = delta.get("content") or ""
                if chunk:
                    yield chunk


def chat(
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = DEFAULT_BASE_URL,
    timeout_sec: Optional[float] = 60.0,
    trust_env: bool = True,
) -> str:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("Missing DeepSeek API key. Set env var DEEPSEEK_API_KEY.")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    timeout = None if timeout_sec is None else httpx.Timeout(timeout_sec)
    with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        obj = resp.json()
        return (obj["choices"][0]["message"]["content"] or "").strip()


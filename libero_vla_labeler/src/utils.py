"""Shared utilities."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image


def get_api_key(config: dict, provider: str) -> str:
    """Return API key from config or environment variable."""
    key = config.get(provider, {}).get("api_key", "")
    if not key:
        env_map = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
        key = os.environ.get(env_map.get(provider, ""), "")
    if not key:
        raise ValueError(
            f"No API key found for '{provider}'. "
            f"Set it in config.yaml or via the corresponding environment variable."
        )
    return key


def frames_to_pil(frames: np.ndarray) -> list[Image.Image]:
    """Convert (N, H, W, C) uint8 numpy array to list of PIL Images."""
    return [Image.fromarray(f.astype(np.uint8)) for f in frames]


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

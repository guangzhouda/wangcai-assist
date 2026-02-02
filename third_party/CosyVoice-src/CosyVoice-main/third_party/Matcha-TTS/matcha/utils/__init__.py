"""Matcha-TTS utilities.

Upstream Matcha-TTS exports a lot of training helpers (Lightning/Rich/etc.) at
import time. CosyVoice uses Matcha-TTS as an inference dependency; keep imports
lightweight to avoid forcing training-only packages in minimal environments.

If you need training utilities, import the specific submodules directly.
"""

try:
    from matcha.utils.pylogger import get_pylogger  # noqa: F401

    __all__ = ["get_pylogger"]
except Exception:  # pragma: no cover
    __all__ = []

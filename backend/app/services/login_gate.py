"""Cooldown for Xiaohongshu login gates.

Repeatedly rebuilding anonymous device sessions after a 406 makes the platform
flag the whole machine (observed 2026-11-04: 32 rapid attempts turned a
probabilistic 406 into a hard block that also broke publishing). Once a login
call fails with 406 we stop calling that endpoint for a while instead of
retrying harder.
"""
from __future__ import annotations

import time

COOLDOWN_SECONDS = 600
_BLOCKED_UNTIL: dict[str, float] = {}


def remaining(key: str) -> int:
    until = _BLOCKED_UNTIL.get(key, 0.0)
    left = int(until - time.time())
    return max(0, left)


def is_blocked(key: str) -> bool:
    return remaining(key) > 0


def penalize(key: str, seconds: int = COOLDOWN_SECONDS) -> None:
    _BLOCKED_UNTIL[key] = max(_BLOCKED_UNTIL.get(key, 0.0), time.time() + seconds)


def clear(key: str) -> None:
    _BLOCKED_UNTIL.pop(key, None)


def looks_like_gate(error: object) -> bool:
    """406 gate errors are surfaced as messages containing the status code."""
    return "406" in str(error)

"""Generate a thin script delegating to the installed categorical runtime."""

from __future__ import annotations

import json
from typing import Any


def generate_thrml_script(payload: dict[str, Any]) -> str:
    """Serialize complete round-trippable values, without importing THRML."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return f'''#!/usr/bin/env python3
"""Experimental THRML 0.1.4 categorical Gibbs smoothing; no policy control."""

import json

from gnn.render.thrml.runtime import run_generated_script

PAYLOAD = json.loads({encoded!r})

if __name__ == "__main__":
    run_generated_script(PAYLOAD, __file__)
'''

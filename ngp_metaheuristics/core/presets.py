"""
core/presets.py

Locates and loads Instant-NGP's own shipped preset configs (base.json,
and whatever else ships alongside it -- e.g. big.json, small.json, a
tensor-encoding variant, etc, depending on the checkout/version). These
are hand-tuned by the Instant-NGP authors and make good seed points for
population initialization: instead of every algorithm starting purely
from noise, part of generation 0 can start from configs a human already
knows work reasonably well.

Nothing here is Instant-NGP-version-specific beyond "look in
configs/nerf/*.json" -- if your checkout ships different preset names,
they're picked up automatically.
"""

import glob
import json
import os
from typing import Dict


def discover_presets(ngp_dir: str, subdir: str = "configs/nerf") -> Dict[str, dict]:
    """
    Returns {preset_name: config_dict} for every *.json file found under
    <ngp_dir>/<subdir>. preset_name is the filename without extension,
    e.g. "base", "big", "small".

    Silently returns {} if the directory doesn't exist -- callers should
    treat "no presets found" as a normal case (fall back to random init)
    rather than an error.
    """
    search_dir = os.path.join(ngp_dir, subdir)
    if not os.path.isdir(search_dir):
        return {}

    presets = {}
    for path in sorted(glob.glob(os.path.join(search_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                presets[name] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # skip unreadable/malformed files rather than crashing the whole run
    return presets


def list_preset_names(ngp_dir: str, subdir: str = "configs/nerf") -> list:
    return sorted(discover_presets(ngp_dir, subdir).keys())

"""
End-to-end pipeline: LIBERO HDF5 + BDDL -> annotated JSON per demo.

Flow
----
1. Load BDDL file and call LLM to get canonical operators.
2. For each demo in the HDF5 file:
   a. Load actions and (optionally) RGB frames.
   b. Segment demo by gripper state changes.
   c. Call Gemini to annotate each segment.
   d. Write output JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from src.bddl_converter import load_bddl, convert_bddl_to_operators
from src.gripper_segmenter import segment_demo, load_demo_actions, list_demo_keys
from src.gemini_annotator import annotate_segments
from src.utils import ensure_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    hdf5_path: str,
    bddl_path: str,
    config: dict,
    output_dir: str | None = None,
    demo_keys: list[str] | None = None,
) -> dict[str, dict]:
    """
    Process one LIBERO task (one HDF5 file + one BDDL file).

    Parameters
    ----------
    hdf5_path : str
        Path to the LIBERO HDF5 dataset file.
    bddl_path : str
        Path to the corresponding .bddl task file.
    config : dict
        Full configuration dict (loaded from config.yaml).
    output_dir : str | None
        Directory where per-demo JSON files are written.
        Defaults to config["paths"]["output_dir"].
    demo_keys : list[str] | None
        Subset of demo keys to process. None = all demos.

    Returns
    -------
    dict mapping demo_key -> annotation dict
    """
    out_dir = Path(output_dir or config["paths"]["output_dir"])
    ensure_dir(out_dir)

    task_name = Path(bddl_path).stem
    bddl_content = load_bddl(bddl_path)

    if demo_keys is None:
        demo_keys = list_demo_keys(hdf5_path)

    # Filter out demos that already have a completed output file
    pending_keys = [
        k for k in demo_keys
        if not (out_dir / f"{task_name}_{k}.json").exists()
    ]
    skipped = len(demo_keys) - len(pending_keys)
    if skipped:
        print(f"      Skipping {skipped} already-processed demo(s).")
    if not pending_keys:
        print(f"      All demos already processed, skipping task.")
        return {}

    print(f"[1/3] Converting BDDL to canonical operators via LLM …")
    canonical_operators = convert_bddl_to_operators(bddl_content, config)
    print(f"      Got {len(canonical_operators['operators'])} canonical operator(s).")

    print(f"[2/3] Processing {len(pending_keys)} demo(s) …")

    all_results: dict[str, dict] = {}

    for demo_key in tqdm(pending_keys, desc="demos"):
        result = _process_demo(
            hdf5_path=hdf5_path,
            demo_key=demo_key,
            task_name=task_name,
            bddl_content=bddl_content,
            canonical_operators=canonical_operators,
            config=config,
        )
        all_results[demo_key] = result

        # Write per-demo JSON
        out_path = out_dir / f"{task_name}_{demo_key}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[3/3] Done. JSON files written to: {out_dir.resolve()}")
    return all_results


# ---------------------------------------------------------------------------
# Single-demo processing
# ---------------------------------------------------------------------------

def _process_demo(
    hdf5_path: str,
    demo_key: str,
    task_name: str,
    bddl_content: str,
    canonical_operators: dict,
    config: dict,
) -> dict:
    actions = load_demo_actions(hdf5_path, demo_key)
    frames = _load_frames(hdf5_path, demo_key, config)

    # Segment by gripper state
    segments = segment_demo(actions, config["gripper"])

    # Annotate each segment with Gemini
    annotated_segments = annotate_segments(
        segments=segments,
        frames=frames,
        task_name=task_name,
        bddl_content=bddl_content,
        canonical_operators=canonical_operators,
        config=config,
    )

    return {
        "task_name": task_name,
        "demo_id": demo_key,
        "hdf5_path": str(hdf5_path),
        "bddl_path": None,           # filled by caller if needed
        "total_frames": int(len(actions)),
        "canonical_operators": canonical_operators,
        "segments": annotated_segments,
    }


def _load_frames(hdf5_path: str, demo_key: str, config: dict) -> np.ndarray:
    """
    Load RGB frames for a demo.  Returns (T, H, W, C) uint8 array.
    Falls back to a placeholder array of zeros if the camera key is absent.
    """
    camera_key = config.get("camera", "agentview_rgb")
    obs_path = f"data/{demo_key}/obs/{camera_key}"

    with h5py.File(hdf5_path, "r") as f:
        if obs_path in f:
            frames = f[obs_path][:]
        else:
            # Fallback: use action length and create empty placeholder
            actions = f[f"data/{demo_key}/actions"][:]
            T = len(actions)
            frames = np.zeros((T, 84, 84, 3), dtype=np.uint8)
            print(
                f"      Warning: camera '{camera_key}' not found for {demo_key}. "
                f"Using blank frames (annotations will be based on text context only)."
            )

    return frames.astype(np.uint8)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

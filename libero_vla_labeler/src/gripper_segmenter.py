"""
Gripper-state-based demo segmentation.

Reads action data from a LIBERO HDF5 demo and detects gripper open/close
transitions to split the demo into segments.

Segment boundary logic
----------------------
- A segment starts at frame 0 and ends just before the first gripper transition.
- Each subsequent segment starts at the transition frame.
- Transitions that are shorter than `min_segment_frames` are merged into the
  previous segment to avoid noise artefacts.

Output format (one entry per segment):
    {
        "segment_id":        "seg_1",
        "start_frame":       0,
        "end_frame":         44,
        "gripper_state":     "open",        # dominant state of THIS segment
        "transition_out":    "close"        # gripper state at the END of this seg
                                            # (None for the last segment)
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import h5py
import numpy as np


GripperState = Literal["open", "close"]


@dataclass
class Segment:
    segment_id: str
    start_frame: int
    end_frame: int          # inclusive
    gripper_state: GripperState
    transition_out: GripperState | None = None  # state AFTER this segment ends

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "gripper_state": self.gripper_state,
            "transition_out": self.transition_out,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_demo(
    actions: np.ndarray,
    gripper_cfg: dict,
) -> list[Segment]:
    """
    Segment a single demo based on gripper state transitions.

    Parameters
    ----------
    actions : np.ndarray, shape (T, action_dim)
        Full action sequence for one demo.
    gripper_cfg : dict
        Sub-dict from config["gripper"].

    Returns
    -------
    list[Segment]
        Ordered list of segments (at least one).
    """
    gripper_idx = int(gripper_cfg.get("gripper_idx", -1))
    open_thresh = float(gripper_cfg.get("open_threshold", 0.5))
    close_thresh = float(gripper_cfg.get("close_threshold", 0.0))
    min_len = int(gripper_cfg.get("min_segment_frames", 3))

    gripper_vals = actions[:, gripper_idx]
    states = _classify_gripper(gripper_vals, open_thresh, close_thresh)
    transitions = _find_transitions(states, min_len)

    return _build_segments(states, transitions)


def load_demo_actions(hdf5_path: str, demo_key: str) -> np.ndarray:
    """
    Load the action array for a single demo from a LIBERO HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to the .hdf5 dataset file.
    demo_key : str
        Key inside the HDF5 file, e.g. "demo_0".

    Returns
    -------
    np.ndarray of shape (T, action_dim)
    """
    with h5py.File(hdf5_path, "r") as f:
        actions = f[f"data/{demo_key}/actions"][:]
    return actions


def list_demo_keys(hdf5_path: str) -> list[str]:
    """Return sorted list of demo keys in an HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys())
    return keys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_gripper(
    vals: np.ndarray,
    open_thresh: float,
    close_thresh: float,
) -> list[GripperState]:
    """Map continuous gripper values to binary open/close states."""
    states: list[GripperState] = []
    for v in vals:
        if v > open_thresh:
            states.append("open")
        else:
            states.append("close")
    return states


def _find_transitions(states: list[GripperState], min_len: int) -> list[int]:
    """
    Return frame indices where the gripper state changes.
    Transitions belonging to runs shorter than `min_len` are discarded.
    """
    n = len(states)
    # Build runs: (state, start, end_inclusive)
    runs: list[tuple[GripperState, int, int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        runs.append((states[i], i, j - 1))
        i = j

    # Filter short runs (merge into previous)
    filtered: list[tuple[GripperState, int, int]] = []
    for run in runs:
        if len(filtered) > 0 and (run[2] - run[1] + 1) < min_len:
            # extend previous run to absorb this short one
            prev = filtered[-1]
            filtered[-1] = (prev[0], prev[1], run[2])
        else:
            filtered.append(run)

    # Transition indices = start of each run except the first
    transitions = [r[1] for r in filtered[1:]]
    return transitions


def _build_segments(
    states: list[GripperState],
    transitions: list[int],
) -> list[Segment]:
    n = len(states)
    boundaries = [0] + transitions + [n]

    segments: list[Segment] = []
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        end_frame = end - 1  # inclusive
        dominant = _dominant_state(states[start:end])
        seg = Segment(
            segment_id=f"seg_{idx + 1}",
            start_frame=start,
            end_frame=end_frame,
            gripper_state=dominant,
        )
        segments.append(seg)

    # Fill transition_out
    for i in range(len(segments) - 1):
        segments[i].transition_out = segments[i + 1].gripper_state

    return segments


def _dominant_state(states: list[GripperState]) -> GripperState:
    return "close" if states.count("close") >= states.count("open") else "open"

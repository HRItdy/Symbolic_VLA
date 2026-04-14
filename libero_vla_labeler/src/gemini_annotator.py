"""
Gemini-based segment annotator.

For each gripper-segmented portion of a demo, this module:
  1. Samples a few representative frames.
  2. Sends those frames + task context to Gemini Vision.
  3. Receives a symbolic action annotation with preconditions, effects, and
     a natural-language description.

The assignment is *flexible*: Gemini is asked to match the segment to the
most semantically appropriate canonical operator regardless of the operator's
original list order, and to infer implicit sub-actions (e.g., moveto) that
are NOT in the canonical list but are needed to connect the steps.
"""

from __future__ import annotations

import json
import re
import numpy as np

from src.gripper_segmenter import Segment
from src.utils import frames_to_pil, pil_to_bytes, get_api_key


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robot task analysis expert. You receive:
1. A task name and its BDDL description.
2. A list of canonical operators (with preconditions and effects) that \
collectively complete the task.
3. Key frames extracted from ONE specific segment of a robot arm demo, \
along with the segment's gripper state (open or closed).

Your job is to assign EXACTLY ONE symbolic action annotation to this segment.

Guidelines:
- The canonical operators do NOT need to be executed in the listed order; \
  choose the one that best matches the visual evidence.
- You MAY annotate a segment with a sub-action that is NOT in the canonical \
  list (e.g., moveto[object]) if the segment clearly represents a \
  navigation/reaching motion. Such implicit actions should have empty effects.
- Preconditions should reflect what must be true in the world BEFORE this \
  segment begins.
- Effects should reflect what changes in the world state AFTER this segment \
  ends. Use "-predicate[obj]" for deletions.
- If a segment is pure motion (arm moving without interacting with an object) \
  label it as moveto[<target_object_or_location>].
- Operator names must be short generic verbs: Close, Open, Place, PickUp, \
  TurnOn, TurnOff, Push, Pull, moveto and so on. Never invent task-specific names \
  like CloseCabinet or PlaceBowlOnCabinet.
- Object arguments must be minimal human-readable nouns. If only one instance \
  of an object type is visible, use just the type (e.g. 'cabinet', 'bowl'). \
  Only add a brief qualifier when needed to disambiguate (e.g. 'left plate', \
  'top drawer'). Never use simulator identifiers (e.g. wooden_cabinet_1) or \
  region suffixes (e.g. wooden_cabinet_1_top_region).

Return ONLY a valid JSON object (no markdown fences) with keys:
  "operator"       : string — e.g. "Place[bowl, cabinet]"
  "preconditions"  : list of strings
  "effects"        : list of strings  (empty list if none)
  "description"    : one sentence plain-English explanation
"""

_USER_TEMPLATE = """\
Task name: {task_name}

BDDL content:
---
{bddl_content}
---

Canonical operators for this task:
{canonical_ops_json}

Segment info:
  segment_id   : {segment_id}
  start_frame  : {start_frame}
  end_frame    : {end_frame}
  gripper state: {gripper_state}
  transition out (gripper state of NEXT segment): {transition_out}

The attached images are evenly-sampled key frames from this segment \
(chronological order, left to right).

Assign a symbolic action annotation to this segment.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_segments(
    segments: list[Segment],
    frames: np.ndarray,           # (T, H, W, C) uint8 — full demo frames
    task_name: str,
    bddl_content: str,
    canonical_operators: list[dict],
    config: dict,
) -> list[dict]:
    """
    Annotate every segment using Gemini Vision.

    Returns a list of annotation dicts, one per segment, each containing the
    segment metadata + the LLM-assigned symbolic action.
    """
    import google.generativeai as genai

    genai.configure(api_key=get_api_key(config, "gemini"))
    model = genai.GenerativeModel(
        model_name=config["gemini"]["model"],
        system_instruction=_SYSTEM_PROMPT,
    )
    max_frames = int(config["gemini"].get("max_frames_per_segment", 5))
    canonical_ops_json = json.dumps(canonical_operators, indent=2)
    results: list[dict] = []

    for seg in segments:
        seg_frames = _sample_frames(frames, seg.start_frame, seg.end_frame, max_frames)
        annotation = _annotate_one_segment(
            model=model,
            seg=seg,
            seg_frames=seg_frames,
            task_name=task_name,
            bddl_content=bddl_content,
            canonical_ops_json=canonical_ops_json,
        )
        results.append({
            **seg.to_dict(),
            "annotation": annotation,
        })

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_frames(
    frames: np.ndarray,
    start: int,
    end: int,
    n: int,
) -> np.ndarray:
    """Evenly sample up to `n` frames from frames[start:end+1]."""
    length = end - start + 1
    if length <= n:
        return frames[start: end + 1]
    indices = np.linspace(start, end, n, dtype=int)
    return frames[indices]


def _annotate_one_segment(
    model,
    seg: Segment,
    seg_frames: np.ndarray,
    task_name: str,
    bddl_content: str,
    canonical_ops_json: str,
) -> dict:
    """Call Gemini Vision for a single segment and parse the response."""
    import google.generativeai as genai

    pil_images = frames_to_pil(seg_frames)

    prompt_text = _USER_TEMPLATE.format(
        task_name=task_name,
        bddl_content=bddl_content,
        canonical_ops_json=canonical_ops_json,
        segment_id=seg.segment_id,
        start_frame=seg.start_frame,
        end_frame=seg.end_frame,
        gripper_state=seg.gripper_state,
        transition_out=seg.transition_out if seg.transition_out else "N/A (last segment)",
    )

    image_parts = [
        {"mime_type": "image/jpeg", "data": pil_to_bytes(img)}
        for img in pil_images
    ]
    content_parts = [prompt_text] + [
        genai.protos.Part(inline_data=genai.protos.Blob(**p)) for p in image_parts
    ]

    response = model.generate_content(content_parts)

    # Handle blocked responses (block_reason: OTHER, SAFETY, etc.)
    if not response.candidates:
        reason = getattr(response.prompt_feedback, "block_reason", "UNKNOWN")
        print(
            f"      Warning: Gemini blocked segment {seg.segment_id} "
            f"(block_reason: {reason}). Inserting placeholder annotation."
        )
        return {
            "operator": "unknown[]",
            "preconditions": [],
            "effects": [],
            "description": f"Annotation unavailable — response blocked ({reason}).",
        }

    return _parse_annotation(response.text)


def _parse_annotation(text: str) -> dict:
    """Strip markdown and parse JSON annotation from LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON:\n{text}\n\nError: {e}") from e
    required = {"operator", "preconditions", "effects", "description"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Annotation missing keys: {missing}\nGot: {data}")
    return data

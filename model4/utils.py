"""Shared utilities for CLPsych 2026 evaluation scripts."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# The 6 ABCD elements
ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]

# Two valences
VALENCES = ["adaptive-state", "maladaptive-state"]

# ---------------------------------------------------------------------------
# Subelement schema: {valence: {element: {index: name}}}
# Indices are 1-based, per valence per element.
# ---------------------------------------------------------------------------

SUBELEMENT_SCHEMA = {
    "adaptive-state": {
        "A": {
            1: "Calm (laid back)",
            2: "Sad (emotional pain, grieving)",
            3: "Happy (content, joyful, hopeful)",
            4: "Vigor (energy)",
            5: "Justifiably angry (assertive anger)",
            6: "Proud",
            7: "Feeling loved",
        },
        "B-O": {
            1: "Relating behavior",
            2: "Autonomous behavior",
        },
        "B-S": {
            1: "Self-care",
        },
        "C-O": {
            1: "Related",
            2: "Facilitating autonomy",
        },
        "C-S": {
            1: "Self-acceptance",
        },
        "D": {
            1: "Relatedness",
            2: "Autonomy",
            3: "Competence",
        },
    },
    "maladaptive-state": {
        "A": {
            1: "Anxious (fearful, tense)",
            2: "Depressed (despair, hopeless)",
            3: "Mania",
            4: "Apathetic (blunted affect)",
            5: "Angry (aggression, disgust, contempt)",
            6: "Ashamed (guilty)",
            7: "Loneliness",
        },
        "B-O": {
            1: "Fight or flight",
            2: "Overcontrolled",
        },
        "B-S": {
            1: "Self-harm",
        },
        "C-O": {
            1: "Detached or over-attached",
            2: "Blocking autonomy",
        },
        "C-S": {
            1: "Self-criticism",
        },
        "D": {
            1: "Relatedness unmet",
            2: "Autonomy unmet",
            3: "Competence unmet",
        },
    },
}

# Number of subelements per element x valence (for validation & evaluation)
NUM_SUBELEMENTS = {
    valence: {elem: len(subs) for elem, subs in elems.items()}
    for valence, elems in SUBELEMENT_SCHEMA.items()
}

# ---------------------------------------------------------------------------
# Gold data conversion: old global category number -> v2 index
# The old format uses global numbering across both valences per element.
# ---------------------------------------------------------------------------

GOLD_CATEGORY_MAP = {
    "adaptive-state": {
        "A":   {1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7},
        "B-O": {1: 1, 3: 2},
        "B-S": {1: 1},
        "C-O": {1: 1, 3: 2},
        "C-S": {1: 1},
        "D":   {1: 1, 3: 2, 5: 3},
    },
    "maladaptive-state": {
        "A":   {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7},
        "B-O": {2: 1, 4: 2},
        "B-S": {2: 1},
        "C-O": {2: 1, 4: 2},
        "C-S": {2: 1},
        "D":   {2: 1, 4: 2, 6: 3},
    },
}

# Cross-annotations: subelements that appear in the "wrong" valence in gold data.
# Map them to the correct v2 index within that valence context.
# E.g., (2) Anxious appearing in adaptive-state:A -> treat as adaptive index 0 (skip)
# We handle these by also including cross-valence mappings.
GOLD_CATEGORY_MAP_CROSS = {
    "adaptive-state": {
        "A":   {2: None},       # Anxious is maladaptive, skip in adaptive context
        "B-O": {2: None},       # Fight/flight is maladaptive
        "D":   {4: None},       # Autonomy unmet is maladaptive
    },
    "maladaptive-state": {
        "A":   {9: None},       # Justifiable anger is adaptive
        "B-O": {1: None},       # Relating is adaptive
    },
}


def extract_category_number(category_str):
    """Extract the number from a gold category string like '(1) Calm/ laid back'."""
    if not category_str:
        return None
    m = re.search(r"\((\d+)\)", str(category_str))
    return int(m.group(1)) if m else None


def convert_gold_category_to_v2(valence, element, category_str):
    """Convert a gold Category string to a v2 subelement index.

    Returns the v2 index (int) or None if the category is a cross-annotation
    or cannot be mapped.
    """
    old_num = extract_category_number(category_str)
    if old_num is None:
        return None

    # Try primary mapping
    mapping = GOLD_CATEGORY_MAP.get(valence, {}).get(element, {})
    if old_num in mapping:
        return mapping[old_num]

    # Check cross-annotation mapping
    cross = GOLD_CATEGORY_MAP_CROSS.get(valence, {}).get(element, {})
    if old_num in cross:
        return cross[old_num]  # None = skip

    return None


def get_presence(state):
    """Get the Presence value from a state dict, returning None if absent or invalid."""
    pres = state.get("Presence")
    if pres is None or isinstance(pres, dict):
        return None
    try:
        return int(pres)
    except (ValueError, TypeError):
        return None


def post_has_evidence(post, valence):
    """Check whether a gold post has valid evidence for a given valence."""
    evidence = post.get("evidence", {})
    state = evidence.get(valence, {})
    return get_presence(state) is not None


def post_has_any_evidence(post):
    """Check whether a gold post has evidence for at least one valence."""
    return any(post_has_evidence(post, v) for v in VALENCES)


def convert_gold_post_to_v2(post):
    """Convert a gold post's evidence to v2 format (subelement index).

    Returns a dict in the v2 prediction format.
    """
    evidence = post.get("evidence", {})
    result = {
        "timeline_id": post.get("timeline_id", ""),
        "post_id": post["post_id"],
    }

    for valence in VALENCES:
        state = evidence.get(valence, {})
        pres = get_presence(state)
        if pres is None:
            continue

        v2_state = {"Presence": pres}
        for elem in ELEMENTS:
            if elem not in state or "Category" not in state[elem]:
                continue
            idx = convert_gold_category_to_v2(valence, elem, state[elem]["Category"])
            if idx is not None:
                v2_state[elem] = {"subelement": idx}

        result[valence] = v2_state

    return result


def load_gold_data(gold_dir):
    """Load gold timeline JSON files from a directory.

    Returns:
        dict mapping (timeline_id, post_id) -> post dict (original format, with timeline_id added)
    """
    gold = {}
    gold_path = Path(gold_dir)
    for fpath in sorted(gold_path.glob("*.json")):
        with open(fpath) as f:
            timeline = json.load(f)
        tid = timeline["timeline_id"]
        for post in timeline["posts"]:
            post["timeline_id"] = tid
            pid = post["post_id"]
            gold[(tid, pid)] = post
    return gold


def load_predictions(pred_file):
    """Load a prediction JSON file (list of per-post predictions)."""
    with open(pred_file) as f:
        return json.load(f)


def build_pred_lookup(predictions):
    """Build a lookup from (timeline_id, post_id) -> prediction dict."""
    lookup = {}
    for pred in predictions:
        key = (pred["timeline_id"], pred["post_id"])
        lookup[key] = pred
    return lookup


def get_subelement(state, element):
    """Extract the subelement index from a v2 state dict for an element.

    Returns an int or None if element is absent.
    """
    if element not in state:
        return None
    elem_data = state[element]
    if isinstance(elem_data, dict) and "subelement" in elem_data:
        return elem_data["subelement"]
    return None


def get_gold_subelement(state, element, valence):
    """Extract subelement index from GOLD state dict (old format with Category).

    Converts the old Category number to v2 index.
    """
    if element not in state or "Category" not in state[element]:
        return None
    return convert_gold_category_to_v2(valence, element, state[element]["Category"])


def validate_predictions_coverage(gold, pred_lookup, task_name, filter_fn=None):
    """Check that predictions cover all required gold posts.

    Returns list of (timeline_id, post_id) keys that are in both gold and pred.
    """
    missing = []
    matched = []
    for key, post in gold.items():
        if filter_fn and not filter_fn(post):
            continue
        if key not in pred_lookup:
            missing.append(key)
        else:
            matched.append(key)

    if missing:
        print("[%s] WARNING: %d gold posts missing from predictions:" % (task_name, len(missing)))
        for tid, pid in missing[:10]:
            print("  timeline=%s, post=%s" % (tid, pid))
        if len(missing) > 10:
            print("  ... and %d more" % (len(missing) - 10))

    return matched
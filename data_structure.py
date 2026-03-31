
# data_structure.py
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from post_embedder import PostEmbedder


WV_MODEL_PATH  = "./wiki-news-300d-1M.vec"
device = "cuda" if torch.cuda.is_available() else "cpu"
PostEmbedder = PostEmbedder(wv_model_path=WV_MODEL_PATH, device=device)

ABCD_TAXONOMY = {
    "A": {
        "adaptive": {
            1:  "Calm/laid back",
            3:  "Sad, Emotional pain, grieving",
            5:  "Content, happy, joy, hopeful",
            7:  "Vigor/energetic",
            9:  "Justifiable anger/assertive anger, justifiable outrage",
            11: "Proud",
            13: "Feel loved, belong",
        },
        "maladaptive": {
            2:  "Anxious/fearful/tense",
            4:  "Depressed, despair, hopeless",
            6:  "Mania",
            8:  "Apathetic, don't care, blunted",
            10: "Angry (aggression), disgust, contempt",
            12: "Ashamed, guilty",
            14: "Feel lonely",
        },
    },
    "B-O": {
        "adaptive": {
            1: "Relating behavior",
            3: "Autonomous or adaptive control behavior",
        },
        "maladaptive": {
            2: "Fight or flight behavior",
            4: "Over controlled or controlling behavior",
        },
    },
    "B-S": {
        "adaptive": {
            1: "Self care and improvement",
        },
        "maladaptive": {
            2: "Self harm, neglect and avoidance",
        },
    },
    "C-O": {
        "adaptive": {
            1: "Perception of the other as related",
            3: "Perception of the other as facilitating autonomy needs",
        },
        "maladaptive": {
            2: "Perception of the other as detached or over attached",
            4: "Perception of the other as blocking autonomy needs",
        },
    },
    "C-S": {
        "adaptive": {
            1: "Self-acceptance and compassion",
        },
        "maladaptive": {
            2: "Self criticism",
        },
    },
    "D": {
        "adaptive": {
            1: "Relatedness",
            3: "Autonomy and adaptive control",
            5: "Competence, self esteem, self-care",
        },
        "maladaptive": {
            2: "Expectation that relatedness needs will not be met",
            4: "Expectation that autonomy needs will not be met",
            6: "Expectation that competence needs will not be met",
        },
    },
}


# Canonical dimension keys (as they appear in JSON)
DIMENSIONS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]

# Change label constants
NO_CHANGE  = "0"
SWITCH     = "S"
ESCALATION = "E"

# Presence scale
PRESENCE_MIN = 1
PRESENCE_MAX = 5

# Date format in JSON
DATE_FORMAT = "%d-%m-%Y, %H:%M:%S"

@dataclass
class SubElement:
    dimension: str        # "A", "B-O", "B-S", "C-O", "C-S", "D"
    valence:   str        # "adaptive" | "maladaptive"
    number:    int        # e.g. 4 for "(4) Depressed, despair, hopeless"
    label:     str        # e.g. "Depressed, despair, hopeless"
    span:      str        # highlighted_evidence from post text

    @classmethod
    def from_json(cls, dimension: str, value: Dict, valence: str) -> "SubElement":
        cat_raw = value.get("Category", "")
        match = re.match(r"\((\d+)\)\s*(.+)", cat_raw)
        if match:
            number = int(match.group(1))
            label  = match.group(2).strip()
        else:
            number = 0
            label  = cat_raw.strip()

        # Fill label from taxonomy if blank
        tax_labels = ABCD_TAXONOMY.get(dimension, {}).get(valence, {})
        if number in tax_labels and not label:
            label = tax_labels[number]

        return cls(
            dimension=dimension,
            valence=valence,
            number=number,
            label=label,
            span=value.get("highlighted_evidence", "").strip(),
        )

    @property
    def short_tag(self) -> str:
        """e.g. 'A-(4)' - used in prompts and summaries."""
        return f"{self.dimension}-({self.number})"

    @property
    def full_tag(self) -> str:
        """e.g. 'A - (4) Depressed, despair, hopeless'"""
        return f"{self.dimension} - ({self.number}) {self.label}"


@dataclass
class SelfState:
    valence:     str               # "adaptive" | "maladaptive"
    subelements: List[SubElement]  # one per dimension at most (Task 1.1)
    presence:    int               # 1-5 (Task 1.2); 1 = not present

    @property
    def by_dimension(self) -> Dict[str, SubElement]:
        return {se.dimension: se for se in self.subelements}

    @property
    def dimensions_present(self) -> List[str]:
        return [se.dimension for se in self.subelements]

    @property
    def is_present(self) -> bool:
        """A self-state is considered present if presence > 1."""
        return self.presence > 1

    def to_prompt_dict(self) -> Dict:
        """Serialise back to the same JSON evidence format for prompting."""
        d = {}
        for se in self.subelements:
            d[se.dimension] = {
                "Category": f"({se.number}) {se.label}",
                "highlighted_evidence": se.span,
            }
        d["Presence"] = self.presence
        return d

def _parse_self_state(block: Dict, valence: str) -> SelfState:
    subelements = []
    presence = 1  # default: not present

    for key, value in block.items():
        if key == "Presence":
            try:
                presence = max(PRESENCE_MIN, min(PRESENCE_MAX, int(value)))
            except (TypeError, ValueError):
                presence = 1
            continue
        if not isinstance(value, dict):
            continue
        try:
            se = SubElement.from_json(key, value, valence)
            subelements.append(se)
        except Exception:
            pass

    # If no subelements found, presence must be 1 (per task spec)
    if not subelements:
        presence = 1

    return SelfState(valence=valence, subelements=subelements, presence=presence)


@dataclass
class Post:
    post_id:    str
    post_index: int
    text:       str
    timestamp:  datetime

    # Task 2: Change labels (INDEPENDENT - both can be set simultaneously)
    switch_label:     str  # "S" | "0"
    escalation_label: str  # "E" | "0"

    # Task 1.2: Well-being score (GAF-based, 1-10 or None)
    wellbeing: Optional[int]

    # Task 1.1 + 1.2: Gold self-states
    adaptive_state:    SelfState  # valence="adaptive"
    maladaptive_state: SelfState  # valence="maladaptive"

    # Whether this post has any annotation
    is_annotated: bool = False

    # Predictions (filled by pipeline)
    pred_adaptive_state:    Optional[SelfState] = None
    pred_maladaptive_state: Optional[SelfState] = None
    pred_switch_label:     str = "0"
    pred_escalation_label: str = "0"
    temporal_embedding: Optional[np.ndarray] = None
    post_embedding: Optional[np.ndarray] = None

    @property
    def is_switch(self) -> bool:
        return self.switch_label == SWITCH

    @property
    def is_escalation(self) -> bool:
        return self.escalation_label == ESCALATION

    @property
    def has_change(self) -> bool:
        return self.is_switch or self.is_escalation

    @property
    def change_tag(self) -> str:
        """Human-readable tag: 'S', 'E', 'S+E', or '-'."""
        tags = []
        if self.is_switch:     tags.append("S")
        if self.is_escalation: tags.append("E")
        return "+".join(tags) if tags else "-"

    @property
    def adaptive_presence(self) -> int:
        return self.adaptive_state.presence

    @property
    def maladaptive_presence(self) -> int:
        return self.maladaptive_state.presence

    @classmethod
    def from_dict(cls, d: Dict) -> "Post":
        try:
            ts = datetime.strptime(d["date"], DATE_FORMAT)
        except (ValueError, KeyError):
            ts = datetime.min

        switch_label     = SWITCH     if str(d.get("Switch",     "0")).upper() == "S" else "0"
        escalation_label = ESCALATION if str(d.get("Escalation", "0")).upper() == "E" else "0"

        if "well-being" in d.keys():
            wb = d.get("Well-being")
            wellbeing = int(wb) if wb is not None else None
        else:
            wellbeing = None

        if "evidence" in d.keys():
            evidence = d.get("evidence", {})
            adaptive_state    = _parse_self_state(evidence.get("adaptive-state",    {}), "adaptive")
            maladaptive_state = _parse_self_state(evidence.get("maladaptive-state", {}), "maladaptive")
            is_annotated = (
                bool(adaptive_state.subelements)
                or bool(maladaptive_state.subelements)
                or wellbeing is not None
            )
        else:
            adaptive_state    = SelfState("adaptive", [], 1)
            maladaptive_state = SelfState("maladaptive", [], 1)
            is_annotated = False

        post_embedding = PostEmbedder.embed(d.get("post", ""))

        return cls(
            post_id=d.get("post_id", ""),
            post_index=int(d.get("post_index", 0)),
            text=d.get("post", ""),
            timestamp=ts,
            switch_label=switch_label,
            escalation_label=escalation_label,
            wellbeing=wellbeing,
            adaptive_state=adaptive_state,
            maladaptive_state=maladaptive_state,
            is_annotated=is_annotated,
            post_embedding=post_embedding,
        )

@dataclass
class Timeline:
    """A complete, chronologically ordered sequence of posts for one user."""
    timeline_id: str
    posts: List[Post]

    # Stats (computed on init)
    n_posts:      int = 0
    n_annotated:  int = 0
    n_switches:   int = 0
    n_escalations: int = 0

    def __post_init__(self):
        self.posts.sort(key=lambda p: (p.timestamp, p.post_index))
        self.n_posts       = len(self.posts)
        self.n_annotated   = sum(1 for p in self.posts if p.is_annotated)
        self.n_switches    = sum(1 for p in self.posts if p.is_switch)
        self.n_escalations = sum(1 for p in self.posts if p.is_escalation)

    def hours_between(self, idx_a: int, idx_b: int) -> float:
        delta = self.posts[idx_b].timestamp - self.posts[idx_a].timestamp
        return max(0.0, delta.total_seconds() / 3600)

    def get_context(self, post_idx: int, window: int = 5) -> List[Post]:
        """Return up to `window` posts BEFORE post_idx (exclusive)."""
        start = max(0, post_idx - window)
        return self.posts[start:post_idx]

    @classmethod
    def from_dict(cls, d: Dict) -> "Timeline":
        posts = [Post.from_dict(p) for p in d.get("posts", [])]
        return cls(timeline_id=d.get("timeline_id", ""), posts=posts)



def load_timeline_file(path: str) -> Timeline:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Timeline.from_dict(data)


def load_all_timelines(data_dir: str, pattern: str = "*.json") -> List[Timeline]:
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No '{pattern}' files found in: {data_dir}")

    timelines = []
    for path in paths:
        try:
            timelines.append(load_timeline_file(path))
        except Exception as e:
            print(f"[WARNING] Skipping {path}: {e}")

    print(f"\nLoaded {len(timelines)} timelines from: {data_dir}")
    _print_dataset_stats(timelines)
    return timelines



# def _empty_self_state(valence: str) -> SelfState:
#     return SelfState(valence=valence, subelements=[], presence=1)


# def load_test_timelines(test_dir: str, embedder: PostEmbedder) -> list[Timeline]:
#     paths     = sorted(glob.glob(os.path.join(test_dir, "*.json")))
#     timelines = []

#     for path in paths:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         posts = []
#         for d in data.get("posts", []):
#             text = d.get("post", "")
#             emb  = embedder.embed(text)

#             # Parse timestamp
#             try:
#                 ts = datetime.strptime(d["date"], DATE_FORMAT)
#             except Exception:
#                 ts = datetime.min

#             post = Post(
#                 post_id          = d.get("post_id", ""),
#                 post_index       = int(d.get("post_index", 0)),
#                 text             = text,
#                 timestamp        = ts,
#                 switch_label     = "0",
#                 escalation_label = "0",
#                 wellbeing        = None,
#                 adaptive_state   = _empty_self_state("adaptive"),
#                 maladaptive_state= _empty_self_state("maladaptive"),
#                 is_annotated     = False,
#                 post_embedding   = emb,
#             )
#             posts.append(post)

#         timelines.append(Timeline(
#             timeline_id=data.get("timeline_id", ""),
#             posts=posts,
#         ))

#     print(f"Loaded {len(timelines)} test timelines  "
#           f"({sum(len(t.posts) for t in timelines)} posts)")
#     return timelines


def _print_dataset_stats(timelines: List[Timeline]) -> None:
    total_posts  = sum(tl.n_posts for tl in timelines)
    total_ann    = sum(tl.n_annotated for tl in timelines)
    total_sw     = sum(tl.n_switches for tl in timelines)
    total_esc    = sum(tl.n_escalations for tl in timelines)
    both         = sum(1 for tl in timelines
                       for p in tl.posts if p.is_switch and p.is_escalation)
    ada_subs     = sum(len(p.adaptive_state.subelements)
                       for tl in timelines for p in tl.posts)
    mal_subs     = sum(len(p.maladaptive_state.subelements)
                       for tl in timelines for p in tl.posts)

    print(f"  Timelines             : {len(timelines)}")
    print(f"  Total posts           : {total_posts}")
    print(f"  Annotated posts       : {total_ann}")
    print(f"  Switch posts          : {total_sw}")
    print(f"  Escalation posts      : {total_esc}")
    print(f"  Both (S+E) posts      : {both}")
    print(f"  Adaptive subelements  : {ada_subs}")
    print(f"  Maladaptive subelements: {mal_subs}")
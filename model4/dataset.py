
import torch 
import torch.nn as nn
import numpy as np
from data_structure import Post, Timeline
from typing import Dict, List, Tuple
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Label definitions
# ─────────────────────────────────────────────────────────────────────────────

DIMENSIONS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]

# Per-head LOCAL class indices (None = "not present in this dimension")
# These are the targets passed to CrossEntropyLoss for each head.
HEAD_LABEL_MAP: Dict[str, Dict[str, Dict]] = {
    "adaptive": {
        "A":   {1: 0, 3: 1, 5: 2,  7: 3,  9: 4,  11: 5, 13: 6, None: 7},
        "B-O": {1: 0, 3: 1, None: 2},
        "B-S": {1: 0, None: 1},
        "C-O": {1: 0, 3: 1, None: 2},
        "C-S": {1: 0, None: 1},
        "D":   {1: 0, 3: 1, 5: 2,  None: 3},
    },
    "maladaptive": {
        "A":   {2: 0, 4: 1, 6: 2,  8: 3,  10: 4, 12: 5, 14: 6, None: 7},
        "B-O": {2: 0, 4: 1, None: 2},
        "B-S": {2: 0, None: 1},
        "C-O": {2: 0, 4: 1, None: 2},
        "C-S": {2: 0, None: 1},
        "D":   {2: 0, 4: 1, 6: 2,  None: 3},
    },
}

# Number of output classes per head
HEAD_SIZES: Dict[str, Dict[str, int]] = {
    valence: {dim: max(idx_map.values()) + 1 for dim, idx_map in dim_map.items()}
    for valence, dim_map in HEAD_LABEL_MAP.items()
}

# Canonical ordered list of (valence, dim) pairs → 12 heads total
HEAD_ORDER: List[Tuple[str, str]] = [
    (v, d) for v in ["adaptive", "maladaptive"] for d in DIMENSIONS
]

PRESENCE_MIN = 1.0
PRESENCE_MAX = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Label conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _post_to_head_labels(post: Post) -> np.ndarray:
    """
    Returns (12,) int64 array of per-head local class indices.
    """
    labels = np.empty(12, dtype=np.int64)
    for h_idx, (valence, dim) in enumerate(HEAD_ORDER):
        state   = post.adaptive_state if valence == "adaptive" else post.maladaptive_state
        idx_map = HEAD_LABEL_MAP[valence][dim]
        none_idx = idx_map[None]
        se_map  = state.by_dimension   # {dim_str: SubElement}
        if dim in se_map:
            labels[h_idx] = idx_map.get(se_map[dim].number, none_idx)
        else:
            labels[h_idx] = none_idx
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimelineSample:
    timeline_id:       str
    embeddings:        np.ndarray   # (T, D)  float32
    head_labels:       np.ndarray   # (T, 12) int64   
    switch_labels:     np.ndarray   # (T,)    float32 
    escalation_labels: np.ndarray   # (T,)    float32 
    ada_presence:      np.ndarray   # (T,)    float32 
    mal_presence:      np.ndarray   # (T,)    float32 
    is_annotated:      np.ndarray   # (T,)    bool    


class TimelineDataset(Dataset):
    def __init__(self,timelines: List[Timeline],skip_no_embedding: bool = True,) -> None:
        self._samples: List[TimelineSample] = []
        skipped = 0

        for tl in timelines:
            posts = tl.posts
            if not posts:
                skipped += 1
                continue
            if skip_no_embedding and any(p.post_embedding is None for p in posts):
                skipped += 1
                continue

            embeddings        = np.stack([p.post_embedding for p in posts], axis=0).astype(np.float32)
            head_labels       = np.stack([_post_to_head_labels(p) for p in posts], axis=0)
            switch_labels     = np.array([1.0 if p.is_switch     else 0.0 for p in posts], dtype=np.float32)
            escalation_labels = np.array([1.0 if p.is_escalation else 0.0 for p in posts], dtype=np.float32)
            ada_presence      = np.array([float(p.adaptive_state.presence)    for p in posts], dtype=np.float32)
            mal_presence      = np.array([float(p.maladaptive_state.presence) for p in posts], dtype=np.float32)
            is_annotated      = np.array([p.is_annotated for p in posts], dtype=bool)

            self._samples.append(TimelineSample(
                timeline_id       = tl.timeline_id,
                embeddings        = embeddings,
                head_labels       = head_labels,
                switch_labels     = switch_labels,
                escalation_labels = escalation_labels,
                ada_presence      = ada_presence,
                mal_presence      = mal_presence,
                is_annotated      = is_annotated,
            ))

        print(f"[TimelineDataset] {len(self._samples)} timelines loaded  ({skipped} skipped).")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> TimelineSample:
        return self._samples[idx]

    @staticmethod
    def collate(batch: List[TimelineSample]) -> Dict:
        lengths = [s.embeddings.shape[0] for s in batch]
        T_max   = max(lengths)
        D       = batch[0].embeddings.shape[1]
        B       = len(batch)

        emb_pad  = torch.zeros(B, T_max, D,   dtype=torch.float32)
        hl_pad   = torch.zeros(B, T_max, 12,  dtype=torch.long)
        sw_pad   = torch.zeros(B, T_max,      dtype=torch.float32)
        esc_pad  = torch.zeros(B, T_max,      dtype=torch.float32)
        ada_pad  = torch.ones(B,  T_max,      dtype=torch.float32)
        mal_pad  = torch.ones(B,  T_max,      dtype=torch.float32)
        ann_mask = torch.zeros(B, T_max,      dtype=torch.bool)
        pad_mask = torch.zeros(B, T_max,      dtype=torch.bool)

        for i, s in enumerate(batch):
            T = s.embeddings.shape[0]
            emb_pad[i, :T]  = torch.from_numpy(s.embeddings)
            hl_pad[i, :T]   = torch.from_numpy(s.head_labels)
            sw_pad[i, :T]   = torch.from_numpy(s.switch_labels)
            esc_pad[i, :T]  = torch.from_numpy(s.escalation_labels)
            ada_pad[i, :T]  = torch.from_numpy(s.ada_presence)
            mal_pad[i, :T]  = torch.from_numpy(s.mal_presence)
            ann_mask[i, :T] = torch.from_numpy(s.is_annotated)
            pad_mask[i, :T] = True

        return {
            "embeddings":        emb_pad,
            "lengths":           lengths,
            "head_labels":       hl_pad,
            "switch_labels":     sw_pad,
            "escalation_labels": esc_pad,
            "ada_presence":      ada_pad,
            "mal_presence":      mal_pad,
            "ann_mask":          ann_mask,
            "pad_mask":          pad_mask,
        }
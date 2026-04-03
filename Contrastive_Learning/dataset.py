import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from data_structure import Timeline, SelfState


TAXONOMY_TO_INDEX: Dict[str, Dict[str, Dict[int, int]]] = {
    "adaptive": {
        "A":   {1: 0,  3: 1,  5: 2,  7: 3,  9: 4,  11: 5, 13: 6},
        "B-O": {1: 7,  3: 8},
        "B-S": {1: 9},
        "C-O": {1: 10, 3: 11},
        "C-S": {1: 12},
        "D":   {1: 13, 3: 14, 5: 15},
    },
    "maladaptive": {
        "A":   {2: 16, 4: 17, 6: 18, 8: 19, 10: 20, 12: 21, 14: 22},
        "B-O": {2: 23, 4: 24},
        "B-S": {2: 25},
        "C-O": {2: 26, 4: 27},
        "C-S": {2: 28},
        "D":   {2: 29, 4: 30, 6: 31},
    },
}
NUM_LABELS = 32   # 16 adaptive + 16 maladaptive


def self_state_to_vector(adaptive_state: SelfState, maladaptive_state: SelfState) -> np.ndarray:
    """
    Convert a post's two SelfState objects into a 32-dim binary vector.

    Each dimension corresponds to one taxonomy sub-element.
    A '1' means that sub-element was annotated as present in this post.
    Presence score is intentionally NOT encoded here — it is used only
    to weight instances during training (see SelfStateContrastiveDataset).
    """
    vec = np.zeros(NUM_LABELS, dtype=np.float32)

    for valence, state in (("adaptive", adaptive_state), ("maladaptive", maladaptive_state)):
        dim_lookup = TAXONOMY_TO_INDEX[valence]
        for se in state.subelements:
            idx = dim_lookup.get(se.dimension, {}).get(se.number)
            if idx is not None:
                vec[idx] = 1.0

    return vec



@dataclass
class _Sample:
    text:        str
    label_vec:   np.ndarray   # (32,) binary
    presence_w:  float        # average of adaptive + maladaptive presence


class SelfStateContrastiveDataset(Dataset):
    def __init__(self, timelines: List[Timeline], tokenizer, max_length: int = 128, annotated_only: bool = True, min_labels: int = 1):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self._samples: List[_Sample] = []

        skipped = 0
        for tl in timelines:
            for post in tl.posts:
                if annotated_only and not post.is_annotated:
                    skipped += 1
                    continue

                label_vec = self_state_to_vector(post.adaptive_state, post.maladaptive_state)
                if label_vec.sum() < min_labels:
                    skipped += 1
                    continue

                # Presence weight: mean presence score, normalised to [0, 1]
                ada_p = post.adaptive_state.presence    # 1-5
                mal_p = post.maladaptive_state.presence # 1-5
                presence_w = (ada_p + mal_p) / 10.0    # range 0.2 – 1.0

                self._samples.append(
                    _Sample(text=post.text, label_vec=label_vec, presence_w=presence_w)
                )

        print(
            f"[SelfStateContrastiveDataset] {len(self._samples)} samples loaded "
            f"({skipped} skipped)."
        )

    # ── torch Dataset protocol ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, float]:
        s = self._samples[idx]
        return s.text, s.label_vec, s.presence_w

    # ── collate ────────────────────────────────────────────────────────────

    def collate_fn(self, batch: List[Tuple[str, np.ndarray, float]]) -> Dict:
        texts, label_vecs, weights = zip(*batch)

        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],       # (B, L)
            "attention_mask": enc["attention_mask"],   # (B, L)
            "labels":         torch.tensor(np.stack(label_vecs), dtype=torch.float32),  # (B, 32)
            "weights":        torch.tensor(weights,  dtype=torch.float32),              # (B,)
        }

# contrastive_model.py
"""
Contrastive Self-State Encoder
================================
Trains a lightweight DistilBERT model to map post text → embeddings
that reflect the ABCD taxonomy self-state labels.

Architecture
------------
  DistilBERT [CLS] token
       ↓
  Projection head  (768 → embedding_dim, with LayerNorm + GELU)
       ↓
  L2-normalised embedding  (shape: B × embedding_dim)
       ↓
  ┌──────────────────────────────┐
  │  MultiLabelSupConLoss        │  ← pulls same-state posts together
  │  BCEWithLogitsLoss (aux)     │  ← keeps embedding label-aware
  └──────────────────────────────┘

After training, SelfStateEmbedder.embed(text) returns a numpy vector
that you can directly concatenate to PostEmbedder output.

Usage
-----
  # 1. Train
  trainer = SelfStateContrastiveTrainer(model, device=device)
  trainer.fit(train_loader, val_loader, epochs=10)

  # 2. Inference / append to PostEmbedder
  embedder = SelfStateEmbedder("self_state_encoder.pt", device=device)
  extra_emb = embedder.embed(post.text)               # (128,)
  full_emb  = np.concatenate([post.post_embedding, extra_emb])
"""

from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizerFast
from dataset import NUM_LABELS

class SelfStateEncoder(nn.Module):
    """
    DistilBERT backbone + two heads:

    encode()      → L2-normalised (B, embedding_dim) — used for contrastive loss
                    and as the final embedding during inference.

    forward()     → (embedding, logits) where logits (B, 32) feeds the auxiliary
                    BCE multi-label classification loss during training.

    Parameters
    ----------
    embedding_dim : dimensionality of the output space (default 128)
    dropout       : dropout applied inside the projection head
    freeze_n_layers : freeze the first N DistilBERT transformer layers
                      (0 = train all, 6 = freeze all).
                      Useful when labelled data is scarce.
    """

    def __init__(self, embedding_dim:int= 128,dropout:float = 0.1,freeze_n_layers: int = 0) -> None:
        super().__init__()

        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.backbone.config.hidden_size  # 768

        # Optionally freeze lower transformer layers to avoid overfitting
        if freeze_n_layers > 0:
            for layer in self.backbone.transformer.layer[:freeze_n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Projection head: CLS → embedding space
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Auxiliary multi-label head (only used during training)
        self.classifier = nn.Linear(embedding_dim, NUM_LABELS)

    # ── core encode (inference-safe) ──────────────────────────────────────

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings of shape (B, embedding_dim)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        emb = self.projector(cls)               # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=-1)    # unit-sphere

    def forward(self,input_ids:torch.Tensor,attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb    = self.encode(input_ids, attention_mask)  # (B, D)
        logits = self.classifier(emb)                    # (B, 32)
        return emb, logits



class MultiLabelSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multi-label binary targets.

    Unlike the original SupCon paper (which assumes a single class label),
    here the "similarity" between two posts is their Jaccard similarity over
    binary label vectors:

        J(a, b) = |a ∩ b| / |a ∪ b|

    Pair roles
    ──────────
    pos_threshold : J >= this  →  positive pair  (pulled together)
    neg_threshold : J <  this  →  negative pair  (pushed apart)
    Between the two thresholds the pair is IGNORED (margin zone).

    Loss formula (per anchor i):
        L_i = - (1 / |P_i|) · Σ_{j ∈ P_i} log [
                  exp(z_i · z_j / τ)
                ──────────────────────────────────
                Σ_{k ∈ N_i ∪ P_i} exp(z_i · z_k / τ)
              ]

    The denominator only sums over non-margin pairs so that ambiguous
    pairs neither help nor hinder learning.

    Parameters
    ----------
    temperature     : τ controls the sharpness of the distribution (default 0.07)
    pos_threshold   : minimum Jaccard to count as a positive (default 0.3)
    neg_threshold   : maximum Jaccard to count as a negative (default 0.1)
    """

    def __init__(self,temperature:   float = 0.07,pos_threshold: float = 0.3,neg_threshold: float = 0.1) -> None:
        super().__init__()
        self.temperature   = temperature
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    @staticmethod
    def _jaccard(labels: torch.Tensor) -> torch.Tensor:
        """
        Pairwise Jaccard similarity for a batch of binary vectors.
        labels: (B, L)  →  returns (B, B)
        """
        intersection = labels @ labels.T                           # (B, B)
        row_sums     = labels.sum(dim=1, keepdim=True)             # (B, 1)
        union        = row_sums + row_sums.T - intersection        # (B, B)
        return intersection / union.clamp(min=1e-8)

    def forward(self,embeddings: torch.Tensor,labels: torch.Tensor) -> torch.Tensor:

        B      = embeddings.size(0)
        device = embeddings.device

        # Pairwise cosine similarities (embeddings already normalised)
        sim = embeddings @ embeddings.T / self.temperature  # (B, B)

        # Label similarity & masks
        label_sim = self._jaccard(labels)                          # (B, B)
        eye       = torch.eye(B, device=device)

        pos_mask  = (label_sim >= self.pos_threshold).float() * (1 - eye)
        neg_mask  = (label_sim <  self.neg_threshold).float() * (1 - eye)
        valid_mask = pos_mask + neg_mask                           # non-margin pairs

        if pos_mask.sum() == 0:
            # Edge case: no positive pairs in this batch (can happen early on)
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        # Log-softmax over non-margin pairs only
        # Set margin pairs and self to -inf so exp() → 0
        inf_mask = (valid_mask == 0).float() * 1e9
        masked_sim = sim - inf_mask                                # (B, B)

        log_denom  = torch.logsumexp(masked_sim, dim=1, keepdim=True)  # (B, 1)
        log_prob   = sim - log_denom                               # (B, B)

        # Average negative log-prob over each anchor's positives
        n_pos            = pos_mask.sum(dim=1).clamp(min=1)        # (B,)
        loss_per_anchor  = -(log_prob * pos_mask).sum(dim=1) / n_pos  # (B,)

        # Only average over anchors that had at least one positive
        has_pos = (pos_mask.sum(dim=1) > 0)
        return loss_per_anchor[has_pos].mean()


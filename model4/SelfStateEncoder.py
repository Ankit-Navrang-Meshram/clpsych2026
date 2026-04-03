from transformers import DistilBertModel
import torch.nn.functional as F
import torch.nn as nn


class SelfStateEncoder(nn.Module):

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
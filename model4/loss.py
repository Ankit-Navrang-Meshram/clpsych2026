import torch
from typing import Dict, Tuple
from dataset import HEAD_ORDER
import torch.nn.functional as F 



class Stage1Loss(nn.Module):
    def __init__(self,dim_ce_weight: float = 1.0,se_weight:     float = 2.0,pos_weight:    float = 5.0,) -> None:
        super().__init__()
        self.dim_ce_weight = dim_ce_weight
        self.se_weight     = se_weight
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self,logits: Dict[str, torch.Tensor],batch:  Dict,) -> Tuple[torch.Tensor, Dict[str, float]]:
        ann_mask  = batch["ann_mask"]    # (B, T) bool
        head_lbl  = batch["head_labels"] # (B, T, 12)
        device    = ann_mask.device

        total   = torch.zeros(1, device=device)
        metrics: Dict[str, float] = {}

        if not ann_mask.any():
            return total.squeeze(), {"total": 0.0}

        flat_ann = ann_mask.reshape(-1)   # (B*T,)

        # ── 12 CrossEntropy heads ────────────────────────────────────────
        ce_sum = torch.zeros(1, device=device)
        for h_idx, (valence, dim) in enumerate(HEAD_ORDER):
            key   = f"{valence}_{dim}"
            logit = logits[key]               # (B, T, n_cls)
            B, T, C = logit.shape
            lf    = logit.reshape(B * T, C)[flat_ann]      # (N_ann, C)
            lblf  = head_lbl[..., h_idx].reshape(B * T)[flat_ann]  # (N_ann,)
            if lblf.numel() == 0:
                continue
            loss = F.cross_entropy(lf, lblf)
            ce_sum   = ce_sum + loss
            metrics[key] = loss.item()

        ce_mean = ce_sum / max(len(HEAD_ORDER), 1)
        total   = total + self.dim_ce_weight * ce_mean

        # ── Switch BCE ───────────────────────────────────────────────────
        sw_logit = logits["switch"].squeeze(-1)   # (B, T)
        sw_lbl   = batch["switch_labels"]          # (B, T)
        sw_loss  = F.binary_cross_entropy_with_logits(
            sw_logit[ann_mask],
            sw_lbl[ann_mask],
            pos_weight=self.pw.to(device),
        )
        total = total + self.se_weight * sw_loss
        metrics["switch"] = sw_loss.item()

        # ── Escalation BCE ───────────────────────────────────────────────
        esc_logit = logits["escalation"].squeeze(-1)
        esc_lbl   = batch["escalation_labels"]
        esc_loss  = F.binary_cross_entropy_with_logits(
            esc_logit[ann_mask],
            esc_lbl[ann_mask],
            pos_weight=self.pw.to(device),
        )
        total = total + self.se_weight * esc_loss
        metrics["escalation"] = esc_loss.item()

        metrics["ce_mean"]    = ce_mean.item()
        metrics["total"]      = total.item()
        return total.squeeze(), metrics


class Stage2Loss(nn.Module):
    def forward(self,preds: torch.Tensor,batch: Dict,) -> Tuple[torch.Tensor, Dict[str, float]]:
        ann_mask = batch["ann_mask"]   # (B, T)
        device   = preds.device

        if not ann_mask.any():
            z = torch.zeros(1, device=device)
            return z.squeeze(), {"ada_rmse": 0.0, "mal_rmse": 0.0, "total": 0.0}

        ada_pred = preds[ann_mask, 0]            # (N_ann,)
        mal_pred = preds[ann_mask, 1]            # (N_ann,)
        ada_tgt  = batch["ada_presence"][ann_mask]
        mal_tgt  = batch["mal_presence"][ann_mask]

        ada_rmse = torch.sqrt(F.mse_loss(ada_pred, ada_tgt) + 1e-8)
        mal_rmse = torch.sqrt(F.mse_loss(mal_pred, mal_tgt) + 1e-8)
        total    = (ada_rmse + mal_rmse) / 2.0

        return total, {
            "ada_rmse": ada_rmse.item(),
            "mal_rmse": mal_rmse.item(),
            "total":    total.item(),
        }

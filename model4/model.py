import torch
import torch.nn as nn
from typing import Optional, List, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import HEAD_ORDER, HEAD_SIZES

class BiLSTMContextEncoder(nn.Module):
    def __init__(self,input_dim:  int,hidden_dim: int   = 512,num_layers: int   = 2,dropout:    float = 0.3,proj_dim:   Optional[int] = None,) -> None:
        super().__init__()

        if proj_dim is not None and proj_dim != input_dim:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            )
            lstm_in = proj_dim
        else:
            self.input_proj = nn.Identity()
            lstm_in = input_dim

        self.lstm = nn.LSTM(input_size    = lstm_in,hidden_size   = hidden_dim,num_layers    = num_layers,batch_first   = True,bidirectional = True,dropout       = dropout if num_layers > 1 else 0.0,)

        self.out_norm  = nn.LayerNorm(2 * hidden_dim)
        self.drop      = nn.Dropout(dropout)
        self.output_dim = 2 * hidden_dim

    def forward(self,embeddings: torch.Tensor,lengths:    List[int],) -> torch.Tensor:
        x = self.input_proj(embeddings)                                   
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _        = pad_packed_sequence(out_packed, batch_first=True) 
        out = self.out_norm(out)
        out = self.drop(out)
        return out   

class TaxonomyClassifier(nn.Module):
    def __init__(self,context_dim: int,hidden_dim:  int   = 256,dropout:     float = 0.2,) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dim_heads = nn.ModuleDict({
            f"{valence}_{dim}": nn.Linear(hidden_dim, HEAD_SIZES[valence][dim])
            for valence, dim in HEAD_ORDER
        })

        self.switch_head     = nn.Linear(hidden_dim, 1)
        self.escalation_head = nn.Linear(hidden_dim, 1)

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(context)   

        logits: Dict[str, torch.Tensor] = {}
        for valence, dim in HEAD_ORDER:
            key = f"{valence}_{dim}"
            logits[key] = self.dim_heads[key](h)   

        logits["switch"]     = self.switch_head(h)       
        logits["escalation"] = self.escalation_head(h)   

        return logits



class PresenceRegressor(nn.Module):
    def __init__(self,context_dim: int,hidden_dim:  int   = 128,dropout:     float = 0.1,) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.net(context)                       # (B, T, 2)
        return 4.0 * torch.sigmoid(raw) + 1.0


class TimelineModel(nn.Module):
    def __init__(self, input_dim:int, lstm_hidden:int= 512,lstm_layers:int= 2,lstm_dropout:float = 0.3, clf_hidden:int   = 256, proj_dim:int = None,reg_hidden:int   = 128) -> None:
        super().__init__()

        self.encoder = BiLSTMContextEncoder(input_dim  = input_dim,hidden_dim = lstm_hidden,num_layers = lstm_layers,dropout    = lstm_dropout,proj_dim   = proj_dim)
        ctx_dim = self.encoder.output_dim   

        self.classifier1 = TaxonomyClassifier(context_dim = ctx_dim,hidden_dim  = clf_hidden)

        self.classifier2 = PresenceRegressor(context_dim = ctx_dim,hidden_dim  = reg_hidden)

    def get_context(self,embeddings: torch.Tensor,lengths:    List[int]) -> torch.Tensor:
        return self.encoder(embeddings, lengths)

    def forward_stage1(self,embeddings: torch.Tensor,lengths:    List[int]) -> Dict[str, torch.Tensor]:
        ctx = self.encoder(embeddings, lengths)
        return self.classifier1(ctx)

    def forward_stage2(self,embeddings: torch.Tensor,lengths:    List[int]) -> torch.Tensor:
        with torch.no_grad():
            ctx = self.encoder(embeddings, lengths)
        return self.classifier2(ctx)

    def forward(self,embeddings: torch.Tensor,lengths:    List[int],stage:      int = 1,):
        if stage == 1:
            return self.forward_stage1(embeddings, lengths)
        return self.forward_stage2(embeddings, lengths)

    def freeze_for_stage2(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier1.parameters():
            param.requires_grad = False
        for param in self.classifier2.parameters():
            param.requires_grad = True
        print("[TimelineModel] Frozen: encoder + classifier1.  Trainable: classifier2.")

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        print("[TimelineModel] All parameters unfrozen.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

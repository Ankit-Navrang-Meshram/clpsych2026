import random
import torch
from torch.utils.data import random_split
from data_structure import load_all_timelines
from dataset import SelfStateContrastiveDataset
from contrastive_model import SelfStateEncoder
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from contrastive_model import MultiLabelSupConLoss
from transformers import DistilBertTokenizerFast
import torch.nn as nn
import math
from SelfStateEmbedder import SelfStateEmbedder


class SelfStateContrastiveTrainer:


    def __init__(
        self,
        model:              SelfStateEncoder,
        device:             str   = "cpu",
        lr:                 float = 2e-5,
        head_lr_multiplier: float = 10.0,
        contrastive_weight: float = 0.7,
        bce_weight:         float = 0.3,
        temperature:        float = 0.07,
        pos_threshold:      float = 0.3,
        neg_threshold:      float = 0.1,
        warmup_epochs:      int   = 1,
    ) -> None:
        self.model              = model.to(device)
        self.device             = device
        self.contrastive_weight = contrastive_weight
        self.bce_weight         = bce_weight
        self._warmup_epochs     = warmup_epochs

        self._con_loss = MultiLabelSupConLoss(
            temperature   = temperature,
            pos_threshold = pos_threshold,
            neg_threshold = neg_threshold,
        )
        self._bce_loss = nn.BCEWithLogitsLoss(reduction="none")   # reduction handled manually

        # Two parameter groups: low LR for backbone, higher LR for heads
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(),   "lr": lr},
                {"params": model.projector.parameters(),  "lr": lr * head_lr_multiplier},
                {"params": model.classifier.parameters(), "lr": lr * head_lr_multiplier},
            ],
            weight_decay=1e-2,
        )

    # ── internal helpers ───────────────────────────────────────────────────

    def _build_scheduler(self, total_steps: int, epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
        # Calculate the actual percentage of training spent warming up
        pct_start = self._warmup_epochs / epochs if epochs > 0 else 0.1
        
        # PyTorch requires pct_start to be strictly less than 1.0
        pct_start = min(pct_start, 0.99) 
        
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr   = [pg["lr"] for pg in self.optimizer.param_groups],
            total_steps = total_steps,
            pct_start   = pct_start,
            anneal_strategy = "cos",
        )

    def _step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run one forward + backward pass. Returns (total_loss, metrics_dict)."""
        ids   = batch["input_ids"].to(self.device)
        mask  = batch["attention_mask"].to(self.device)
        lbl   = batch["labels"].to(self.device)
        wts   = batch["weights"].to(self.device)  # (B,)

        emb, logits = self.model(ids, mask)

        con_loss = self._con_loss(emb, lbl)

        # Presence-weighted BCE: per-sample loss, then weight and mean
        bce_raw  = self._bce_loss(logits, lbl).mean(dim=1)  # (B,)
        bce_loss = (bce_raw * wts).mean()

        loss = self.contrastive_weight * con_loss + self.bce_weight * bce_loss

        return loss, {"con": con_loss.item(), "bce": bce_loss.item(), "total": loss.item()}

    # ── public API ─────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, scheduler=None) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {"con": 0.0, "bce": 0.0, "total": 0.0}

        for batch in loader:
            loss, metrics = self._step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for k, v in metrics.items():
                totals[k] += v

        n = max(len(loader), 1)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {"con": 0.0, "bce": 0.0, "total": 0.0}

        for batch in loader:
            _, metrics = self._step(batch)
            for k, v in metrics.items():
                totals[k] += v

        n = max(len(loader), 1)
        return {k: v / n for k, v in totals.items()}

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10, save_path: str = "self_state_encoder.pt",) -> None:
        """Full training loop with optional validation and best-model checkpointing."""

        # Pass both total_steps and epochs to the scheduler builder
        total_steps = len(train_loader) * epochs
        scheduler = self._build_scheduler(total_steps, epochs)
        best_val  = math.inf

        for epoch in range(1, epochs + 1):
            tr = self.train_epoch(train_loader, scheduler)
            log = (
                f"Epoch {epoch:03d}/{epochs} | "
                f"train  total={tr['total']:.4f}  con={tr['con']:.4f}  bce={tr['bce']:.4f}"
            )

            if val_loader is not None:
                vl = self.eval_epoch(val_loader)
                log += (
                    f" || val  total={vl['total']:.4f}  con={vl['con']:.4f}  bce={vl['bce']:.4f}"
                )
                monitor = vl["total"]
            else:
                monitor = tr["total"]

            if monitor < best_val:
                best_val = monitor
                self.save(save_path)
                log += "  ✓ saved"

            print(log)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"[Trainer] Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[Trainer] Checkpoint loaded ← {path}")



if __name__ == "__main__":


    # ── Config ────────────────────────────────────────────────────────────
    DATA_DIR      = "../../../data/train/"          # folder containing your JSON timeline files
    SAVE_PATH     = "./model_weights/self_state_encoder.pt"
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_DIM = 128
    BATCH_SIZE    = 16               # keep small to have varied pairs per batch
    EPOCHS        = 50
    SEED          = 42

    random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    timelines = load_all_timelines(DATA_DIR)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    full_dataset = SelfStateContrastiveDataset(
        timelines,
        tokenizer,
        max_length=128,
        annotated_only=True,
        min_labels=1,
    )

    # 80 / 20 train-val split
    n_train = int(0.8 * len(full_dataset))
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=full_dataset.collate_fn,
        drop_last=True,           # ensures every batch can form pairs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=full_dataset.collate_fn,
    )

    # ── Model + trainer ───────────────────────────────────────────────────
    model = SelfStateEncoder(
        embedding_dim   = EMBEDDING_DIM,
        dropout         = 0.1,
        freeze_n_layers = 2,      # freeze bottom 2 layers (helpful with small data)
    )

    trainer = SelfStateContrastiveTrainer(
        model              = model,
        device             = DEVICE,
        lr                 = 2e-5,
        head_lr_multiplier = 10.0,
        contrastive_weight = 0.7,
        bce_weight         = 0.3,
        temperature        = 0.07,
        pos_threshold      = 0.3,   # Jaccard >= 0.3 → positive pair
        neg_threshold      = 0.1,   # Jaccard < 0.1  → negative pair
        warmup_epochs      = 1,
    )

    trainer.fit(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = EPOCHS,
        save_path    = SAVE_PATH,
    )

    # ── Inference demo ────────────────────────────────────────────────────
    embedder = SelfStateEmbedder(SAVE_PATH, embedding_dim=EMBEDDING_DIM, device=DEVICE)

    # Enrich every post in the first timeline
    tl = timelines[0]
    for post in tl.posts[:3]:
        emb = embedder.embed(post.text)
        print(
            f"Post {post.post_id}: "
            f"self-state dim={EMBEDDING_DIM}, "
            f"embedding={emb}"
        )
from clpsych2026.Submission.Task_1.model1.code import train_dataset
import random
from torch.utils.data import random_split
from data_structure import load_all_timelines
from dataset import TimelineDataset
from torch.utils.data import DataLoader
from model import TimelineModel
from loss import Stage1Loss, Stage2Loss



class TimelineModelTrainer:

    def __init__(self,model:TimelineModel, device:str="cpu", lr_stage1:float=1e-3, lr_stage2:float=5e-4, weight_decay:float=1e-4, 
                dim_ce_weight:float=1.0, se_weight:float=2.0, pos_weight:float=5.0,) -> None:
        self.model    = model.to(device)
        self.device   = device
        self.lr_s1    = lr_stage1
        self.lr_s2    = lr_stage2
        self.wd       = weight_decay
        self._loss1   = Stage1Loss(dim_ce_weight, se_weight, pos_weight)
        self._loss2   = Stage2Loss()

    # ── Stage 1 ───────────────────────────────────────────────────────────

    def train_stage1(self,train_loader: DataLoader,val_loader:   Optional[DataLoader] = None,epochs:       int  = 15,save_path:    str  = "timeline_model_s1.pt",) -> None:
        """Train encoder + classifier1 end-to-end."""
        self.model.unfreeze_all()
        print(f"[Stage 1] Trainable params: {self.model.trainable_params():,}")

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr_s1, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr_s1,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self._fit(train_loader, val_loader, epochs, save_path, opt, scheduler, stage=1)

    # ── Stage 2 ───────────────────────────────────────────────────────────

    def train_stage2(self,train_loader: DataLoader,val_loader:   Optional[DataLoader] = None,epochs:       int  = 10,save_path:    str  = "timeline_model_s2.pt",) -> None:
        """Freeze encoder + classifier1; train only PresenceRegressor."""
        self.model.freeze_for_stage2()
        print(f"[Stage 2] Trainable params: {self.model.trainable_params():,}")

        opt = torch.optim.AdamW(
            self.model.classifier2.parameters(), lr=self.lr_s2, weight_decay=self.wd
        )
        self._fit(train_loader, val_loader, epochs, save_path, opt, scheduler=None, stage=2)

    # ── Internal loop ────────────────────────────────────────────────────

    def _fit(self,train_loader: DataLoader,val_loader:   Optional[DataLoader],epochs:       int,save_path:    str,opt:          torch.optim.Optimizer,scheduler,stage:        int,) -> None:
        best_val = math.inf
        for epoch in range(1, epochs + 1):
            tr  = self._epoch(train_loader, opt, scheduler, stage, train=True)
            log = f"[S{stage}] Epoch {epoch:03d}/{epochs} | train {self._fmt(tr)}"

            if val_loader is not None:
                vl  = self._epoch(val_loader, stage=stage, train=False)
                log += f" || val {self._fmt(vl)}"
                monitor = vl["total"]
            else:
                monitor = tr["total"]

            if monitor < best_val:
                best_val = monitor
                torch.save(self.model.state_dict(), save_path)
                log += "  ✓ saved"

            print(log)

    def _epoch(self,loader:    DataLoader,opt        = None,scheduler  = None,stage:     int  = 1,train:     bool = True,) -> Dict[str, float]:
        self.model.train(train)
        totals: Dict[str, float] = {}

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                emb     = batch["embeddings"]
                lengths = batch["lengths"]

                if stage == 1:
                    logits = self.model.forward_stage1(emb, lengths)
                    loss, metrics = self._loss1(logits, batch)
                else:
                    preds  = self.model.forward_stage2(emb, lengths)
                    loss, metrics = self._loss2(preds, batch)

                if train and opt is not None and loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()

                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v

        n = max(len(loader), 1)
        return {k: v / n for k, v in totals.items()}

    @staticmethod
    def _fmt(m: Dict[str, float]) -> str:
        important = ["total", "ce_mean", "switch", "escalation", "ada_rmse", "mal_rmse"]
        parts     = [f"{k}={m[k]:.4f}" for k in important if k in m]
        return "  ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="../../../data/train/")
    parser.add_argument("--val_dir", type=str, default="../../../data/val/")
    parser.add_argument("--test_dir", type=str, default="../../../data/test/")
    parser.add_argument("--seed" , type = int , default=42)
    parser.add_argument("--batch_size", type=int , default=1)
    parser.add_argument("--epochs", type=int , default=10)
    parser.add_argument("--input_dim" , type=int , default=1357)
    parser.add_argument("--lstm_hidden_dim" , type=int , default=512)
    parser.add_argument("--lstm_layers" , type=int , default=2)
    parser.add_argument("--proj_dim" , type=int , default=512)
    parser.add_argument("--device" , type=str , default="cuda")
    parser.add_argument("--lr_stage1" , type=float , default=1e-3)
    parser.add_argument("--lr_stage2" , type=float , default=5e-4)
    parser.add_argument("--lstm_dropout" , type=float , default=0.2)
    parser.add_argument("--reg_hidden_dim" , type=int , default=512)
    parser.add_argument("--clf_hidden_dim" , type=int , default=512)
    args = parser.parse_args()

    train_timelines = load_all_timelines(args.train_dir)
    val_timelines = load_all_timelines(args.val_dir)

    train_dataset = TimelineDataset(train_timelines)
    val_dataset = TimelineDataset(val_timelines)    
    

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TimelineModel(input_dim=args.input_dim, lstm_hidden_dim=args.lstm_hidden_dim, lstm_layers=args.lstm_layers, proj_dim=args.proj_dim,
    clf_hidden=args.clf_hidden_dim,lstm_dropout=args.lstm_dropout, reg_hidden_dim=args.reg_hidden_dim)
    print(f"[Model] Total params: {model.total_params():,}")

    trainer = TimelineModelTrainer(model, device=device)
    trainer.train_stage1(train_dataloader, val_dataloader, epochs=args.epochs, save_path="timeline_model_s1.pt")

    model.load_state_dict(torch.load("timeline_model_s1.pt") , map_location=device)
    trainer.train_stage2(train_dataloader, val_dataloader, epochs=args.epochs, save_path="timeline_model_s2.pt")
    

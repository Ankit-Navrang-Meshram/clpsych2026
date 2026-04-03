from clpsych2026.Submission.Task_1.model4.code import task2_predictions
from clpsych2026.Submission.Task_1.model4.code import task1_predictions
import torch
import numpy as np
import argparse
from data_structure import load_all_timelines
from dataset import TimelineDataset
from torch.utils.data import DataLoader
from model import TimelineModel
from typing import List, Dict


class TimelineInference:

    def __init__(self, model: TimelineModel, device: str = "cpu") -> None:
        self.model  = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_context_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        t = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        ctx = self.model.encoder(t, [embeddings.shape[0]])  # (1, T, 2H)
        return ctx.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict(self, embeddings: np.ndarray, threshold:  float = 0.5,) -> List[Dict]:
        t       = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = [embeddings.shape[0]]

        ctx    = self.model.encoder(t, lengths)       # (1, T, 2H)
        logits = self.model.classifier1(ctx)
        pres   = self.model.classifier2(ctx)          # (1, T, 2)

        T = embeddings.shape[0]
        results = []

        for t_idx in range(T):
            pred: Dict = {"adaptive": {}, "maladaptive": {}}

            for valence, dim in HEAD_ORDER:
                key      = f"{valence}_{dim}"
                logit_t  = logits[key][0, t_idx]             # (n_cls,)
                local_idx = logit_t.argmax().item()
                pred[valence][dim] = _INV_HEAD[valence][dim][local_idx]

            pred["switch"]      = logits["switch"][0, t_idx, 0].sigmoid().item() >= threshold
            pred["escalation"]  = logits["escalation"][0, t_idx, 0].sigmoid().item() >= threshold
            pred["ada_presence"] = float(pres[0, t_idx, 0].item())
            pred["mal_presence"] = float(pres[0, t_idx, 1].item())

            results.append(pred)

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="../../../data/test/")
    parser.add_argument("--seed" , type = int , default=42)
    parser.add_argument("--batch_size", type=int , default=1)
    parser.add_argument("--input_dim" , type=int , default=1357)
    parser.add_argument("--lstm_hidden_dim" , type=int , default=512)
    parser.add_argument("--lstm_layers" , type=int , default=2)
    parser.add_argument("--proj_dim" , type=int , default=None)
    parser.add_argument("--lstm_dropout" , type=float , default=0.2)
    parser.add_argument("--clf_hidden_dim" , type=int , default=512)
    parser.add_argument("--reg_hidden_dim" , type=int , default=512)
    parser.add_argument("--model_path" , type=str , default="timeline_model_s2.pt")
    parser.add_argument("--output_dir" , type=str , default="./results/")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_timelines = load_all_timelines(args.test_dir)
    # test_dataset = TimelineDataset(test_timelines)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = TimelineModel(
        input_dim=args.input_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        proj_dim=args.proj_dim,
        lstm_dropout=args.lstm_dropout,
        clf_hidden_dim=args.clf_hidden_dim,
        reg_hidden_dim=args.reg_hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path) , map_location=device)

    inference = TimelineInference(model, device)
    
    task1_predictions = []
    task2_predictions = []

    print("Generating predictions...")

    for tl in test_timelines:
        # Extract valid posts
        valid_posts = [p for p in tl.posts if p.post_embedding is not None]
        
        emb_matrix = np.stack([p.post_embedding for p in valid_posts])

        preds = inference.predict(emb_matrix)

        
        for post, pred in zip(valid_posts, preds):

            ada_presence = max(1, min(5, round(pred["ada_presence"])))
            mal_presence = max(1, min(5, round(pred["mal_presence"])))
            
            ada_state = {"Presence": ada_presence}
            for dim, subelement in pred["adaptive"].items():
                if subelement is not None:
                    ada_state[dim] = {"subelement": subelement}
                    
            mal_state = {"Presence": mal_presence}
            for dim, subelement in pred["maladaptive"].items():
                if subelement is not None:
                    mal_state[dim] = {"subelement": subelement}

            task1_predictions.append({
                "timeline_id": tl.timeline_id,
                "post_id": post.post_id,
                "adaptive-state": ada_state,
                "maladaptive-state": mal_state
            })

            task2_predictions.append({
                "timeline_id": tl.timeline_id,
                "post_id": post.post_id,
                "Switch": "S" if pred["switch"] else "0",
                "Escalation": "E" if pred["escalation"] else "0"
            })
        
    os.makedirs(args.output_dir, exist_ok=True)

    task1_out_path = os.path.join(args.output_dir, "task1_pred.json")
    with open(task1_out_path, "w", encoding="utf-8") as f:
        json.dump(task1_predictions, f, indent=4)
    print(f"Task 1 predictions saved to: {task1_out_path} ({len(task1_predictions)} posts)")

    task2_out_path = os.path.join(args.output_dir, "task2_pred.json")
    with open(task2_out_path, "w", encoding="utf-8") as f:
        json.dump(task2_predictions, f, indent=4)
    print(f"Task 2 predictions saved to: {task2_out_path} ({len(task2_predictions)} posts)")


    

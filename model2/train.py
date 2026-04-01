# train.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import pickle
from tqdm import tqdm

from data_structure import load_all_timelines
from dataset import PostIndex, TopKSimilarDataset
from model import QwenSelfStatePredictor, decode_predictions, TAXONOMY_TO_INDEX



def vectorize_target(adaptive_state, maladaptive_state):
    subelements_vec = torch.zeros(32, dtype=torch.float32)
    presence_vec = torch.zeros(2, dtype=torch.float32)
    
    if adaptive_state and hasattr(adaptive_state, 'subelements'):
        for se in adaptive_state.subelements:
            idx = TAXONOMY_TO_INDEX["adaptive"].get(se.dimension, {}).get(se.number)
            if idx is not None:
                subelements_vec[idx] = 1.0
                
    if maladaptive_state and hasattr(maladaptive_state, 'subelements'):
        for se in maladaptive_state.subelements:
            idx = TAXONOMY_TO_INDEX["maladaptive"].get(se.dimension, {}).get(se.number)
            if idx is not None:
                subelements_vec[idx] = 1.0

    presence_vec[0] = float(adaptive_state.presence) if adaptive_state else 1.0
    presence_vec[1] = float(maladaptive_state.presence) if maladaptive_state else 1.0
    
    return subelements_vec, presence_vec

def qwen_custom_collate(batch):
    batch_prompts = []
    batch_subelements = []
    batch_presence = []
    raw_posts = []

    for inst in batch:
        raw_posts.append(inst.post)
        lines = []
        # Construct Few-Shot Context

        # --- NEW: Construct Chronological Context Section ---
        lines.append("You are a clinical psychologist assistant. ")
        lines.append("Given a social media post, identify the adaptive and maladaptive and rate the presence of adaptive and maladaptive")
        lines.append("Follow the output format shown in the examples exactly.")
        lines.append("A post may contain only an adaptive self state, only a maladaptive self state, or both. Each post must have atleast one self state")
        lines.append("### Current Post History")
        for j, hist_post in enumerate(inst.context_posts):
            lines.append(f"History {j+1}")
            lines.append(f'Post: "{hist_post.text}"')
            lines.append("Output:")
            
            lines.append("  Adaptive Self-State:")
            if hist_post.adaptive_state.subelements:
                for se in hist_post.adaptive_state.subelements:
                    lines.append(f"    {se.full_tag}")
            else:
                lines.append("    none")
            lines.append(f"  Adaptive Presence: {hist_post.adaptive_state.presence} / 5")
            
            lines.append("  Maladaptive Self-State:")
            if hist_post.maladaptive_state.subelements:
                for se in hist_post.maladaptive_state.subelements:
                    lines.append(f"    {se.full_tag}")
            else:
                lines.append("    none")
            lines.append(f"  Maladaptive Presence: {hist_post.maladaptive_state.presence} / 5\n")
            lines.append("")

        lines.append("### Similar Posts to current Post")
        for rank, (ctx_post, score) in enumerate(zip(inst.similar_posts, inst.scores), 1):
            lines.append(f"### Example {rank}  (similarity: {score:.3f})")
            lines.append(f'Post: "{ctx_post.text}"')
            lines.append("Output:")
            
            lines.append("  Adaptive Self-State:")
            if ctx_post.adaptive_state.subelements:
                for se in ctx_post.adaptive_state.subelements:
                    lines.append(f"    {se.full_tag}")
            else:
                lines.append("    none")
            lines.append(f"  Adaptive Presence: {ctx_post.adaptive_state.presence} / 5")
            
            lines.append("  Maladaptive Self-State:")
            if ctx_post.maladaptive_state.subelements:
                for se in ctx_post.maladaptive_state.subelements:
                    lines.append(f"    {se.full_tag}")
            else:
                lines.append("    none")
            lines.append(f"  Maladaptive Presence: {ctx_post.maladaptive_state.presence} / 5\n")

        # Current Query Post
        lines.append("### Current Post")
        lines.append(f'Post: "{inst.text}"')
        lines.append("Output:")
        
        batch_prompts.append("\n".join(lines))

        # Vectorize Targets
        sub_vec, pres_vec = vectorize_target(inst.post.adaptive_state, inst.post.maladaptive_state)
        batch_subelements.append(sub_vec)
        batch_presence.append(pres_vec)

    return {
        "prompts": batch_prompts,
        "labels_subelements": torch.stack(batch_subelements),
        "labels_presence": torch.stack(batch_presence),
        "raw_posts": raw_posts,
        "timeline_ids": [inst.timeline_id for inst in batch]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    #parser.add_argument("--output_file", type=str, default="./task1_pred.json")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    #parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    #parser.add_argument("--wv_model_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default=f"./saved_qwen_clpsych/")
    args = parser.parse_args()
    args.save_dir = args.save_dir + f"{args.model_name}_k{args.k}_t{args.t}_epoch{args.epochs}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.cache_dir, exist_ok=True)
    train_cache_path = os.path.join(args.cache_dir, f"train_dataset_k{args.k}.pkl")
    eval_cache_path = os.path.join(args.cache_dir, f"eval_dataset_k{args.k}.pkl")

    # --- 1. Load Data (with Caching) ---
    if os.path.exists(train_cache_path) and os.path.exists(eval_cache_path):
        print("Loading previously cached datasets from disk...")
        with open(train_cache_path, "rb") as f:
            train_dataset = pickle.load(f)
        with open(eval_cache_path, "rb") as f:
            eval_dataset = pickle.load(f)
        print("Cached datasets loaded successfully!")
    else:
        print("Loading timelines and building indices (this will take a while but will be cached)...")
        train_timelines = load_all_timelines(args.train_dir)
        eval_timelines = load_all_timelines(args.eval_dir)
        
        train_index = PostIndex(train_timelines, exclude_same_timeline=True)
        
        print("Building Top-K datasets...")
        train_dataset = TopKSimilarDataset(train_timelines, train_index, k=args.k, t=args.t, annotated_only=True)
        eval_dataset = TopKSimilarDataset(eval_timelines, train_index, k=args.k, t=args.t, annotated_only=True)
        
        print(f"Saving datasets to cache directory '{args.cache_dir}' for faster future runs...")
        with open(train_cache_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(eval_cache_path, "wb") as f:
            pickle.dump(eval_dataset, f)
        print("Cache saved!")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=qwen_custom_collate)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=qwen_custom_collate)

    # --- 2. Load Tokenizer & Model ---
    print("Initializing Qwen Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token 

    model = QwenSelfStatePredictor.from_pretrained(args.model_name , quantization_config =BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier"]
    ) , device_map="auto")

    # Setup LoRA (PEFT)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["classifier"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Define loss functions for the training loop
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    # --- 3. Training Loop ---
    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            inputs = tokenizer(batch["prompts"], padding=True, truncation=True, max_length=1536, return_tensors="pt").to(device)
            labels_subelements = batch["labels_subelements"].to(device)
            labels_presence = batch["labels_presence"].to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                subelement_logits = outputs["subelement_logits"]
                presence_preds = outputs["presence_preds"]
                
                loss_subelements = bce_loss_fn(subelement_logits.float(), labels_subelements)
                loss_presence = mse_loss_fn(presence_preds.float(), labels_presence)
                
                # Weighting factor
                loss = loss_subelements + (0.5 * loss_presence)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")

    # --- 4. Save the Model ---
    print(f"Saving trained model and tokenizer to '{args.save_dir}'...")
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Save complete.")

    # --- 5. Evaluation & JSON Generation ---
    print("Starting Inference and JSON generation...")
    model.eval()
    submission_results = []

    with torch.no_grad():
        for batch in eval_loader:
            inputs = tokenizer(batch["prompts"], padding=True, truncation=True, max_length=1536, return_tensors="pt").to(device)
            
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            
            sub_logits = outputs["subelement_logits"].cpu()
            pres_preds = outputs["presence_preds"].cpu()
            
            for i in range(len(batch["raw_posts"])):
                post = batch["raw_posts"][i]
                timeline_id = batch["timeline_ids"][i]
                
                decoded_states = decode_predictions(sub_logits[i], pres_preds[i], threshold=0.5)
                
                pred_obj = {
                    "timeline_id": timeline_id,
                    "post_id": post.post_id,
                    "adaptive-state": decoded_states["adaptive-state"],
                    "maladaptive-state": decoded_states["maladaptive-state"]
                }
                
                # if len(pred_obj["adaptive-state"]) == 1 and pred_obj["adaptive-state"]["Presence"] == 1:
                #     del pred_obj["adaptive-state"]
                # if len(pred_obj["maladaptive-state"]) == 1 and pred_obj["maladaptive-state"]["Presence"] == 1:
                #     del pred_obj["maladaptive-state"]
                    
                submission_results.append(pred_obj)

    # Save to JSON
    output_file = f"./eval_result/task1_pred_{args.model_name}_k{args.k}_t{args.t}_epoch{args.epochs}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(submission_results, f, indent=4)
        
    print(f"Evaluation complete! Saved {len(submission_results)} predictions to '{output_file}'.")



"""
python train.py --model_name Qwen/Qwen2.5-7B --train_dir ../../data/train/ --eval_dir ../../data/val/ --cache_dir ./dataset_cache --batch_size 1 --k 5 --epochs 5
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig
import pickle
from tqdm import tqdm
import re
from data_structure import load_all_timelines, Post , DIMENSIONS
from dataset import PostIndex, TopKSimilarDataset,TopKSimilarInstance
from model import QwenPredictor
from dataclasses import dataclass
import argparse
from typing import Dict, Optional



SYSTEM_PROMPT = ("You are a clinical psychologist assistant. "
        "Given a social media post, identify the adaptive and maladaptive and rate the presence of adaptive and maladaptive"
        "Follow the output format shown in the examples exactly."
        "A post may contain only an adaptive self state, only a maladaptive self state, or both. Each post must have atleast one self state"
)

def qwen_custom_collate(batch):
    batch_prompts = []
    raw_posts = []

    for inst in batch:
        raw_posts.append(inst.post)
        lines = []
        # Construct Few-Shot Context

        # --- NEW: Construct Chronological Context Section ---
        
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
        # lines.append("  Adaptive Self-State:")
        # lines.append("    <fill>")
        # lines.append("  Maladaptive Self-State:")
        # lines.append("    <fill>")
        batch_prompts.append("\n".join(lines))

    return {
        "prompts": batch_prompts,
        "raw_posts": raw_posts,
        "timeline_ids": [inst.timeline_id for inst in batch]
    }

def build_chat(tokenizer , system_prompt : str , user_prompt : str) -> str :
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

# Matches: "A - (4) Depressed..."  "B-S-(2) Self harm..."  "C-O - (1) ..."
_SUB_RE  = re.compile(r"([A-Z](?:-[A-Z])?)\s*[-]\s*\((\d+)\)\s*(.+)")
_PRES_RE = re.compile(r"(\d)\s*/\s*5")

@dataclass
class parsedOutput:
    adaptive_state : Dict[str , Optional[Dict[str,int]] | int]
    maladaptive_state : Dict[str , Optional[Dict[str,int]] | int]


def parse_output(text: str)-> parsedOutput:
    adaptive_subs:    Dict[str, int] = {}
    maladaptive_subs: Dict[str, int] = {}
    adaptive_state : Dict[str , Optional[Dict[str,int]] | int] = {}
    maladaptive_state : Dict[str , Optional[Dict[str,int]] | int] = {}
    ada_val = mal_val = None
    current_valence = None
    for line in text.splitlines():
        stripped = line.strip()
        low      = stripped.lower()

        if "adaptive self-state" in low and "maladaptive" not in low:
            current_valence = "adaptive"
            continue
        if "maladaptive self-state" in low:
            current_valence = "maladaptive"
            continue
        if current_valence is None or stripped in ("none", "<fill>", ""):
            continue

        m = _SUB_RE.match(stripped)
        if m:
            dim = m.group(1).upper()
            num = int(m.group(2))
            if dim in DIMENSIONS:                   # ignore hallucinated dims
                if current_valence == "adaptive":
                    adaptive_subs[dim] = num
                else:
                    maladaptive_subs[dim] = num


        m   = _PRES_RE.search(stripped)
        if not m:
            continue
        val = max(1, min(5, int(m.group(1))))
        if current_valence == "adaptive":
            ada_val = val
        else:
            mal_val = val

    adaptive_state["Presence"] = ada_val
    for dim , num in adaptive_subs.items():
        adaptive_state[dim] = {"subelement" : num}
    maladaptive_state["Presence"] = mal_val
    for dim , num in maladaptive_subs.items():
        maladaptive_state[dim] = {"subelement" : num}

    return parsedOutput(adaptive_state , maladaptive_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_dir", type=str, default="../../../data/train/")
    parser.add_argument("--test_dir", type=str, default="../../../data/test/")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./saved_qwen_clpsych/")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    args = parser.parse_args()


    # # 1. Replace Argument Parser with a Dictionary
    # args = {
    #     "model_name": "Qwen/Qwen2.5-7B-Instruct",
    #     "dataset_dir": "../../../data/train/",
    #     "cache_dir": "./dataset_cache",
    #     "batch_size": 1,
    #     "threshold": 0.5,
    #     "k": 5,
    #     "t": 2, # Your new temporal context parameter
    #     "save_dir": "./saved_qwen_clpsych/",
    #     "split" : "train",
    #     "max_new_tokens":256,
    #     "load_in_8bit" : False
    # }

    # Update save directory logic using dictionary keys
    args.save_dir = os.path.join(args.save_dir, f"{args.model_name}_k{args.k}_t{args.t}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Data (with Caching) ---
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{args.split}_dataset_k{args.k}_t{args.t}.pkl")


    if args.split == "train":
        if os.path.exists(cache_path):
            print("Loading previously cached datasets from disk...")
            with open(cache_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            print("Loading timelines and building indices...")
            timelines = load_all_timelines(args.train_dir)
            index = PostIndex(timelines, exclude_same_timeline=True)
            print("Building datasets...")
            dataset = TopKSimilarDataset(timelines, index, k=args.k, t=args.t, annotated_only=False)
            with open(cache_path, "wb") as f:
                pickle.dump(dataset, f)
            print("Cache saved!")
    else:
        if os.path.exists(cache_path):
            print("Loading previously cached datasets from disk...")
            with open(cache_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            print("Loading timelines and building indices...")
            train_timelines = load_all_timelines(args.train_dir)
            test_timelines = load_all_timelines(args.test_dir)
            index = PostIndex(train_timelines, exclude_same_timeline=True)
            print("Building datasets...")
            dataset = TopKSimilarDataset(test_timelines, index, k=args.k, t=args.t, annotated_only=False)
            with open(cache_path, "wb") as f:
                pickle.dump(dataset, f)
            print("Cache saved!")

        

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=qwen_custom_collate)

    # --- 2. Load Tokenizer & Model ---
    print("Initializing Qwen Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 

    wrapper = QwenPredictor( model_name=args.model_name,max_new_tokens=args.max_new_tokens,load_in_8bit=args.load_in_8bit)
    model = wrapper.model
    # --- 3. Prediction Loop ---
    print("Starting Prediction...")

    predictions = []

    for step, batch in enumerate(loader):
        # Using updated max_length 2048 for temporal context
        print(f"Step : {step}")
        chats = [build_chat(tokenizer , SYSTEM_PROMPT ,p ) for p in  batch["prompts"]]
        enc = tokenizer(chats, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(wrapper.device)  
        with torch.no_grad():
            out_ids = model.generate(**enc,max_new_tokens=wrapper.max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            # out_ids = model.generate(**enc,
            #     max_new_tokens=256,
            #     do_sample=False,          # Greedy decoding (temperature=0) for factual extraction
            #     repetition_penalty=1.1,   # Prevents infinite loops
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id
            # )
        input_len = enc["input_ids"].shape[1]     
        raw_outputs = [tokenizer.decode(ids[input_len:], skip_special_tokens=True) for ids in out_ids]    
        for inst , tid , raw in zip (batch['raw_posts'] , batch['timeline_ids'] , raw_outputs):
            pred = parse_output(raw)
            predictions.append({
                "timeline_id":      tid,
                "post_id":          inst.post_id,
                #"text":             inst.text,
                #"raw_output":       raw,
                "adaptive-state":    pred.adaptive_state,
                "maladaptive-state": pred.maladaptive_state,
            })


    # Save to JSON
    output_file = f"./pred_result/task1_pred_{args.model_name}_{args.split}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
        
    print(f"Evaluation complete! Saved {len(predictions)} predictions to '{output_file}'.")


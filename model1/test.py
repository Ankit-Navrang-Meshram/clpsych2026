# test.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType , PeftModel
import pickle
from tqdm import tqdm

from data_structure import load_all_timelines
from dataset import PostIndex, TopKSimilarDataset
from model import QwenSelfStatePredictor, decode_predictions, TAXONOMY_TO_INDEX
from post_embedder import PostEmbedder
# from  data_structure import load_test_timelines
def test_collate(batch):
    """
    batch: List[TopKSimilarInstance]
    Builds prompts identical to training collate but skips target vectorisation.
    """
    prompts      = []
    raw_posts    = []
    timeline_ids = []

    for inst in batch:
        raw_posts.append(inst.post)
        timeline_ids.append(inst.timeline_id)

        lines = []
        lines.append("You are a clinical psychologist assistant. ")
        lines.append("Given a social media post, identify the adaptive and maladaptive and rate the presence of adaptive and maladaptive")
        lines.append("Follow the output format shown in the examples exactly.")
        lines.append("A post may contain only an adaptive self state, only a maladaptive self state, or both. Each post must have atleast one self state")
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

        lines.append("### Current Post")
        lines.append(f'Post: "{inst.text}"')
        lines.append("Output:")
        prompts.append("\n".join(lines))

    return {"prompts": prompts, "raw_posts": raw_posts, "timeline_ids": timeline_ids}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    #parser.add_argument("--output_file", type=str, default="./task1_pred.json")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--wv_model_path", type=str, required=True)
    parser.add_argument("--model_weights", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.cache_dir, exist_ok=True)
    test_cache_path = os.path.join(args.cache_dir, f"test_dataset_k{args.k}.pkl")

    if os.path.exists(test_cache_path):
        print("\nLoading test dataset from cached files ...")
        with open(test_cache_path, "rb") as f:
            test_dataset = pickle.load(f)    
    else:
        train_timelines = load_all_timelines(args.train_dir)
        test_timelines = load_all_timelines(args.test_dir)
        train_index     = PostIndex(train_timelines, exclude_same_timeline=True)
        test_dataset   = TopKSimilarDataset( test_timelines, train_index, k=args.k, annotated_only=False)
        with open(test_cache_path, "wb") as f:
            pickle.dump(test_dataset, f)
        print(f"  Test dataset cached -> {test_cache_path}")

    test_loader = DataLoader( test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collate)
    print(f"  Test instances: {len(test_dataset)}")


    print(f"\nLoading model from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for decoder-only inference

    # Load base model with 4-bit quant (same config as training)
    base_model = QwenSelfStatePredictor.from_pretrained(
        args.model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["classifier"],
        ),
        device_map="auto",
    )
    # 2. Load the LoRA Adapters and the Classifier
    # args.model_weights should point to the folder containing adapter_model.bin
    print(f"Loading LoRA Adapters from: {args.model_weights}")
    model = PeftModel.from_pretrained(base_model, args.model_weights)
    model.eval()

    print("\nRunning inference ...")
    submission = []
    total      = len(test_loader)

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            inputs = tokenizer(
                batch["prompts"],
                padding=True,
                truncation=True,
                max_length=1536,
                return_tensors="pt",
            ).to(device)

            outputs    = model(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"])
            sub_logits = outputs["subelement_logits"].cpu()
            pres_preds = outputs["presence_preds"].cpu()

            for i, (post, tid) in enumerate(zip(batch["raw_posts"], batch["timeline_ids"])):
                decoded = decode_predictions(sub_logits[i], pres_preds[i], args.threshold)

                pred_obj = {
                    "timeline_id":       tid,
                    "post_id":           post.post_id,
                    "adaptive-state":    decoded["adaptive-state"],
                    "maladaptive-state": decoded["maladaptive-state"],
                }

                # Omit state entirely if presence=1 and no subelements predicted
                # (mirrors the format in the example pred.json)
                # if (len(pred_obj["adaptive-state"]) == 1
                #         and pred_obj["adaptive-state"]["Presence"] == 1):
                #     del pred_obj["adaptive-state"]

                # if (len(pred_obj["maladaptive-state"]) == 1
                #         and pred_obj["maladaptive-state"]["Presence"] == 1):
                #     del pred_obj["maladaptive-state"]

                submission.append(pred_obj)

            if (step + 1) % 10 == 0 or (step + 1) == total:
                print(f"  {step + 1}/{total} batches done", end="\r")

    print()

    # ── Write output ──────────────────────────────────────────────────────────
    output_file = f"./test_result/task1_pred_{args.model_name}.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=4, ensure_ascii=False)

    # Stats
    has_ada = sum(1 for r in submission if "adaptive-state"    in r)
    has_mal = sum(1 for r in submission if "maladaptive-state" in r)
    has_both= sum(1 for r in submission if "adaptive-state" in r and "maladaptive-state" in r)
    print(f"\nWrote {len(submission)} records -> {output_file}")
    print(f"  With adaptive-state    : {has_ada}")
    print(f"  With maladaptive-state : {has_mal}")
    print(f"  With both states       : {has_both}")
    print(f"  With neither state     : {len(submission) - has_ada - has_mal + has_both}")



"""
python test.py --model_name Qwen/Qwen2.5-7B --train_dir ../../data/train/ --test_dir ../../data/test/ --cache_dir ./dataset_cache --wv_model_path ./wiki-news-300d-1M.vec --model_weights ./saved_qwen_clpsych/Qwen/Qwen2.5-7B_epoch5
"""
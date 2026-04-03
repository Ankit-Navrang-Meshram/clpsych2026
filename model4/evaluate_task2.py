"""Evaluation for Task 2 — Moments of Change (Switch & Escalation)."""

import argparse
import json
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from utils import (
    build_pred_lookup,
    load_gold_data,
    load_predictions,
    validate_predictions_coverage,
)

LABELS = ["Switch", "Escalation"]


def parse_label(value, positive):
    """Parse a Switch/Escalation label string to binary."""
    return int(positive in str(value))


def evaluate_post_level(gold, pred_lookup, matched):
    """Post-level evaluation: pool all posts together."""
    positive_chars = {"Switch": "S", "Escalation": "E"}
    results = {}

    for label_name in LABELS:
        pos_char = positive_chars[label_name]
        g_list = []
        p_list = []

        for key in matched:
            g_val = parse_label(gold[key].get(label_name, "0"), pos_char)
            p_val = parse_label(pred_lookup[key].get(label_name, "0"), pos_char)
            g_list.append(g_val)
            p_list.append(p_val)

        g = np.array(g_list)
        p = np.array(p_list)

        results[label_name] = {
            "precision": precision_score(g, p),
            "recall": recall_score(g, p),
            "f1": f1_score(g, p),
            "support_positive": int(g.sum()),
            "support_total": len(g),
        }

    results["macro_f1"] = np.mean([results[l]["f1"] for l in LABELS])
    return results


def evaluate_timeline_level(gold, pred_lookup, matched):
    """Timeline-level evaluation: compute P/R/F1 per timeline, macro-average."""
    positive_chars = {"Switch": "S", "Escalation": "E"}

    # Group matched keys by timeline
    timeline_posts = defaultdict(list)
    for key in matched:
        tid = key[0]
        timeline_posts[tid].append(key)

    results = {}

    for label_name in LABELS:
        pos_char = positive_chars[label_name]
        tl_precisions = []
        tl_recalls = []
        tl_f1s = []

        for tid, keys in sorted(timeline_posts.items()):
            g_list = []
            p_list = []
            for key in keys:
                g_val = parse_label(gold[key].get(label_name, "0"), pos_char)
                p_val = parse_label(pred_lookup[key].get(label_name, "0"), pos_char)
                g_list.append(g_val)
                p_list.append(p_val)

            g = np.array(g_list)
            p = np.array(p_list)

            # If no positives in gold and no positives predicted, count as perfect
            if g.sum() == 0 and p.sum() == 0:
                tl_precisions.append(1.0)
                tl_recalls.append(1.0)
                tl_f1s.append(1.0)
            else:
                tl_precisions.append(precision_score(g, p))
                tl_recalls.append(recall_score(g, p))
                tl_f1s.append(f1_score(g, p))

        results[label_name] = {
            "precision": np.mean(tl_precisions),
            "recall": np.mean(tl_recalls),
            "f1": np.mean(tl_f1s),
            "num_timelines": len(timeline_posts),
        }

    results["macro_f1"] = np.mean([results[l]["f1"] for l in LABELS])
    return results


def evaluate(gold_dir, pred_file):
    """Run Task 2 evaluation. Returns results dict."""
    gold = load_gold_data(gold_dir)
    predictions = load_predictions(pred_file)
    pred_lookup = build_pred_lookup(predictions)

    matched = validate_predictions_coverage(gold, pred_lookup, "Task 2")

    post_results = evaluate_post_level(gold, pred_lookup, matched)
    tl_results = evaluate_timeline_level(gold, pred_lookup, matched)

    combined_macro_f1 = (post_results["macro_f1"] + tl_results["macro_f1"]) / 2.0

    results = {
        "post_level": post_results,
        "timeline_level": tl_results,
        "combined_macro_f1": combined_macro_f1,
    }

    print_results(results)
    return results


def print_results(results):
    print("=" * 70)
    print("TASK 2 — Moments of Change (Switch & Escalation)")
    print("=" * 70)

    print("\n--- Post-Level ---")
    print("%-15s %7s %7s %7s %6s %6s" % ("Label", "Prec", "Rec", "F1", "Pos", "Total"))
    print("-" * 50)
    for label_name in LABELS:
        m = results["post_level"][label_name]
        print("%-15s %7.3f %7.3f %7.3f %6d %6d" % (
            label_name, m["precision"], m["recall"], m["f1"],
            m["support_positive"], m["support_total"]))
    print("-" * 50)
    print("%-15s %7s %7s %7.3f" % ("Macro F1", "", "", results["post_level"]["macro_f1"]))

    print("\n--- Timeline-Level (macro-averaged over %d timelines) ---" % (
        results["timeline_level"][LABELS[0]]["num_timelines"]))
    print("%-15s %7s %7s %7s" % ("Label", "Prec", "Rec", "F1"))
    print("-" * 40)
    for label_name in LABELS:
        m = results["timeline_level"][label_name]
        print("%-15s %7.3f %7.3f %7.3f" % (label_name, m["precision"], m["recall"], m["f1"]))
    print("-" * 40)
    print("%-15s %7s %7s %7.3f" % ("Macro F1", "", "", results["timeline_level"]["macro_f1"]))

    print("\n--- Combined (for ranking) ---")
    print("%-15s %7.3f" % ("Combined F1", results["combined_macro_f1"]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 (v2: post + timeline level)")
    parser.add_argument("--gold-dir", required=True, help="Directory with gold JSON timeline files")
    parser.add_argument("--pred-file", required=True, help="Prediction JSON file")
    parser.add_argument("--output", help="Optional: save results to JSON file")
    args = parser.parse_args()

    results = evaluate(args.gold_dir, args.pred_file)

    if args.output:
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print("\nResults saved to %s" % args.output)


if __name__ == "__main__":
    main()
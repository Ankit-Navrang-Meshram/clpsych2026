"""Evaluation for Task 1.1 (ABCD subelement classification) and Task 1.2 (presence rating)."""

import argparse
import json
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from utils import (
    ELEMENTS,
    NUM_SUBELEMENTS,
    SUBELEMENT_SCHEMA,
    VALENCES,
    build_pred_lookup,
    convert_gold_category_to_v2,
    get_gold_subelement,
    get_presence,
    get_subelement,
    load_gold_data,
    load_predictions,
    post_has_any_evidence,
    post_has_evidence,
    validate_predictions_coverage,
)


# ---------------------------------------------------------------------------
# Task 1.1 — ABCD Element Presence + Subelement Classification
# ---------------------------------------------------------------------------

def evaluate_task1_1(gold, pred_lookup, matched_keys):
    """Evaluate element presence (binary) and subelement accuracy (among TPs)."""
    results = {}

    # Element presence: binary per element x valence
    gold_elem_binary = {v: {e: [] for e in ELEMENTS} for v in VALENCES}
    pred_elem_binary = {v: {e: [] for e in ELEMENTS} for v in VALENCES}

    # Subelement: multi-class per element (0=absent, 1..K=subelement index)
    # For each post's adaptive state: vector [A_sub, BO_sub, BS_sub, CO_sub, CS_sub, D_sub]
    gold_sub_labels = {v: {e: [] for e in ELEMENTS} for v in VALENCES}
    pred_sub_labels = {v: {e: [] for e in ELEMENTS} for v in VALENCES}

    for mkey in matched_keys:
        gpost = gold[mkey]
        pred = pred_lookup[mkey]
        gevidence = gpost.get("evidence", {})

        for valence in VALENCES:
            if not post_has_evidence(gpost, valence):
                continue

            gstate = gevidence.get(valence, {})
            pstate = pred.get(valence, {})

            for elem in ELEMENTS:
                g_sub = get_gold_subelement(gstate, elem, valence)
                p_sub = get_subelement(pstate, elem)

                # Convert prediction from global (Table 1) numbering to v2
                if p_sub is not None:
                    p_sub = convert_gold_category_to_v2(
                        valence, elem, "(%d)" % p_sub)
                    # If mapping fails (invalid ID), treat as absent
                    if p_sub is None:
                        p_has = False

                g_has = g_sub is not None
                p_has = p_sub is not None

                gold_elem_binary[valence][elem].append(int(g_has))
                pred_elem_binary[valence][elem].append(int(p_has))

                # Multi-class subelement labels (0=absent)
                gold_sub_labels[valence][elem].append(g_sub if g_sub is not None else 0)
                pred_sub_labels[valence][elem].append(p_sub if p_sub is not None else 0)

    # --- Element presence metrics ---
    element_results = {}
    all_gold_elem = []
    all_pred_elem = []

    # Per-valence tracking
    valence_f1s = {v: [] for v in VALENCES}
    valence_gold = {v: [] for v in VALENCES}
    valence_pred = {v: [] for v in VALENCES}

    for valence in VALENCES:
        for elem in ELEMENTS:
            g = gold_elem_binary[valence][elem]
            p = pred_elem_binary[valence][elem]
            if not g:
                continue
            label = "%s:%s" % (valence, elem)
            prec = precision_score(g, p)
            rec = recall_score(g, p)
            f1 = f1_score(g, p)
            element_results[label] = {
                "precision": prec, "recall": rec, "f1": f1, "support": sum(g),
            }
            all_gold_elem.extend(g)
            all_pred_elem.extend(p)
            valence_f1s[valence].append(f1)
            valence_gold[valence].extend(g)
            valence_pred[valence].extend(p)

    # Overall
    elem_macro_f1 = np.mean([v["f1"] for v in element_results.values()])
    elem_micro_f1 = f1_score(all_gold_elem, all_pred_elem)

    # Per-valence aggregates
    valence_metrics = {}
    for valence in VALENCES:
        if valence_f1s[valence]:
            valence_metrics[valence] = {
                "macro_f1": np.mean(valence_f1s[valence]),
                "micro_f1": f1_score(valence_gold[valence], valence_pred[valence]),
            }

    # Average of adaptive and maladaptive
    if len(valence_metrics) == 2:
        avg_macro_f1 = np.mean([valence_metrics[v]["macro_f1"] for v in VALENCES])
        avg_micro_f1 = np.mean([valence_metrics[v]["micro_f1"] for v in VALENCES])
    else:
        avg_macro_f1 = elem_macro_f1
        avg_micro_f1 = elem_micro_f1

    results["element_presence"] = {
        "per_element": element_results,
        "adaptive_macro_f1": valence_metrics.get("adaptive-state", {}).get("macro_f1", 0.0),
        "adaptive_micro_f1": valence_metrics.get("adaptive-state", {}).get("micro_f1", 0.0),
        "maladaptive_macro_f1": valence_metrics.get("maladaptive-state", {}).get("macro_f1", 0.0),
        "maladaptive_micro_f1": valence_metrics.get("maladaptive-state", {}).get("micro_f1", 0.0),
        "avg_macro_f1": avg_macro_f1,
        "avg_micro_f1": avg_micro_f1,
        "macro_f1": elem_macro_f1,
        "micro_f1": elem_micro_f1,
    }

    # --- Subelement classification F1 (multi-class per element) ---
    # For each element: classes = {0=absent, 1, 2, ..., K}
    # Compute F1 over positive classes only (exclude 0=absent, already covered by element presence)
    sub_elem_results = {}
    all_gold_sub = []
    all_pred_sub = []

    # Per-valence tracking
    valence_sub_f1s = {v: [] for v in VALENCES}
    valence_sub_gold = {v: [] for v in VALENCES}
    valence_sub_pred = {v: [] for v in VALENCES}

    for valence in VALENCES:
        for elem in ELEMENTS:
            g = gold_sub_labels[valence][elem]
            p = pred_sub_labels[valence][elem]
            if not g:
                continue

            # Positive classes (exclude 0=absent)
            n_subs = NUM_SUBELEMENTS[valence][elem]
            pos_labels = list(range(1, n_subs + 1))

            label = "%s:%s" % (valence, elem)
            macro_f1 = f1_score(g, p, labels=pos_labels, average="macro")
            micro_f1 = f1_score(g, p, labels=pos_labels, average="micro")
            support = sum(1 for x in g if x > 0)

            sub_elem_results[label] = {
                "macro_f1": macro_f1, "micro_f1": micro_f1, "support": support,
            }
            valence_sub_f1s[valence].append(macro_f1)

            # For valence-level micro: use globally unique labels "ELEM:k"
            valence_sub_gold[valence].extend(
                ["%s:%d" % (elem, x) if x > 0 else "0" for x in g])
            valence_sub_pred[valence].extend(
                ["%s:%d" % (elem, x) if x > 0 else "0" for x in p])

    # Per-valence aggregates
    valence_sub_metrics = {}
    for valence in VALENCES:
        if valence_sub_f1s[valence]:
            g_v = valence_sub_gold[valence]
            p_v = valence_sub_pred[valence]
            pos_labels_v = sorted(set(g_v) | set(p_v))
            pos_labels_v = [l for l in pos_labels_v if l != "0"]
            valence_sub_metrics[valence] = {
                "macro_f1": np.mean(valence_sub_f1s[valence]),
                "micro_f1": f1_score(g_v, p_v, labels=pos_labels_v, average="micro"),
            }

    # Overall: pool both valences
    all_g = valence_sub_gold.get("adaptive-state", []) + valence_sub_gold.get("maladaptive-state", [])
    all_p = valence_sub_pred.get("adaptive-state", []) + valence_sub_pred.get("maladaptive-state", [])
    all_pos = sorted(set(all_g) | set(all_p))
    all_pos = [l for l in all_pos if l != "0"]
    overall_macro_f1 = np.mean([v["macro_f1"] for v in sub_elem_results.values()]) if sub_elem_results else 0.0
    overall_micro_f1 = f1_score(all_g, all_p, labels=all_pos, average="micro") if all_pos else 0.0

    # Average of adaptive and maladaptive
    if len(valence_sub_metrics) == 2:
        avg_sub_macro_f1 = np.mean([valence_sub_metrics[v]["macro_f1"] for v in VALENCES])
        avg_sub_micro_f1 = np.mean([valence_sub_metrics[v]["micro_f1"] for v in VALENCES])
    else:
        avg_sub_macro_f1 = overall_macro_f1
        avg_sub_micro_f1 = overall_micro_f1

    results["subelement_classification"] = {
        "per_element": sub_elem_results,
        "adaptive_macro_f1": valence_sub_metrics.get("adaptive-state", {}).get("macro_f1", 0.0),
        "adaptive_micro_f1": valence_sub_metrics.get("adaptive-state", {}).get("micro_f1", 0.0),
        "maladaptive_macro_f1": valence_sub_metrics.get("maladaptive-state", {}).get("macro_f1", 0.0),
        "maladaptive_micro_f1": valence_sub_metrics.get("maladaptive-state", {}).get("micro_f1", 0.0),
        "avg_macro_f1": avg_sub_macro_f1,
        "avg_micro_f1": avg_sub_micro_f1,
        "macro_f1": overall_macro_f1,
        "micro_f1": overall_micro_f1,
    }

    return results


# ---------------------------------------------------------------------------
# Task 1.2 — Presence Rating (1-5 scale)
# ---------------------------------------------------------------------------

def evaluate_task1_2(gold, pred_lookup, matched_keys):
    """Evaluate presence ratings: MAE, RMSE, QWK, Spearman."""
    ratings = {v: {"gold": [], "pred": []} for v in VALENCES}

    for key in matched_keys:
        gpost = gold[key]
        pred = pred_lookup[key]
        gevidence = gpost.get("evidence", {})

        for valence in VALENCES:
            if not post_has_evidence(gpost, valence):
                continue

            gstate = gevidence.get(valence, {})
            pstate = pred.get(valence, {})

            g_pres = get_presence(gstate)
            p_pres = get_presence(pstate)

            if g_pres is None:
                continue
            if p_pres is None:
                p_pres = 1

            ratings[valence]["gold"].append(g_pres)
            ratings[valence]["pred"].append(p_pres)

    results = {}
    all_gold = []
    all_pred = []

    for valence in VALENCES:
        g = np.array(ratings[valence]["gold"])
        p = np.array(ratings[valence]["pred"])
        if len(g) == 0:
            continue

        mae = mean_absolute_error(g, p)
        rmse = np.sqrt(mean_squared_error(g, p))
        qwk = cohen_kappa_score(g, p, weights="quadratic")
        rho, _ = spearmanr(g, p)

        results[valence] = {
            "mae": mae, "rmse": rmse, "qwk": qwk, "spearman": rho, "n": len(g),
        }
        all_gold.extend(g.tolist())
        all_pred.extend(p.tolist())

    if all_gold:
        g = np.array(all_gold)
        p = np.array(all_pred)
        results["combined"] = {
            "mae": mean_absolute_error(g, p),
            "rmse": np.sqrt(mean_squared_error(g, p)),
            "qwk": cohen_kappa_score(g, p, weights="quadratic"),
            "spearman": spearmanr(g, p)[0],
            "n": len(g),
        }

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_task1_1_results(results):
    print("=" * 70)
    print("TASK 1.1 — ABCD Element & Subelement Classification")
    print("=" * 70)

    ep = results["element_presence"]
    print("\n--- Element Presence (Binary) ---")
    print("%-35s %7s %7s %7s %6s" % ("Element", "Prec", "Rec", "F1", "Supp"))
    print("-" * 65)
    for label, m in sorted(ep["per_element"].items()):
        print("%-35s %7.3f %7.3f %7.3f %6d" % (
            label, m["precision"], m["recall"], m["f1"], m["support"]))
    print("-" * 65)
    print("%-35s %7s %7s %7.3f" % ("Adaptive Macro F1", "", "", ep["adaptive_macro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Adaptive Micro F1", "", "", ep["adaptive_micro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Maladaptive Macro F1", "", "", ep["maladaptive_macro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Maladaptive Micro F1", "", "", ep["maladaptive_micro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Avg Macro F1", "", "", ep["avg_macro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Avg Micro F1", "", "", ep["avg_micro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Overall Macro F1", "", "", ep["macro_f1"]))
    print("%-35s %7s %7s %7.3f" % ("Overall Micro F1", "", "", ep["micro_f1"]))

    sc = results["subelement_classification"]
    print("\n--- Subelement Classification (multi-class per element) ---")
    print("%-35s %7s %7s %6s" % ("Element", "MacroF1", "MicroF1", "Supp"))
    print("-" * 55)
    for label, m in sorted(sc["per_element"].items()):
        print("%-35s %7.3f %7.3f %6d" % (
            label, m["macro_f1"], m["micro_f1"], m["support"]))
    print("-" * 55)
    print("%-35s %7.3f %7.3f" % ("Adaptive", sc["adaptive_macro_f1"], sc["adaptive_micro_f1"]))
    print("%-35s %7.3f %7.3f" % ("Maladaptive", sc["maladaptive_macro_f1"], sc["maladaptive_micro_f1"]))
    print("%-35s %7.3f %7.3f" % ("Avg", sc["avg_macro_f1"], sc["avg_micro_f1"]))
    print("%-35s %7.3f %7.3f" % ("Overall", sc["macro_f1"], sc["micro_f1"]))


def print_task1_2_results(results):
    print("\n" + "=" * 70)
    print("TASK 1.2 — Presence Rating (1-5)")
    print("=" * 70)
    print("%-25s %7s %7s %7s %7s %6s" % ("Valence", "MAE", "RMSE", "QWK", "Spear", "N"))
    print("-" * 62)
    for label in VALENCES + ["combined"]:
        if label not in results:
            continue
        m = results[label]
        print("%-25s %7.3f %7.3f %7.3f %7.3f %6d" % (
            label, m["mae"], m["rmse"], m["qwk"], m["spearman"], m["n"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(gold_dir, pred_file):
    """Run full Task 1 evaluation. Returns results dict."""
    gold = load_gold_data(gold_dir)
    predictions = load_predictions(pred_file)
    pred_lookup = build_pred_lookup(predictions)

    matched = validate_predictions_coverage(
        gold, pred_lookup, "Task 1",
        filter_fn=post_has_any_evidence,
    )

    results = {
        "task1_1": evaluate_task1_1(gold, pred_lookup, matched),
        "task1_2": evaluate_task1_2(gold, pred_lookup, matched),
    }

    print_task1_1_results(results["task1_1"])
    print_task1_2_results(results["task1_2"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 1 (v4)")
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
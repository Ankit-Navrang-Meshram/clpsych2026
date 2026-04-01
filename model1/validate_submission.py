"""Validate submission files for CLPsych 2026 Shared Task.

Usage:
  python validate_submission.py --task1 task1_pred.json --task2 task2_pred.json --test-dir <test_dir>
"""

import argparse
import json
import sys
from pathlib import Path

ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
VALENCES = ["adaptive-state", "maladaptive-state"]
PRESENCE_RANGE = range(1, 6)
SWITCH_VALUES = {"0", "S"}
ESCALATION_VALUES = {"0", "E"}

# Valid subelement indices per element x valence (global numbering from Table 1)
VALID_SUBELEMENTS = {
    "adaptive-state": {
        "A": {1, 3, 5, 7, 9, 11, 13},
        "B-O": {1, 3},
        "B-S": {1},
        "C-O": {1, 3},
        "C-S": {1},
        "D": {1, 3, 5},
    },
    "maladaptive-state": {
        "A": {2, 4, 6, 8, 10, 12, 14},
        "B-O": {2, 4},
        "B-S": {2},
        "C-O": {2, 4},
        "C-S": {2},
        "D": {2, 4, 6},
    },
}


# Keys that suggest post text is included — these should NOT be in submissions
POST_TEXT_KEYS = {"post", "post_text", "text", "content", "body", "Post Summary"}


def check_post_text(data, task_name):
    """Warn if any entry contains fields that look like post text."""
    warnings = []
    flagged = 0
    for i, entry in enumerate(data):
        found_keys = [k for k in entry if k in POST_TEXT_KEYS]
        if found_keys:
            flagged += 1
            if flagged <= 3:
                warnings.append(
                    "Entry %d: contains text field(s) %s — please remove before submitting"
                    % (i, found_keys))
    if flagged > 3:
        warnings.append("... and %d more entries with text fields" % (flagged - 3))
    if flagged > 0:
        warnings.insert(0,
            "WARNING: %d %s entries contain post text. "
            "For privacy reasons, please remove all post text fields "
            "(e.g., 'post', 'text', 'body') before uploading your submission."
            % (flagged, task_name))
    return warnings


def load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, "Invalid JSON: %s" % e
    except FileNotFoundError:
        return None, "File not found: %s" % path
    if not isinstance(data, list):
        return None, "Top-level structure must be a JSON array"
    return data, None


def load_test_posts(test_dir):
    posts = set()
    for fpath in sorted(Path(test_dir).glob("*.json")):
        with open(fpath) as f:
            timeline = json.load(f)
        tid = timeline["timeline_id"]
        for post in timeline["posts"]:
            posts.add((tid, post["post_id"]))
    return posts


def validate_task1(data, test_posts=None):
    errors = []
    seen = set()

    for i, entry in enumerate(data):
        prefix = "Entry %d" % i

        if "timeline_id" not in entry:
            errors.append("%s: missing 'timeline_id'" % prefix)
            continue
        if "post_id" not in entry:
            errors.append("%s: missing 'post_id'" % prefix)
            continue

        tid = entry["timeline_id"]
        pid = entry["post_id"]
        prefix = "Entry %d (timeline=%s, post=%s)" % (i, tid, pid)

        if not isinstance(tid, str) or not isinstance(pid, str):
            errors.append("%s: 'timeline_id' and 'post_id' must be strings" % prefix)
            continue

        key = (tid, pid)
        if key in seen:
            errors.append("%s: duplicate entry" % prefix)
        seen.add(key)

        has_any_valence = False
        for valence in VALENCES:
            if valence not in entry:
                continue
            has_any_valence = True
            state = entry[valence]

            if not isinstance(state, dict):
                errors.append("%s: '%s' must be an object" % (prefix, valence))
                continue

            # Presence required
            if "Presence" not in state:
                errors.append("%s: '%s' missing 'Presence'" % (prefix, valence))
            else:
                pres = state["Presence"]
                if not isinstance(pres, int) or pres not in PRESENCE_RANGE:
                    errors.append("%s: '%s.Presence' must be integer 1-5, got %r" % (prefix, valence, pres))

            # Check element keys
            for k, v in state.items():
                if k == "Presence":
                    continue
                if k not in ELEMENTS:
                    errors.append("%s: '%s' has unknown element '%s'" % (prefix, valence, k))
                    continue
                if not isinstance(v, dict):
                    errors.append("%s: '%s.%s' must be an object" % (prefix, valence, k))
                    continue
                if "subelement" not in v:
                    errors.append("%s: '%s.%s' missing 'subelement'" % (prefix, valence, k))
                    continue
                sub = v["subelement"]
                valid = VALID_SUBELEMENTS[valence][k]
                if not isinstance(sub, int) or sub not in valid:
                    errors.append(
                        "%s: '%s.%s.subelement' must be one of %s, got %r" % (
                            prefix, valence, k, sorted(valid), sub))

        if not has_any_valence:
            errors.append("%s: must have at least one of 'adaptive-state' or 'maladaptive-state'" % prefix)

    if test_posts is not None:
        extra = seen - test_posts
        if extra:
            errors.append("Found %d posts not in test data (first 5: %s)" % (len(extra), list(extra)[:5]))

    return errors


def validate_task2(data, test_posts=None):
    errors = []
    seen = set()

    for i, entry in enumerate(data):
        prefix = "Entry %d" % i

        if "timeline_id" not in entry:
            errors.append("%s: missing 'timeline_id'" % prefix)
            continue
        if "post_id" not in entry:
            errors.append("%s: missing 'post_id'" % prefix)
            continue

        tid = entry["timeline_id"]
        pid = entry["post_id"]
        prefix = "Entry %d (timeline=%s, post=%s)" % (i, tid, pid)

        if not isinstance(tid, str) or not isinstance(pid, str):
            errors.append("%s: 'timeline_id' and 'post_id' must be strings" % prefix)
            continue

        key = (tid, pid)
        if key in seen:
            errors.append("%s: duplicate entry" % prefix)
        seen.add(key)

        if "Switch" not in entry:
            errors.append("%s: missing 'Switch'" % prefix)
        elif entry["Switch"] not in SWITCH_VALUES:
            errors.append("%s: 'Switch' must be '0' or 'S', got %r" % (prefix, entry["Switch"]))

        if "Escalation" not in entry:
            errors.append("%s: missing 'Escalation'" % prefix)
        elif entry["Escalation"] not in ESCALATION_VALUES:
            errors.append("%s: 'Escalation' must be '0' or 'E', got %r" % (prefix, entry["Escalation"]))

    if test_posts is not None:
        missing = test_posts - seen
        extra = seen - test_posts
        if missing:
            errors.append("Missing %d test posts (first 5: %s)" % (len(missing), list(missing)[:5]))
        if extra:
            errors.append("Found %d posts not in test data (first 5: %s)" % (len(extra), list(extra)[:5]))

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate CLPsych 2026 submission files (v3 format)")
    parser.add_argument("--task1", help="Task 1 prediction JSON file")
    parser.add_argument("--task2", help="Task 2 prediction JSON file")
    parser.add_argument("--test-dir", help="Test data directory (for coverage checks)")
    args = parser.parse_args()

    if not args.task1 and not args.task2:
        parser.error("At least one of --task1 or --task2 is required")

    test_posts = None
    if args.test_dir:
        test_posts = load_test_posts(args.test_dir)
        print("Loaded %d posts from test data\n" % len(test_posts))

    all_valid = True

    if args.task1:
        print("=" * 50)
        print("Validating Task 1 submission...")
        print("=" * 50)
        data, load_err = load_json(args.task1)
        if load_err:
            print("FAILED: %s" % load_err)
            all_valid = False
        else:
            text_warnings = check_post_text(data, "Task 1")
            if text_warnings:
                for w in text_warnings:
                    print("  * %s" % w)
                print()
            errors = validate_task1(data, test_posts)
            if errors:
                all_valid = False
                print("FAILED: %d error(s) found:\n" % len(errors))
                for e in errors[:20]:
                    print("  - %s" % e)
                if len(errors) > 20:
                    print("  ... and %d more" % (len(errors) - 20))
            else:
                print("PASSED: %d entries, all valid." % len(data))
        print()

    if args.task2:
        print("=" * 50)
        print("Validating Task 2 submission...")
        print("=" * 50)
        data, load_err = load_json(args.task2)
        if load_err:
            print("FAILED: %s" % load_err)
            all_valid = False
        else:
            text_warnings = check_post_text(data, "Task 2")
            if text_warnings:
                for w in text_warnings:
                    print("  * %s" % w)
                print()
            errors = validate_task2(data, test_posts)
            if errors:
                all_valid = False
                print("FAILED: %d error(s) found:\n" % len(errors))
                for e in errors[:20]:
                    print("  - %s" % e)
                if len(errors) > 20:
                    print("  ... and %d more" % (len(errors) - 20))
            else:
                print("PASSED: %d entries, all valid." % len(data))
        print()

    if all_valid:
        print("All checks passed. Ready to submit!")
        sys.exit(0)
    else:
        print("Validation failed. Please fix the errors above before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
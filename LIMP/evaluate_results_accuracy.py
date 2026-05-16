#!/usr/bin/env python3
"""Accuracy monitor for labeled train/dev/local_eval MuMA-ToM runs."""

import argparse
import json
import re
import time
from pathlib import Path


SAFE_ANSWER_PATH_MARKERS = {"training_set", "train", "dev", "validation", "val", "local_eval"}
UNSAFE_PUBLIC_ANSWER_PATHS = {
    Path("Files/questions.json"),
    Path("MUMA-TOM-BENCHMARK/questions.json"),
}


def is_safe_answer_path(path):
    resolved = Path(path).resolve()
    lower_parts = {part.lower() for part in resolved.parts}
    if not (lower_parts & SAFE_ANSWER_PATH_MARKERS):
        return False
    resolved_text = str(resolved)
    return not any(str(unsafe_path) in resolved_text for unsafe_path in UNSAFE_PUBLIC_ANSWER_PATHS)


def answer_letter(value):
    if isinstance(value, list) and value:
        value = value[0]
    text = str(value or "").strip()
    match = re.match(r"^\s*([A-C])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-C])\)", text)
    return match.group(1) if match else None


def load_json(path):
    with Path(path).open("r") as file:
        return json.load(file)


def extract_answers(data):
    answers = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            episode_id = item.get("scenario_id", item.get("episode_id", item.get("episode")))
            question_id = item.get("question_id", item.get("qid", item.get("question")))
            letter = answer_letter(item.get("answer", item.get("gold", item.get("label"))))
            if episode_id is not None and question_id is not None and letter:
                answers.setdefault(str(episode_id), {})[str(question_id)] = letter
        return answers

    if not isinstance(data, dict):
        return answers

    for episode_id, episode_blob in data.items():
        if not isinstance(episode_blob, dict):
            continue
        raw_answers = None
        for key in ("answers", "answer", "gold"):
            if isinstance(episode_blob.get(key), dict):
                raw_answers = episode_blob[key]
                break
        if raw_answers is None and all(str(key).isdigit() for key in episode_blob.keys()):
            raw_answers = episode_blob
        if not isinstance(raw_answers, dict):
            continue
        for question_id, answer in raw_answers.items():
            letter = answer_letter(answer)
            if letter:
                answers.setdefault(str(episode_id), {})[str(question_id)] = letter
    return answers


def iter_predictions(results):
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            episode_id = item.get("scenario_id", item.get("episode_id", item.get("episode")))
            question_id = item.get("question_id", item.get("qid", item.get("question")))
            prediction = answer_letter(item.get("prediction", item.get("answer")))
            if episode_id is not None and question_id is not None and prediction:
                yield str(episode_id), str(question_id), prediction, item
        return

    if not isinstance(results, dict):
        return
    for episode_id, episode_record in results.get("episodes", {}).items():
        predictions = episode_record.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        for question_id, prediction_record in predictions.items():
            if not isinstance(prediction_record, dict):
                continue
            prediction = answer_letter(prediction_record.get("prediction"))
            if prediction:
                yield str(episode_id), str(question_id), prediction, prediction_record


def compute_accuracy(results, answers):
    total = 0
    correct = 0
    missing_answer = 0
    by_label = {}
    by_source = {}
    mistakes = []

    for episode_id, question_id, prediction, record in iter_predictions(results):
        gold = answers.get(episode_id, {}).get(question_id)
        if gold is None:
            missing_answer += 1
            continue
        label = record.get("question_label") or "unknown"
        source = record.get("base_prediction_source") or record.get("prediction_source") or "unknown"
        label_bucket = by_label.setdefault(label, {"correct": 0, "total": 0})
        source_bucket = by_source.setdefault(source, {"correct": 0, "total": 0})
        total += 1
        label_bucket["total"] += 1
        source_bucket["total"] += 1
        if prediction == gold:
            correct += 1
            label_bucket["correct"] += 1
            source_bucket["correct"] += 1
        elif len(mistakes) < 20:
            mistakes.append(
                {
                    "episode_id": episode_id,
                    "question_id": question_id,
                    "prediction": prediction,
                    "gold": gold,
                    "question_label": label,
                    "source": source,
                    "score_margin": record.get("score_margin"),
                }
            )

    for bucket in list(by_label.values()) + list(by_source.values()):
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "missing_answer": missing_answer,
        "by_label": by_label,
        "by_source": by_source,
        "sample_mistakes": mistakes,
    }


def print_summary(summary):
    print(
        f"Accuracy: {summary['correct']}/{summary['total']} "
        f"({summary['accuracy']:.4f}); missing answers: {summary['missing_answer']}"
    )
    if summary["by_label"]:
        print("By label:")
        for label, bucket in sorted(summary["by_label"].items()):
            print(f"  {label}: {bucket['correct']}/{bucket['total']} ({bucket['accuracy']:.4f})")
    if summary["by_source"]:
        print("By source:")
        for source, bucket in sorted(summary["by_source"].items()):
            print(f"  {source}: {bucket['correct']}/{bucket['total']} ({bucket['accuracy']:.4f})")
    if summary["sample_mistakes"]:
        print("Sample mistakes:")
        for item in summary["sample_mistakes"]:
            print(
                "  "
                f"episode={item['episode_id']} question={item['question_id']} "
                f"pred={item['prediction']} gold={item['gold']} "
                f"label={item['question_label']} source={item['source']} "
                f"margin={item['score_margin']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy for a MuMA-ToM results JSON file.")
    parser.add_argument("--results", required=True, help="Path to results JSON.")
    parser.add_argument(
        "--answers",
        required=True,
        help="Path to train/dev/local_eval answers JSON. Public test questions.json is refused.",
    )
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds. 0 means run once.")
    args = parser.parse_args()

    if not is_safe_answer_path(args.answers):
        raise SystemExit(
            "Refusing this answers path. Use an official training/dev/local_eval annotation file, "
            "not public test labels."
        )

    answers = extract_answers(load_json(args.answers))
    if not answers:
        raise SystemExit(f"No answers found in {args.answers}")

    while True:
        try:
            summary = compute_accuracy(load_json(args.results), answers)
            print_summary(summary)
        except FileNotFoundError:
            print(f"Waiting for results file: {args.results}")
        except json.JSONDecodeError as exc:
            print(f"Waiting for valid JSON: {exc}")

        if args.watch <= 0:
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()

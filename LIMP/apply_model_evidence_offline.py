from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import LIMP


DEFAULT_RESULTS_FILE = (
    Path(__file__).resolve().parent.parent
    / "local_runs"
    / "qwen3_5_27b"
    / "results_new_16_943.json"
)
DEFAULT_EVIDENCE_FILE = (
    Path(__file__).resolve().parent.parent
    / "output_evidence"
    / "qwen3_5_27b"
    / "model_evidence_strong.json"
)
DEFAULT_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "local_runs"
    / "qwen3_5_27b"
    / "results_new_16_943_evidence_rescored.json"
)


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def answer_letter(answer: Any) -> str | None:
    return LIMP._answer_letter(answer)


def choice_from_scores(option_scores: dict[str, Any], prompt: str, fallback: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    if not option_scores:
        return fallback, None
    values = {choice: float(score) for choice, score in option_scores.items()}
    target = min(values.values()) if "LEAST LIKELY" in prompt.upper() else max(values.values())
    tied = [
        choice
        for choice, score in values.items()
        if abs(score - target) <= 1e-12
    ]
    if len(tied) == 1:
        return tied[0], None
    if fallback in tied:
        return fallback, {
            "method": "offline_no_model_tie_keep_fallback",
            "tied_choices": tied,
            "fallback_choice": fallback,
        }
    return sorted(tied)[0], {
        "method": "offline_no_model_tie_alphabetical",
        "tied_choices": tied,
        "fallback_choice": sorted(tied)[0],
    }


def get_evidence(evidence_data: dict[str, Any], evidence_file: Path, episode_id: str, question_id: str) -> dict[str, Any] | None:
    episode_record = evidence_data.get("episodes", {}).get(str(episode_id))
    if not isinstance(episode_record, dict):
        return None
    evidence_bucket = episode_record.get("evidence")
    if isinstance(evidence_bucket, dict) and isinstance(evidence_bucket.get(str(question_id)), dict):
        evidence = dict(evidence_bucket[str(question_id)])
        evidence["evidence_cache_source"] = str(evidence_file)
        return evidence
    for key in ("predictions", "questions"):
        bucket = episode_record.get(key)
        if not isinstance(bucket, dict):
            continue
        candidate = bucket.get(str(question_id))
        if isinstance(candidate, dict) and isinstance(candidate.get("model_evidence"), dict):
            evidence = dict(candidate["model_evidence"])
            evidence["evidence_cache_source"] = str(evidence_file)
            return evidence
    return None


def refresh_summary(results: dict[str, Any]) -> dict[str, Any]:
    total = 0
    correct = 0
    by_label: dict[str, dict[str, Any]] = {}
    changed = 0
    evidence_available = 0
    prior_applied = 0
    for episode_record in results.get("episodes", {}).values():
        predictions = episode_record.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        for prediction in predictions.values():
            if not isinstance(prediction, dict):
                continue
            pred = answer_letter(prediction.get("prediction"))
            gold = answer_letter(prediction.get("gold"))
            if gold is None:
                continue
            label = prediction.get("question_label") or "unknown"
            label_record = by_label.setdefault(label, {"correct": 0, "total": 0, "accuracy": 0.0})
            total += 1
            label_record["total"] += 1
            if pred == gold:
                correct += 1
                label_record["correct"] += 1
            if prediction.get("evidence_rescore_changed"):
                changed += 1
            if prediction.get("model_evidence"):
                evidence_available += 1
            if (prediction.get("evidence_prior") or {}).get("applied"):
                prior_applied += 1

    for label_record in by_label.values():
        label_record["accuracy"] = (
            label_record["correct"] / label_record["total"]
            if label_record["total"]
            else 0.0
        )
    summary = {
        "total_questions": total,
        "total_correct": correct,
        "accuracy": correct / total if total else 0.0,
        "changed_predictions": changed,
        "evidence_available": evidence_available,
        "evidence_prior_applied": prior_applied,
        "by_label": by_label,
    }
    results["offline_evidence_summary"] = summary
    return summary


def summarize_option_baseline(results: dict[str, Any]) -> dict[str, Any]:
    total = 0
    correct = 0
    by_label: dict[str, dict[str, Any]] = {}
    for episode_record in results.get("episodes", {}).values():
        predictions = episode_record.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        for prediction in predictions.values():
            if not isinstance(prediction, dict):
                continue
            prompt = prediction.get("question", "")
            label = prediction.get("question_label") or LIMP._infer_question_label_from_prompt(prompt) or "unknown"
            pred = answer_letter(prediction.get("score_based_prediction"))
            if pred is None and isinstance(prediction.get("option_scores"), dict):
                pred, _ = choice_from_scores(
                    prediction["option_scores"],
                    prompt,
                    fallback=answer_letter(prediction.get("prediction")),
                )
            gold = answer_letter(prediction.get("gold"))
            if pred is None or gold is None:
                continue
            label_record = by_label.setdefault(label, {"correct": 0, "total": 0, "accuracy": 0.0})
            total += 1
            label_record["total"] += 1
            if pred == gold:
                correct += 1
                label_record["correct"] += 1
    for label_record in by_label.values():
        label_record["accuracy"] = (
            label_record["correct"] / label_record["total"]
            if label_record["total"]
            else 0.0
        )
    return {
        "total_questions": total,
        "total_correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_label": by_label,
    }


def rescore_prediction(
    prediction: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = deepcopy(prediction)
    prompt = updated.get("question", "")
    question_label = (
        updated.get("question_label")
        or (evidence or {}).get("question_label")
        or LIMP._infer_question_label_from_prompt(prompt)
    )
    updated["question_label"] = question_label
    updated["model_evidence"] = evidence
    updated["evidence_prior"] = None
    updated["prior_adjusted_option_scores"] = None
    updated["evidence_rescore_tie_break"] = None

    original_prediction = answer_letter(updated.get("prediction"))
    base_prediction = answer_letter(updated.get("score_based_prediction")) or answer_letter(updated.get("base_prediction")) or original_prediction
    base_source = "offline_original_score_based"

    if (
        question_label == "social_goal"
        and evidence is not None
        and isinstance(updated.get("social_goal_posterior"), dict)
    ):
        posterior, prior = LIMP._apply_social_goal_rule_prior(
            updated["social_goal_posterior"],
            evidence,
            prompt=prompt,
        )
        updated["social_goal_posterior"] = posterior
        updated["evidence_prior"] = prior
        posterior_choice, selection = LIMP._select_social_goal_choice_from_posterior(
            prompt,
            LIMP._parse_question_options(prompt)[1],
            posterior,
        )
        updated["social_goal_selection"] = selection
        if posterior_choice is not None:
            base_prediction = posterior_choice
            base_source = (
                "offline_social_goal_posterior_with_model_evidence_prior"
                if (prior or {}).get("applied")
                else "offline_social_goal_posterior"
            )
    else:
        option_scores = updated.get("option_scores") if isinstance(updated.get("option_scores"), dict) else {}
        if evidence is not None:
            adjusted_scores, prior = LIMP._apply_option_score_rule_prior(
                question_label,
                prompt,
                option_scores,
                evidence,
            )
            updated["prior_adjusted_option_scores"] = adjusted_scores
            updated["evidence_prior"] = prior
            if prior is not None:
                prior_choice, tie_break = choice_from_scores(adjusted_scores, prompt, fallback=base_prediction)
                updated["evidence_rescore_tie_break"] = tie_break
                if prior_choice is not None:
                    base_prediction = prior_choice
                    base_source = (
                        "offline_option_scores_with_model_evidence_prior"
                        if prior.get("applied")
                        else "offline_option_scores_prior_not_applied"
                    )
        else:
            base_prediction, tie_break = choice_from_scores(option_scores, prompt, fallback=base_prediction)
            updated["evidence_rescore_tie_break"] = tie_break

    updated["base_prediction"] = base_prediction
    updated["base_prediction_source"] = base_source
    updated["prediction"] = base_prediction
    updated["evidence_prior_applied"] = bool((updated.get("evidence_prior") or {}).get("applied"))
    updated["evidence_rescore_changed"] = base_prediction != original_prediction
    if updated.get("gold") is not None:
        updated["correct"] = base_prediction == answer_letter(updated.get("gold"))
    return updated


def rescore(results: dict[str, Any], evidence_data: dict[str, Any], evidence_file: Path) -> dict[str, Any]:
    output = deepcopy(results)
    output.setdefault("config", {})["offline_evidence_file"] = str(evidence_file)
    output["config"]["enable_belief_of_goal_evidence_prior"] = True
    output["config"]["enable_social_goal_evidence_prior"] = LIMP._ENABLE_SOCIAL_GOAL_EVIDENCE_PRIOR
    for episode_id, episode_record in output.get("episodes", {}).items():
        predictions = episode_record.get("predictions")
        if not isinstance(predictions, dict):
            continue
        episode_correct = 0
        episode_total = 0
        for question_id, prediction in list(predictions.items()):
            evidence = get_evidence(evidence_data, evidence_file, str(episode_id), str(question_id))
            rescored = rescore_prediction(prediction, evidence)
            predictions[question_id] = rescored
            if rescored.get("gold") is not None:
                episode_total += 1
                if rescored.get("correct"):
                    episode_correct += 1
        if episode_total:
            episode_record["correct"] = episode_correct
            episode_record["total"] = episode_total
            episode_record["accuracy"] = episode_correct / episode_total
    refresh_summary(output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply cached model evidence to an existing LIMP results file without rerunning model inference."
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_FILE), help="Existing LIMP results JSON.")
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE_FILE), help="Cached model evidence JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE), help="Output rescored results JSON.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_file = Path(args.results)
    evidence_file = Path(args.evidence)
    output_file = Path(args.output)

    LIMP._ENABLE_BELIEF_OF_GOAL_EVIDENCE_PRIOR = True
    results = load_json(results_file)
    evidence_data = load_json(evidence_file)
    rescored = rescore(results, evidence_data, evidence_file)
    rescored["baseline_final_summary"] = results.get("summary", {})
    rescored["baseline_option_summary"] = summarize_option_baseline(results)
    save_json(output_file, rescored)

    baseline = results.get("summary", {})
    option_baseline = rescored["baseline_option_summary"]
    summary = rescored["offline_evidence_summary"]
    print("Baseline final:", json.dumps(baseline, ensure_ascii=False))
    print("Baseline option:", json.dumps(option_baseline, ensure_ascii=False))
    print("Rescored:", json.dumps(summary, ensure_ascii=False))
    print("Saved:", output_file)


if __name__ == "__main__":
    main()

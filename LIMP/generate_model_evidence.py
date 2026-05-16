from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

import LIMP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "Files" / "questions.json"
DEFAULT_TEXTS_FILE = PROJECT_ROOT / "Files" / "texts.json"


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def parse_strengths(raw: str) -> set[str]:
    return {
        item.strip().lower()
        for item in str(raw or "").split(",")
        if item.strip()
    }


def parse_episode_list(raw: str, all_episode_ids: list[int]) -> list[int]:
    raw = str(raw or "").strip()
    if not raw:
        return all_episode_ids
    if raw.startswith("["):
        parsed = ast.literal_eval(raw)
        return [int(item) for item in parsed]

    episodes: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(item))
    return episodes


def default_output() -> dict[str, Any]:
    return {
        "config": {},
        "checkpoint": {},
        "episodes": {},
        "summary": {
            "episodes": 0,
            "questions_seen": 0,
            "strong_evidence": 0,
            "skipped": 0,
            "errors": 0,
        },
    }


def load_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_output()
    try:
        loaded = load_json(path)
    except json.JSONDecodeError:
        return default_output()
    if not isinstance(loaded, dict):
        return default_output()
    loaded.setdefault("checkpoint", {})
    loaded.setdefault("episodes", {})
    loaded.setdefault("summary", {})
    return loaded


def ensure_episode_record(output: dict[str, Any], episode_id: int) -> dict[str, Any]:
    episodes = output.setdefault("episodes", {})
    episode_key = str(episode_id)
    record = episodes.get(episode_key)
    if not isinstance(record, dict):
        record = {}
        episodes[episode_key] = record
    record["episode_id"] = episode_id
    record.setdefault("evidence", {})
    record.setdefault("skipped", {})
    record.setdefault("questions", {})
    record.setdefault("errors", {})
    return record


def refresh_summary(output: dict[str, Any]) -> None:
    questions_seen = 0
    strong_evidence = 0
    skipped = 0
    errors = 0
    for record in output.get("episodes", {}).values():
        if not isinstance(record, dict):
            continue
        evidence_count = len(record.get("evidence", {})) if isinstance(record.get("evidence"), dict) else 0
        skipped_count = len(record.get("skipped", {})) if isinstance(record.get("skipped"), dict) else 0
        error_count = len(record.get("errors", {})) if isinstance(record.get("errors"), dict) else 0
        total_questions = int(record.get("total_questions", 0) or 0)
        record["strong_evidence_count"] = evidence_count
        record["skipped_count"] = skipped_count
        record["error_count"] = error_count
        questions_seen += total_questions
        strong_evidence += evidence_count
        skipped += skipped_count
        errors += error_count
    output["summary"] = {
        "episodes": len(output.get("episodes", {})),
        "questions_seen": questions_seen,
        "strong_evidence": strong_evidence,
        "skipped": skipped,
        "errors": errors,
    }


def question_done(record: dict[str, Any], question_id: str) -> bool:
    for key in ("evidence", "skipped"):
        entries = record.get(key)
        if isinstance(entries, dict) and question_id in entries:
            return True
    return False


def clear_question(record: dict[str, Any], question_id: str) -> None:
    for key in ("evidence", "skipped", "questions", "errors"):
        entries = record.get(key)
        if isinstance(entries, dict):
            entries.pop(question_id, None)


def person_has_actions(person_info: dict[str, Any]) -> bool:
    return bool(LIMP._person_actions(person_info))


def set_person_actions(person_info: dict[str, Any], actions: list[str]) -> None:
    if "actions" in person_info or "utterances" in person_info:
        person_info["actions"] = actions
    else:
        person_info["action"] = actions


def build_question_context(
    episode_id: int,
    question_id: str,
    prompt: str,
    questions: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    context_sources = ["description:questions"]
    episode_description = LIMP._clean_description(questions.get("description", ""))

    name_list = LIMP.extract_name_from_question(prompt)
    context_sources.append("names:limp_extract_name_from_question")
    if not isinstance(name_list, list) or not name_list:
        raise ValueError(f"No names extracted for episode {episode_id}, question {question_id}.")

    main_person = str(name_list[0]).strip()
    other_person = name_list[1] if len(name_list) > 1 else None
    name_alignment, text_names = LIMP._build_name_alignment(name_list, text, episode_description)
    context_sources.append("name_alignment:limp_build_name_alignment")

    info = {}
    for name in name_list:
        source_name = name_alignment.get(name, name) if isinstance(name_alignment, dict) else name
        info[name] = LIMP.text_parsing.parse_text_info(text, source_name)
    context_sources.append("info:text_parsing.parse_text_info")

    visual_action_result = LIMP._load_visual_action_result(episode_id)
    if isinstance(visual_action_result, dict):
        context_sources.append("visual_action_result:actions_extracted")
    action_target_name = None
    if episode_id > 4000:
        action_target_name = name_list[1] if len(name_list) > 1 else main_person
    else:
        if not person_has_actions(info[main_person]):
            action_target_name = main_person
        elif len(name_list) > 1:
            action_target_name = name_list[1]

    if action_target_name is not None and action_target_name in info and not person_has_actions(info[action_target_name]):
        set_person_actions(
            info[action_target_name],
            LIMP.visual_action_extraction.get_action(
                episode_id,
                person_name=name_alignment.get(action_target_name, action_target_name),
                additional_information=text,
            ),
        )
        context_sources.append("actions:visual_action_extraction.get_action")

    question_label = LIMP._infer_question_label_from_prompt(prompt)
    context_sources.append("question_label:prompt_heuristic")
    _, option_texts = LIMP._parse_question_options(prompt)
    visual_summary = LIMP._build_visual_summary(visual_action_result)
    if visual_summary:
        context_sources.append("visual_summary:build_visual_summary")

    return {
        "episode_description": episode_description,
        "name_list": name_list,
        "text_names": text_names,
        "name_alignment": name_alignment,
        "main_person": main_person,
        "other_person": other_person,
        "info": info,
        "visual_action_result": visual_action_result,
        "visual_summary": visual_summary,
        "question_label": question_label,
        "option_texts": option_texts,
        "context_sources": context_sources,
    }


def collect_evidence(
    episode_id: int,
    question_id: str,
    prompt: str,
    questions: dict[str, Any],
    text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = build_question_context(
        episode_id,
        question_id,
        prompt,
        questions,
        text,
    )
    evidence = LIMP._collect_rule_evidence(
        context["question_label"],
        prompt,
        context["info"],
        context["main_person"],
        context["other_person"],
        context["option_texts"],
        description=context["episode_description"],
        visual_summary=context["visual_summary"],
    )
    if not isinstance(evidence, dict):
        evidence = {
            "method": "model_question_evidence",
            "question_label": context["question_label"],
            "evidence_source": "model_allowed_inputs",
            "evidence_types": ["empty_model_evidence"],
            "evidence_strength": "none",
            "confidence": 0.0,
            "option_support": {},
            "target_choice": None,
            "inferred_goal": None,
            "reason": "Evidence extractor returned no evidence.",
            "key_facts": [],
            "counter_evidence": [],
        }

    evidence = dict(evidence)
    passed, reason = LIMP._evidence_passes_prior_filter(evidence, context["question_label"])
    evidence["prior_filter"] = {
        "passed": passed,
        "reason": reason,
        "min_confidence": LIMP._EVIDENCE_PRIOR_MIN_CONFIDENCE,
        "allowed_strengths": sorted(LIMP._EVIDENCE_PRIOR_STRENGTHS),
    }
    return evidence, context


def skipped_record(prompt: str, context: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": prompt,
        "question_label": context["question_label"],
        "main_person": context["main_person"],
        "other_person": context["other_person"],
        "prior_filter": evidence.get("prior_filter", {}),
        "model_evidence": evidence,
    }


def configure_limp_filters(min_confidence: float, strengths: set[str]) -> None:
    LIMP._EVIDENCE_PRIOR_MIN_CONFIDENCE = min_confidence
    LIMP._EVIDENCE_PRIOR_STRENGTHS = strengths


def count_questions(question_data: dict[str, Any], episode_ids: list[int]) -> int:
    total = 0
    for episode_id in episode_ids:
        episode_questions = question_data.get(str(episode_id), {}).get("questions", {})
        if isinstance(episode_questions, dict):
            total += len(episode_questions)
    return total


def progress_postfix(output: dict[str, Any]) -> dict[str, int]:
    summary = output.get("summary", {})
    return {
        "strong": int(summary.get("strong_evidence", 0) or 0),
        "skipped": int(summary.get("skipped", 0) or 0),
        "errors": int(summary.get("errors", 0) or 0),
    }


def update_checkpoint(
    output: dict[str, Any],
    status: str,
    episode_id: int | None = None,
    question_id: str | None = None,
    processed: int = 0,
    total: int = 0,
    detail: dict[str, Any] | None = None,
) -> None:
    checkpoint: dict[str, Any] = {
        "status": status,
        "processed_new_questions": processed,
        "total_scheduled_questions": total,
    }
    if episode_id is not None:
        checkpoint["episode_id"] = episode_id
    if question_id is not None:
        checkpoint["question_id"] = question_id
    if detail:
        checkpoint.update(detail)
    output["checkpoint"] = checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cached model-question evidence for MuMA-ToM inference."
    )
    parser.add_argument("--output", required=True, help="Path to write the cached evidence JSON.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE), help="Questions JSON path.")
    parser.add_argument("--texts", default=str(DEFAULT_TEXTS_FILE), help="Texts JSON path.")
    parser.add_argument(
        "--episodes",
        default=os.getenv("MUMATOM_EPISODES", ""),
        help="Comma-separated episode ids, ranges like 1-10, or a Python list. Defaults to all.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of questions to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate questions already present in output.")
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Also save weak/failed evidence under questions for debugging; inference still filters it.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=LIMP._EVIDENCE_PRIOR_MIN_CONFIDENCE,
        help="Minimum confidence required for evidence to be stored as strong evidence.",
    )
    parser.add_argument(
        "--strengths",
        default=",".join(sorted(LIMP._EVIDENCE_PRIOR_STRENGTHS)) or "strong",
        help="Comma-separated evidence strengths accepted as strong evidence.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Raise instead of recording per-question errors.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = Path(args.output)
    questions_file = Path(args.questions)
    texts_file = Path(args.texts)
    strengths = parse_strengths(args.strengths)
    configure_limp_filters(args.min_confidence, strengths)

    question_data = load_json(questions_file)
    text_data = load_json(texts_file)
    all_episode_ids = sorted(int(key) for key in question_data.keys())
    episode_ids = parse_episode_list(args.episodes, all_episode_ids)

    output = load_output(output_path)
    output["config"] = {
        "questions_file": str(questions_file),
        "texts_file": str(texts_file),
        "context_source": "generated_from_current_limp_flow",
        "min_confidence": args.min_confidence,
        "allowed_strengths": sorted(strengths),
        "save_all": args.save_all,
        "evidence_source": "model_allowed_inputs",
    }

    processed = 0
    total_questions = count_questions(question_data, episode_ids)
    refresh_summary(output)
    update_checkpoint(output, "started", processed=processed, total=total_questions)
    save_json(output_path, output)
    with tqdm(total=total_questions, desc="Generating model evidence", unit="question") as progress:
        for episode_id in episode_ids:
            episode_key = str(episode_id)
            questions = question_data[episode_key]
            text = text_data[episode_key]
            question_prompts = questions["questions"]
            episode_record = ensure_episode_record(output, episode_id)
            episode_record["description"] = LIMP._clean_description(questions.get("description", ""))
            episode_record["total_questions"] = len(question_prompts)
            refresh_summary(output)
            update_checkpoint(output, "episode_started", episode_id=episode_id, processed=processed, total=total_questions)
            save_json(output_path, output)

            for question_id, prompt in question_prompts.items():
                if args.limit and processed >= args.limit:
                    refresh_summary(output)
                    update_checkpoint(output, "stopped_by_limit", processed=processed, total=total_questions)
                    save_json(output_path, output)
                    progress.set_postfix(progress_postfix(output))
                    tqdm.write(f"Stopped after --limit={args.limit}.")
                    return

                question_key = str(question_id)
                progress.set_description(f"Episode {episode_id} question {question_key}")
                if args.overwrite:
                    clear_question(episode_record, question_key)
                elif question_done(episode_record, question_key):
                    update_checkpoint(
                        output,
                        "skip_existing",
                        episode_id=episode_id,
                        question_id=question_key,
                        processed=processed,
                        total=total_questions,
                    )
                    save_json(output_path, output)
                    tqdm.write(f"[skip-existing] episode={episode_id} question={question_key}")
                    progress.update(1)
                    continue

                try:
                    update_checkpoint(
                        output,
                        "running",
                        episode_id=episode_id,
                        question_id=question_key,
                        processed=processed,
                        total=total_questions,
                    )
                    save_json(output_path, output)
                    evidence, context = collect_evidence(episode_id, question_key, prompt, questions, text)
                    for key in (
                        "name_list",
                        "text_names",
                        "name_alignment",
                        "visual_summary",
                        "visual_action_result",
                        "info",
                    ):
                        if key == "name_list":
                            episode_record["names"] = context[key]
                        else:
                            episode_record[key] = context[key]
                    question_record = {
                        "question": prompt,
                        "question_label": context["question_label"],
                        "main_person": context["main_person"],
                        "other_person": context["other_person"],
                        "context_sources": context["context_sources"],
                        "model_evidence": evidence,
                        "passes_prior_filter": bool(evidence.get("prior_filter", {}).get("passed")),
                    }
                    if evidence.get("prior_filter", {}).get("passed"):
                        episode_record["evidence"][question_key] = evidence
                        episode_record["skipped"].pop(question_key, None)
                        status = "strong"
                    else:
                        episode_record["skipped"][question_key] = skipped_record(prompt, context, evidence)
                        episode_record["evidence"].pop(question_key, None)
                        status = "skipped"
                    if args.save_all:
                        episode_record["questions"][question_key] = question_record
                    else:
                        episode_record["questions"].pop(question_key, None)
                    episode_record["errors"].pop(question_key, None)
                    filter_info = evidence.get("prior_filter", {})
                    tqdm.write(
                        "[{}] episode={} question={} label={} strength={} confidence={:.2f} reason={}".format(
                            status,
                            episode_id,
                            question_key,
                            context["question_label"],
                            evidence.get("evidence_strength"),
                            LIMP._confidence_to_score(evidence.get("confidence", 0.0)),
                            filter_info.get("reason"),
                        )
                    )
                except Exception as exc:
                    if args.stop_on_error:
                        raise
                    episode_record["errors"][question_key] = {
                        "question": prompt,
                        "error": str(exc),
                    }
                    status = "error"
                    tqdm.write(f"[error] episode={episode_id} question={question_key} error={exc}")
                processed += 1
                refresh_summary(output)
                update_checkpoint(
                    output,
                    status,
                    episode_id=episode_id,
                    question_id=question_key,
                    processed=processed,
                    total=total_questions,
                    detail=progress_postfix(output),
                )
                save_json(output_path, output)
                progress.set_postfix(progress_postfix(output))
                progress.update(1)

    refresh_summary(output)
    update_checkpoint(output, "completed", processed=processed, total=total_questions, detail=progress_postfix(output))
    save_json(output_path, output)
    print("Saved model evidence to", output_path)
    print("Summary:", json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

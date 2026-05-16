from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from local_backend import LocalChatModel
from runtime_hparams import get_enable_thinking_default
from visual_action_extraction import (
    extract_visual_action_bundle,
    format_visual_action_bundle,
    load_prompt_entries,
    resolve_video_path,
)


NAME_STOPWORDS = {
    "A",
    "After",
    "Afterward",
    "B",
    "C",
    "D",
    "E",
    "Given",
    "He",
    "Her",
    "His",
    "If",
    "Later",
    "LEAST",
    "MOST",
    "She",
    "The",
    "Then",
    "There",
    "They",
    "When",
}


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def normalize_list(items: list[str] | None) -> list[str]:
    if not items:
        return []
    normalized = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def extract_names_from_text(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+\b", text)
    names: list[str] = []
    for candidate in candidates:
        if candidate in NAME_STOPWORDS:
            continue
        if candidate not in names:
            names.append(candidate)
    return names


def extract_names_with_model(model: LocalChatModel, text: str) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract the distinct person names that appear in the narrative. "
                "Return JSON only in the form {\"names\": [\"name1\", \"name2\"]}."
            ),
        },
        {"role": "user", "content": text},
    ]
    data = model.generate_json(messages, max_new_tokens=128)
    names = data.get("names", []) if isinstance(data, dict) else []
    return normalize_list(names)


def ordered_names_for_question(question: str, names: list[str]) -> list[str]:
    return sorted(
        names,
        key=lambda name: (-len(re.findall(rf"\b{re.escape(name)}\b", question)), names.index(name)),
    )


def extract_person_info(model: LocalChatModel, text: str, name: str) -> dict[str, list[str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction model. Return valid JSON only.\n"
                "Focus on exactly one person in a multi-person narrative.\n"
                "Return JSON with this schema:\n"
                "{\n"
                '  "actions": ["..."],\n'
                '  "utterances": ["..."]\n'
                "}\n"
                "Rules:\n"
                "- Only include actions directly performed by the target person.\n"
                "- Only include utterances directly spoken by the target person.\n"
                "- Keep actions chronological, short, and explicit about object/location when possible.\n"
                "- Use [] when nothing is present."
            ),
        },
        {
            "role": "user",
            "content": f"Target person: {name}\n\nNarrative:\n{text}",
        },
    ]
    data = model.generate_json(messages, max_new_tokens=384)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for person info, got: {data}")
    return {
        "actions": normalize_list(data.get("actions")),
        "utterances": normalize_list(data.get("utterances")),
    }


def infer_initial_state(
    model: LocalChatModel,
    info: dict[str, dict[str, list[str]]],
    visual_summary: str | None = None,
) -> list[str]:
    action_lines: list[str] = []
    for name, person_info in info.items():
        if not person_info["actions"]:
            continue
        action_lines.append(f"{name}'s actions:")
        for index, action in enumerate(person_info["actions"], start=1):
            action_lines.append(f"{index}. {action}")

    user_content = (
        "Infer the environment state before anyone acted.\n"
        "Only include states that are strongly supported by the action history.\n"
        "Return JSON with schema {\"states\": [\"...\"]}.\n\n"
        + "\n".join(action_lines)
    )
    if visual_summary:
        user_content += f"\n\nOptional visual summary:\n{visual_summary}"

    messages = [
        {
            "role": "system",
            "content": (
                "You infer initial states from action descriptions. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    data = model.generate_json(messages, max_new_tokens=256)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for initial state, got: {data}")
    return normalize_list(data.get("states"))


def extract_latent_variables(
    model: LocalChatModel,
    question: str,
    main_person: str,
    other_person: str,
    init_state: list[str],
    info: dict[str, dict[str, list[str]]],
    visual_summary: str | None = None,
) -> dict[str, dict[str, str]]:
    initial_state_text = "\n".join(f"- {state}" for state in init_state) or "- Unknown"
    main_actions = "\n".join(f"- {item}" for item in info[main_person]["actions"]) or "- None"
    other_actions = "\n".join(f"- {item}" for item in info[other_person]["actions"]) or "- None"

    user_content = (
        f"Target person: {main_person}\n"
        f"Other person: {other_person}\n\n"
        "Initial state:\n"
        f"{initial_state_text}\n\n"
        f"{other_person}'s observed actions:\n{other_actions}\n\n"
        f"{main_person}'s observed actions:\n{main_actions}\n\n"
        "For each option in the question, extract:\n"
        "- belief: the target person's belief about the environment\n"
        "- social_goal: one of help, hinder, independent, unknown\n"
        f"- believed_goal: what {main_person} believes {other_person}'s physical goal is\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "A": {"belief": "...", "social_goal": "...", "believed_goal": "..."},\n'
        '  "B": {"belief": "...", "social_goal": "...", "believed_goal": "..."},\n'
        '  "C": {"belief": "...", "social_goal": "...", "believed_goal": "..."}\n'
        "}\n\n"
        f"Question:\n{question}"
    )
    if visual_summary:
        user_content += f"\n\nOptional visual summary:\n{visual_summary}"

    messages = [
        {
            "role": "system",
            "content": "You extract structured latent variables from multiple-choice questions. Return JSON only.",
        },
        {"role": "user", "content": user_content},
    ]
    data = model.generate_json(messages, max_new_tokens=384)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for latent variables, got: {data}")

    result: dict[str, dict[str, str]] = {}
    for label in ("A", "B", "C"):
        option = data.get(label, {}) if isinstance(data, dict) else {}
        result[label] = {
            "belief": str(option.get("belief", "")).strip(),
            "social_goal": str(option.get("social_goal", "unknown")).strip().lower(),
            "believed_goal": str(option.get("believed_goal", "")).strip(),
        }
    return result


def score_binary_likely(model: LocalChatModel, prompt: str) -> float:
    messages = [
        {
            "role": "system",
            "content": (
                "Decide which option is more plausible. "
                "Answer with a single letter only."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    probabilities = model.score_candidates(messages, ["A", "B"])
    return probabilities["A"]


def compute_prob_utterance(
    model: LocalChatModel,
    main_person: str,
    other_person: str,
    other_utterance: str,
    main_utterance: str,
    latent_vars: dict[str, str],
    init_state: list[str],
    visual_summary: str | None = None,
) -> float:
    initial_state_text = "\n".join(f"- {state}" for state in init_state) or "- Unknown"
    prompt = (
        f"{main_person}'s social goal: {latent_vars['social_goal']}\n"
        f"{main_person}'s belief: {latent_vars['belief']}\n"
        f"{main_person}'s belief about {other_person}'s goal: {latent_vars['believed_goal']}\n"
        f"{other_person}'s utterance: {other_utterance}\n"
        f"Initial state:\n{initial_state_text}\n"
    )
    if visual_summary:
        prompt += f"Visual summary:\n{visual_summary}\n"
    prompt += (
        f"\nIs it likely that {main_person} would say the utterance below under these conditions?\n"
        f"{main_person}'s utterance: {main_utterance}\n\n"
        "A) Likely\n"
        "B) Unlikely\n"
        "Answer:\n"
    )
    return score_binary_likely(model, prompt)


def compute_prob_action(
    model: LocalChatModel,
    main_person: str,
    other_person: str,
    init_state: list[str],
    previous_actions: str,
    current_action: str,
    latent_vars: dict[str, str],
    visual_summary: str | None = None,
) -> float:
    initial_state_text = "\n".join(f"- {state}" for state in init_state) or "- Unknown"
    prompt = (
        f"{main_person}'s social goal: {latent_vars['social_goal']}\n"
        f"{main_person}'s belief: {latent_vars['belief']}\n"
        f"{main_person}'s belief about {other_person}'s goal: {latent_vars['believed_goal']}\n"
        f"Initial state:\n{initial_state_text}\n"
    )
    if visual_summary:
        prompt += f"Visual summary:\n{visual_summary}\n"
    prompt += (
        f"\nUse {other_person}'s actions to infer object locations before {main_person} acts.\n"
        f"Previous actions:\n{previous_actions}\n\n"
        f"Candidate action by {main_person}: {current_action}\n\n"
        "When the social goal is 'help', actions that move objects toward the believed goal are more likely.\n"
        "When the social goal is 'hinder', actions that disrupt the believed goal are more likely.\n"
        "Unrelated walking or neutral manipulation can still be likely.\n\n"
        "A) Likely\n"
        "B) Unlikely\n"
        "Answer:\n"
    )
    return score_binary_likely(model, prompt)


def compute_statement_probability(
    model: LocalChatModel,
    init_state: list[str],
    latent_vars: dict[str, str],
    info: dict[str, dict[str, list[str]]],
    main_person: str,
    visual_summary: str | None = None,
) -> float:
    names = list(info.keys())
    other_person = next(name for name in names if name != main_person)
    probability = 1.0

    main_utterances = info[main_person]["utterances"]
    other_utterances = info[other_person]["utterances"]
    if main_utterances:
        reference_utterance = other_utterances[0] if other_utterances else ""
        probability *= compute_prob_utterance(
            model=model,
            main_person=main_person,
            other_person=other_person,
            other_utterance=reference_utterance,
            main_utterance=main_utterances[0],
            latent_vars=latent_vars,
            init_state=init_state,
            visual_summary=visual_summary,
        )

    main_actions = info[main_person]["actions"]
    other_actions = info[other_person]["actions"]
    if main_actions:
        for index, action in enumerate(main_actions):
            previous_lines = [f"{other_person}'s actions:"]
            if other_actions:
                previous_lines.extend(other_actions)
            else:
                previous_lines.append("None")
            previous_lines.append(f"{main_person}'s previous actions:")
            if index:
                previous_lines.extend(main_actions[:index])
            else:
                previous_lines.append("None")
            probability *= compute_prob_action(
                model=model,
                main_person=main_person,
                other_person=other_person,
                init_state=init_state,
                previous_actions="\n".join(previous_lines),
                current_action=action,
                latent_vars=latent_vars,
                visual_summary=visual_summary,
            )
    return probability


def choose_answer(question: str, option_scores: dict[str, float]) -> str:
    if "LEAST LIKELY" in question.upper():
        return min(option_scores.items(), key=lambda item: item[1])[0]
    return max(option_scores.items(), key=lambda item: item[1])[0]


def summarize_visual_data(visual_data_dir: Path, episode_id: int, max_states: int = 40) -> str | None:
    episode_dir = visual_data_dir / str(episode_id)
    required_files = ["closeness.json", "hold.json", "inside.json", "on.json", "opened.json"]
    if not all((episode_dir / filename).exists() for filename in required_files):
        return None

    closeness = load_json(episode_dir / "closeness.json")
    hold = load_json(episode_dir / "hold.json")
    inside = load_json(episode_dir / "inside.json")
    on_data = load_json(episode_dir / "on.json")
    opened = load_json(episode_dir / "opened.json")

    episode_key = str(episode_id)
    frame_keys = sorted(on_data[episode_key].keys(), key=lambda value: int(value))

    summaries: list[str] = []
    previous_signature: tuple[str, ...] | None = None
    for frame_key in frame_keys:
        statements: list[str] = []
        for obj in closeness[episode_key].get(frame_key, [])[:3]:
            statements.append(f"The active character is close to {obj}.")
        for statement in on_data[episode_key].get(frame_key, []):
            if statement:
                statements.append(f"{statement}.")
        for statement in inside[episode_key].get(frame_key, []):
            if statement:
                statements.append(f"{statement}.")
        for statement in hold[episode_key].get(frame_key, []):
            if statement:
                statements.append(f"{statement}.")
        for obj in opened[episode_key].get(frame_key, []):
            statements.append(f"{obj} is opened.")

        signature = tuple(sorted(statements))
        if not statements or signature == previous_signature:
            continue

        summaries.append(f"Frame {frame_key}: {' '.join(statements)}")
        previous_signature = signature
        if len(summaries) >= max_states:
            break

    if not summaries:
        return None
    return "\n".join(summaries)


def combine_visual_summaries(*summaries: str | None) -> str | None:
    parts = [summary.strip() for summary in summaries if summary and summary.strip()]
    return "\n\n".join(parts) if parts else None


def select_episode_ids(
    all_episode_ids: list[int],
    requested_ids: list[int] | None,
    max_episodes: int | None,
) -> list[int]:
    if requested_ids:
        selected = [episode_id for episode_id in requested_ids if episode_id in all_episode_ids]
    else:
        selected = list(all_episode_ids)
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected


def run_episode(
    model: LocalChatModel,
    episode_id: int,
    text: str,
    question_blob: dict[str, Any],
    visual_summary: str | None,
) -> dict[str, Any]:
    names = extract_names_from_text(text)
    if len(names) < 2:
        names = extract_names_with_model(model, text)
    if len(names) < 2:
        raise ValueError(f"Failed to identify two names in episode {episode_id}: {names}")

    info = {name: extract_person_info(model, text, name) for name in names}
    init_state = infer_initial_state(model, info, visual_summary=visual_summary)

    predictions: dict[str, Any] = {}
    correct = 0
    total = 0
    for question_id, question in question_blob["questions"].items():
        ordered_names = ordered_names_for_question(question, names)
        if len(ordered_names) < 2:
            ordered_names = list(names)
        main_person, other_person = ordered_names[:2]
        ordered_info = {
            main_person: info[main_person],
            other_person: info[other_person],
        }

        latents = extract_latent_variables(
            model=model,
            question=question,
            main_person=main_person,
            other_person=other_person,
            init_state=init_state,
            info=ordered_info,
            visual_summary=visual_summary,
        )

        option_scores = {
            label: compute_statement_probability(
                model=model,
                init_state=init_state,
                latent_vars=latents[label],
                info=ordered_info,
                main_person=main_person,
                visual_summary=visual_summary,
            )
            for label in ("A", "B", "C")
        }
        prediction = choose_answer(question, option_scores)
        gold_text = str(question_blob["answers"][question_id]).strip()
        gold_label = gold_text[0]

        predictions[question_id] = {
            "question": question,
            "main_person": main_person,
            "other_person": other_person,
            "latent_variables": latents,
            "option_scores": option_scores,
            "prediction": prediction,
            "gold": gold_label,
            "correct": prediction == gold_label,
        }
        correct += int(prediction == gold_label)
        total += 1

    return {
        "episode_id": episode_id,
        "names": names,
        "text": text,
        "visual_summary": visual_summary,
        "info": info,
        "init_state": init_state,
        "predictions": predictions,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MuMA-ToM benchmark with local Qwen3.5-4B.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/LPP/cvpr/muti_agent/MUMA-TOM-BENCHMARK"),
        help="Path to the benchmark directory containing questions.json and texts.json.",
    )
    parser.add_argument(
        "--visual-data-dir",
        type=Path,
        default=None,
        help="Optional path to visual_data/<episode_id> directories.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Optional path to benchmark videos. If omitted, <dataset-dir>/videos is used when available.",
    )
    parser.add_argument(
        "--visual-prompt-file",
        type=Path,
        default=Path("/data/LPP/cvpr/muti_agent/MuMA-ToM/Files/actions_extracted.json"),
        help="Original MuMA-ToM prompt file used to guide local video action extraction.",
    )
    parser.add_argument(
        "--visual-action-cache",
        type=Path,
        default=None,
        help="Optional cache file for Qwen3.5 video action summaries. Defaults to <output-dir>/visual_actions.json.",
    )
    parser.add_argument(
        "--video-frame-count",
        type=int,
        default=8,
        help="How many frames to sample uniformly from each episode video.",
    )
    parser.add_argument(
        "--skip-video-extraction",
        action="store_true",
        help="Disable the Qwen3.5 video extraction stage even if videos are available.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="ModelScope model id.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Existing local model directory. If omitted, ModelScope snapshot_download is used.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/data/.cache/modelscope",
        help="Cache directory used by ModelScope when downloading the model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision for ModelScope download.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for loading the model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/LPP/cvpr/muti_agent/MuMA-ToM/local_runs/qwen3.5-4b"),
        help="Directory to write results.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode ids to run.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Run only the first N selected episodes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results.json file in the output directory.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    questions = load_json(args.dataset_dir / "questions.json")
    texts = load_json(args.dataset_dir / "texts.json")
    all_episode_ids = sorted(int(key) for key in questions.keys())
    selected_episode_ids = select_episode_ids(all_episode_ids, args.episodes, args.max_episodes)

    videos_dir = None
    if not args.skip_video_extraction:
        videos_dir = args.videos_dir
        if videos_dir is None:
            auto_videos_dir = args.dataset_dir / "videos"
            if auto_videos_dir.exists():
                videos_dir = auto_videos_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    visual_action_cache_path = args.visual_action_cache or (args.output_dir / "visual_actions.json")
    run_config = {
        "dataset_dir": str(args.dataset_dir),
        "visual_data_dir": str(args.visual_data_dir) if args.visual_data_dir else None,
        "videos_dir": str(videos_dir) if videos_dir else None,
        "visual_prompt_file": str(args.visual_prompt_file) if args.visual_prompt_file else None,
        "visual_action_cache": str(visual_action_cache_path) if videos_dir else None,
        "video_frame_count": args.video_frame_count,
        "model_id": args.model_id,
        "model_dir": args.model_dir,
        "model_cache_dir": args.model_cache_dir,
        "revision": args.revision,
        "torch_dtype": args.torch_dtype,
    }
    results = {
        "config": run_config,
        "episodes": {},
    }
    if args.resume and results_path.exists():
        results = load_json(results_path)
        results["config"] = {**results.get("config", {}), **run_config}

    visual_prompt_entries = load_prompt_entries(args.visual_prompt_file)
    visual_action_cache = {}
    if videos_dir and visual_action_cache_path.exists():
        visual_action_cache = load_json(visual_action_cache_path)

    model = LocalChatModel(
        model_id=args.model_id,
        model_dir=args.model_dir,
        cache_dir=args.model_cache_dir,
        revision=args.revision,
        torch_dtype=args.torch_dtype,
        enable_thinking=get_enable_thinking_default(),
    )

    for episode_id in selected_episode_ids:
        episode_key = str(episode_id)
        if args.resume and episode_key in results.get("episodes", {}):
            print(f"Skip episode {episode_id}: already finished.")
            continue

        print(f"Running episode {episode_id}")
        visual_scene_graph_summary = None
        if args.visual_data_dir:
            visual_scene_graph_summary = summarize_visual_data(args.visual_data_dir, episode_id)

        visual_action_result = None
        visual_action_summary = None
        if videos_dir:
            cached_visual_action = visual_action_cache.get(episode_key)
            if isinstance(cached_visual_action, dict):
                visual_action_result = cached_visual_action
            else:
                video_path = resolve_video_path(videos_dir, episode_id)
                if video_path.exists():
                    print(f"  Extract visual action summary from {video_path.name}")
                    visual_action_result = extract_visual_action_bundle(
                        model=model,
                        episode_id=episode_id,
                        video_path=video_path,
                        prompt_entry=visual_prompt_entries.get(episode_key)
                        if isinstance(visual_prompt_entries.get(episode_key), dict)
                        else None,
                        num_frames=args.video_frame_count,
                    )
                    visual_action_cache[episode_key] = visual_action_result
                    save_json(visual_action_cache_path, visual_action_cache)
                else:
                    print(f"  No video found for episode {episode_id}: {video_path}")
            visual_action_summary = format_visual_action_bundle(visual_action_result)

        visual_summary = combine_visual_summaries(
            visual_scene_graph_summary,
            visual_action_summary,
        )

        episode_result = run_episode(
            model=model,
            episode_id=episode_id,
            text=texts[episode_key],
            question_blob=questions[episode_key],
            visual_summary=visual_summary,
        )
        if visual_action_result:
            episode_result["visual_action_result"] = visual_action_result
        results.setdefault("episodes", {})[episode_key] = episode_result
        save_json(results_path, results)

        print(
            f"Episode {episode_id} accuracy: "
            f"{episode_result['correct']}/{episode_result['total']} "
            f"({episode_result['accuracy']:.3f})"
        )

    total_correct = 0
    total_questions = 0
    for episode in results.get("episodes", {}).values():
        total_correct += int(episode["correct"])
        total_questions += int(episode["total"])

    summary = {
        "episodes_finished": len(results.get("episodes", {})),
        "total_correct": total_correct,
        "total_questions": total_questions,
        "accuracy": total_correct / total_questions if total_questions else 0.0,
    }
    results["summary"] = summary
    save_json(results_path, results)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

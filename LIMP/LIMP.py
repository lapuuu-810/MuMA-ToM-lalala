import ast
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path

import compute_prob_GPT
import scipy.special
import text_parsing
import visual_action_extraction
from model_backend import get_backend_name, get_default_model
from runtime_hparams import (
    get_enable_thinking_default,
    get_local_model_defaults,
    get_model_evidence_file_default,
    get_qwen_api_defaults,
    get_results_file_default,
)
from tqdm import tqdm


_OPTION_PATTERN = re.compile(r"^\s*([A-C])\)\s*(.+?)(?=^\s*[A-C]\)\s*|\Z)", re.MULTILINE | re.DOTALL)
_TIE_REL_TOL = 1e-9
_TIE_ABS_TOL = 1e-12
_MODEL_EVIDENCE_CONFIDENCE_THRESHOLD = float(os.getenv("MUMATOM_MODEL_EVIDENCE_CONFIDENCE_THRESHOLD", "0.55"))
_MODEL_EVIDENCE_MAX_TOKENS = int(os.getenv("MUMATOM_MODEL_EVIDENCE_MAX_TOKENS", "384"))
_MODEL_EVIDENCE_FILE = os.getenv("MUMATOM_MODEL_EVIDENCE_FILE", get_model_evidence_file_default()).strip()
_USE_MODEL_EVIDENCE_FILE = os.getenv("MUMATOM_USE_MODEL_EVIDENCE_FILE", "1") == "1"
_EVIDENCE_PRIOR_MIN_CONFIDENCE = float(os.getenv("MUMATOM_EVIDENCE_PRIOR_MIN_CONFIDENCE", "0.70"))
_EVIDENCE_PRIOR_STRENGTHS = {
    item.strip().lower()
    for item in os.getenv("MUMATOM_EVIDENCE_PRIOR_STRENGTHS", "strong").split(",")
    if item.strip()
}
_ENABLE_BELIEF_OF_GOAL_EVIDENCE_PRIOR = False
_QUESTION_LABELS = {"belief", "social_goal", "belief_of_goal"}
_SOCIAL_GOAL_LABELS = ("help", "hinder", "indifferent")
_NAME_STOPWORDS = {
    "A",
    "After",
    "Afterward",
    "Any",
    "As",
    "B",
    "C",
    "D",
    "E",
    "Given",
    "He",
    "Her",
    "His",
    "If",
    "Input",
    "Later",
    "LEAST",
    "MOST",
    "Meanwhile",
    "She",
    "The",
    "Then",
    "There",
    "They",
    "While",
    "When",
}


def extract_name_from_question(question):
    prompt = """You will read a question asking about a person's mental state or actions. From the prompt and options, extract any name of the people you encountered. Determine the person whose mental state or action the question is asking about. Produce your output in this form: [main person's name, name2, name3, ...]. Do not record names appearing multiple times, and do not give any extra information. An example question is like this:
    Example Question: Given that Emma has seen David walking to school yesterday, what will Emma most likely believe
    A David will walk to school tomorrow
    B David will drive to school tomorrow
    C David will not come to school tomorrow
    Example Output: ["Emma", "David"]

    Input Question: {}
    """

    temp_str = get_default_model().chat(
        [
            {"role": "system", "content": prompt.format(question)},
        ],
        max_new_tokens=128,
        temperature=0.0,
        enable_thinking=get_enable_thinking_default(),
    ).strip()
    name_list = ast.literal_eval(temp_str)
    return name_list


def _clean_description(description):
    text = str(description or "").strip()
    if text.startswith(("b'", 'b"')):
        try:
            value = ast.literal_eval(text)
        except Exception:
            return text
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    return text


def _normalize_inline_text(text):
    return " ".join(str(text or "").split())


def _extract_names_from_text(text):
    names = []
    counts = {}
    for candidate in re.findall(r"\b[A-Z][a-z]+\b", str(text or "")):
        if candidate in _NAME_STOPWORDS:
            continue
        counts[candidate] = counts.get(candidate, 0) + 1
        if candidate not in names:
            names.append(candidate)

    repeated_names = [candidate for candidate in names if counts.get(candidate, 0) > 1]
    if len(repeated_names) >= 2:
        return repeated_names
    return names


def _ordered_question_names(name_list, description):
    description_names = _extract_names_from_text(description)
    ordered = [name for name in description_names if name in name_list]
    for name in name_list:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _build_name_alignment(name_list, text, description):
    text_names = _extract_names_from_text(text)
    if len(text_names) < 2 or not name_list:
        return {}, text_names
    if all(name in text_names for name in name_list):
        return {name: name for name in name_list}, text_names

    question_order = _ordered_question_names(name_list, description)
    alignment = {}
    for question_name, text_name in zip(question_order, text_names):
        alignment[question_name] = text_name
    for name in name_list:
        alignment.setdefault(name, name)
    return alignment, text_names


def _parse_question_options(question):
    matches = list(_OPTION_PATTERN.finditer(str(question)))
    options = {match.group(1): _normalize_inline_text(match.group(2)) for match in matches}
    stem = question[: matches[0].start()].strip() if matches else str(question).strip()
    return _normalize_inline_text(stem), options


def _infer_question_label_from_prompt(question):
    stem, options = _parse_question_options(question)
    option_blob = " ".join(options.values()).lower()
    stem_lower = stem.lower()
    if (
        "based on the actions of the agents" in stem_lower
        or "wants to place" in option_blob
        or "doesn't know" in option_blob
        or "does not know" in option_blob
        or "without thinking about what" in option_blob
    ):
        return "belief_of_goal"
    if (
        "believed that there was" in option_blob
        or "believed that there is" in option_blob
        or "believed that there were" in option_blob
    ):
        return "belief"
    if (
        "has been trying to" in option_blob
        or "was indifferent" in option_blob
        or "trying to help" in option_blob
        or "trying to prevent" in option_blob
        or " to hinder " in f" {option_blob} "
    ):
        return "social_goal"
    return None


def _person_actions(person_info):
    actions = person_info.get("actions")
    if actions is None:
        actions = person_info.get("action")
    return [str(item).strip() for item in (actions or []) if str(item).strip()]


def _person_utterances(person_info):
    utterances = person_info.get("utterances")
    if utterances is None:
        utterances = person_info.get("utterance")
    return [str(item).strip() for item in (utterances or []) if str(item).strip()]


def _confidence_to_score(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(1.0, max(0.0, float(value)))
    normalized = _normalize_inline_text(value).lower()
    if normalized in {"high", "strong", "confident"}:
        return 0.85
    if normalized in {"medium", "moderate"}:
        return 0.62
    if normalized in {"low", "weak", "uncertain"}:
        return 0.35
    try:
        return min(1.0, max(0.0, float(normalized)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_evidence_types(raw_types):
    if raw_types is None:
        return []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    if not isinstance(raw_types, list):
        return []
    normalized = []
    for raw_type in raw_types:
        value = re.sub(r"[^a-z0-9]+", "_", str(raw_type).strip().lower()).strip("_")
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _compact_person_block(name, person_info):
    actions = _person_actions(person_info)
    utterances = _person_utterances(person_info)
    return (
        f"{name}'s utterances:\n"
        + ("\n".join(f"- {item}" for item in utterances) if utterances else "- None")
        + "\n\n"
        + f"{name}'s actions:\n"
        + ("\n".join(f"- {item}" for item in actions) if actions else "- None")
    )


def _social_goal_evidence_from_rule_evidence(rule_evidence):
    if not rule_evidence or rule_evidence.get("method") not in {"social_goal_evidence", "model_question_evidence"}:
        return None
    if rule_evidence.get("method") == "model_question_evidence":
        passed_filter, _ = _evidence_passes_prior_filter(rule_evidence, "social_goal")
        if not passed_filter:
            return None
    return {
        "goal": rule_evidence.get("inferred_goal") or rule_evidence.get("goal"),
        "reason": rule_evidence.get("evidence_reason") or rule_evidence.get("reason"),
        "evidence_source": rule_evidence.get("evidence_source", "model_allowed_inputs"),
        "evidence_types": rule_evidence.get("evidence_types", []),
        "evidence_strength": rule_evidence.get("evidence_strength"),
        "confidence": rule_evidence.get("confidence"),
        "target_object": rule_evidence.get("target_object"),
        "target_location": rule_evidence.get("target_location"),
        "actual_grab_location": rule_evidence.get("actual_grab_location"),
    }


def _truncate_text(text, limit):
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _normalize_choice_support(raw_support, choices):
    if not isinstance(raw_support, dict):
        return {}
    normalized = {}
    for choice in choices:
        if choice in raw_support:
            normalized[choice] = _confidence_to_score(raw_support.get(choice))
    return normalized


def _normalize_model_evidence_strength(value, confidence, key_facts):
    strength = str(value or "").strip().lower()
    if strength in {"none", "weak", "medium", "strong"}:
        return strength
    if confidence < 0.20 and not key_facts:
        return "none"
    if confidence < _MODEL_EVIDENCE_CONFIDENCE_THRESHOLD:
        return "weak"
    if confidence >= 0.78:
        return "strong"
    return "medium"


def _infer_model_question_evidence(
    question_label,
    prompt,
    info,
    main_person,
    other_person,
    option_texts,
    description="",
    visual_summary="",
):
    choices = [choice for choice in ("A", "B", "C") if choice in option_texts]
    if not choices:
        return None

    person_blocks = "\n\n".join(
        _compact_person_block(name, person_info)
        for name, person_info in info.items()
    )
    option_lines = "\n".join(f"{choice}) {option_texts.get(choice, '')}" for choice in choices)
    model = get_default_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence extractor for a theory-of-mind multiple-choice task. "
                "Use only the provided question, scenario description, extracted text actions, utterances, "
                "and visual action summary. Do not use answer keys, labels from a dataset file, episode ids, "
                "or memorized test-set information. Infer evidence types from the supplied facts; you may invent "
                "short snake_case evidence type names when the facts need a new category. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return this JSON schema exactly:\n"
                "{\n"
                '  "question_type": "belief|social_goal|belief_of_goal|unknown",\n'
                '  "evidence_source": "model_allowed_inputs",\n'
                '  "evidence_types": ["short_snake_case_type"],\n'
                '  "evidence_strength": "none|weak|medium|strong",\n'
                '  "confidence": 0.0,\n'
                '  "option_support": {"A": 0.0, "B": 0.0, "C": 0.0},\n'
                '  "target_choice": "A|B|C|null",\n'
                '  "inferred_goal": "help|hinder|indifferent|null",\n'
                '  "reason": "one concise sentence",\n'
                '  "key_facts": ["fact from allowed input"],\n'
                '  "counter_evidence": ["optional contrary fact"]\n'
                "}\n\n"
                "The option_support scores are each option's support as the final answer after applying MOST/LEAST wording.\n\n"
                f"Inferred question type from wording: {question_label or 'unknown'}\n"
                f"Main person: {main_person}\n"
                f"Other person: {other_person or 'None'}\n\n"
                f"Scenario description:\n{_truncate_text(description, 3000) or 'Unknown'}\n\n"
                f"Question:\n{prompt}\n\n"
                f"Options:\n{option_lines}\n\n"
                f"Extracted actions and utterances:\n{_truncate_text(person_blocks, 3000) or 'Unknown'}\n\n"
                f"Visual action summary:\n{_truncate_text(visual_summary, 2000) or 'Unknown'}"
            ),
        },
    ]

    data = model.generate_json(
        messages,
        max_new_tokens=_MODEL_EVIDENCE_MAX_TOKENS,
        enable_thinking=get_enable_thinking_default(),
    )
    if not isinstance(data, dict):
        return None

    support = _normalize_choice_support(data.get("option_support"), choices)
    target_choice = str(data.get("target_choice") or "").strip().upper()
    if target_choice not in choices:
        target_choice = None
    if not support and target_choice:
        support = {choice: (0.80 if choice == target_choice else 0.50) for choice in choices}

    confidence = _confidence_to_score(data.get("confidence", 0.0))
    key_facts = [str(item).strip() for item in data.get("key_facts", []) if str(item).strip()]
    evidence_types = _normalize_evidence_types(data.get("evidence_types"))
    inferred_goal = str(data.get("inferred_goal") or data.get("goal") or "").strip().lower()
    if inferred_goal not in _SOCIAL_GOAL_LABELS:
        inferred_goal = None

    return {
        "method": "model_question_evidence",
        "question_label": question_label,
        "model_question_type": str(data.get("question_type", question_label or "unknown")).strip().lower()
        or (question_label or "unknown"),
        "evidence_source": "model_allowed_inputs",
        "evidence_types": evidence_types,
        "evidence_strength": _normalize_model_evidence_strength(
            data.get("evidence_strength"),
            confidence,
            key_facts,
        ),
        "confidence": confidence,
        "option_support": support,
        "target_choice": target_choice,
        "inferred_goal": inferred_goal,
        "reason": str(data.get("reason", "")).strip(),
        "key_facts": key_facts,
        "counter_evidence": [
            str(item).strip()
            for item in data.get("counter_evidence", [])
            if str(item).strip()
        ],
    }


def _option_mentions_help(text):
    lowered = text.lower()
    return "trying to help" in lowered or " to help " in f" {lowered} "


def _option_mentions_hinder(text):
    lowered = text.lower()
    return "trying to prevent" in lowered or " to hinder " in f" {lowered} " or "prevent " in lowered


def _option_mentions_indifferent(text):
    return "indifferent" in text.lower()


def _normalize_distribution(raw_values, labels):
    values = []
    for label in labels:
        try:
            value = max(0.0, float((raw_values or {}).get(label, 0.0)))
        except (TypeError, ValueError):
            value = 0.0
        values.append(value)
    total = sum(values)
    if total <= 0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}
    return {
        label: value / total
        for label, value in zip(labels, values)
    }


def _social_goal_option_map(option_texts):
    goal_to_choices = {goal: [] for goal in _SOCIAL_GOAL_LABELS}
    for choice, option_text in option_texts.items():
        if _option_mentions_help(option_text):
            goal_to_choices["help"].append(choice)
        if _option_mentions_hinder(option_text):
            goal_to_choices["hinder"].append(choice)
        if _option_mentions_indifferent(option_text):
            goal_to_choices["indifferent"].append(choice)
    return goal_to_choices


def _infer_social_goal_posterior(
    info,
    main_person,
    other_person,
    prompt,
    description="",
    visual_summary="",
    social_goal_evidence=None,
):
    if other_person is None:
        return None

    speaker_info = info.get(main_person, {})
    seeker_info = info.get(other_person, {})
    prompt_stem, _ = _parse_question_options(prompt)

    model = get_default_model()
    messages = [
        {
            "role": "system",
            "content": (
                "Infer a shared posterior distribution over social intent labels for the full interaction. "
                "Use the whole trajectory jointly. Return valid JSON only with keys "
                '{"help": number, "hinder": number, "indifferent": number, '
                '"confidence": "low|medium|high", "evidence_summary": "...", "key_facts": ["..."]}. '
                "The three probabilities must sum to 1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question stem:\n{prompt_stem or 'Unknown'}\n\n"
                f"Scenario description:\n{description or 'Unknown'}\n\n"
                f"Speaker of the information: {main_person}\n"
                f"Seeker affected by the information: {other_person}\n\n"
                f"{main_person}'s utterances:\n" + "\n".join(f"- {item}" for item in _person_utterances(speaker_info))
                + "\n\n"
                + f"{main_person}'s actions:\n"
                + "\n".join(f"- {item}" for item in _person_actions(speaker_info))
                + "\n\n"
                + f"{other_person}'s actions:\n"
                + "\n".join(f"- {item}" for item in _person_actions(seeker_info))
                + (
                    f"\n\nOptional visual summary:\n{visual_summary}"
                    if visual_summary
                    else ""
                )
                + (
                      "\n\nNon-binding social-goal evidence from an earlier evidence pass:\n"
                      + json.dumps(
                        {
                            "goal": social_goal_evidence.get("goal"),
                            "reason": social_goal_evidence.get("reason"),
                            "source": social_goal_evidence.get("evidence_source"),
                            "types": social_goal_evidence.get("evidence_types"),
                            "confidence": social_goal_evidence.get("confidence"),
                            "target_object": social_goal_evidence.get("target_object"),
                            "target_location": social_goal_evidence.get("target_location"),
                            "actual_grab_location": social_goal_evidence.get("actual_grab_location"),
                        },
                        ensure_ascii=False,
                    )
                    if social_goal_evidence
                    else ""
                )
                + "\n\nReturn JSON only."
            ),
        },
    ]

    try:
        data = model.generate_json(
            messages,
            max_new_tokens=384,
            enable_thinking=get_enable_thinking_default(),
        )
    except Exception as exc:
        return {
            "method": "social_goal_posterior_unavailable",
            "posterior": {},
            "confidence": "low",
            "evidence_summary": str(exc),
            "key_facts": [],
        }

    if not isinstance(data, dict):
        return {
            "method": "social_goal_posterior_unavailable",
            "posterior": {},
            "confidence": "low",
            "evidence_summary": "invalid_model_output",
            "key_facts": [],
        }

    return {
        "method": "social_goal_posterior_model",
        "posterior": _normalize_distribution(data, _SOCIAL_GOAL_LABELS),
        "confidence": str(data.get("confidence", "medium")).strip().lower() or "medium",
        "evidence_summary": str(data.get("evidence_summary", "")).strip(),
        "key_facts": [str(item).strip() for item in data.get("key_facts", []) if str(item).strip()],
        "heuristic_evidence": social_goal_evidence,
    }


def _select_social_goal_choice_from_posterior(prompt, option_texts, posterior_info):
    if not posterior_info:
        return None, None

    goal_to_choices = _social_goal_option_map(option_texts)
    unique_goal_to_choice = {
        goal: choices[0]
        for goal, choices in goal_to_choices.items()
        if len(choices) == 1
    }
    if len(unique_goal_to_choice) < 2:
        return None, {
            "method": "social_goal_posterior",
            "reason": "insufficient_goal_option_map",
            "goal_to_choice": unique_goal_to_choice,
            "posterior": posterior_info.get("posterior", {}),
        }

    posterior = posterior_info.get("posterior", {})
    polarity = "least" if "LEAST LIKELY" in prompt.upper() else "most"
    candidate_goals = list(unique_goal_to_choice)
    ranked_goals = sorted(
        candidate_goals,
        key=lambda goal: (float(posterior.get(goal, 0.0)), goal),
        reverse=(polarity == "most"),
    )
    best_goal = ranked_goals[0]
    if len(ranked_goals) > 1:
        best_score = float(posterior.get(best_goal, 0.0))
        second_score = float(posterior.get(ranked_goals[1], 0.0))
        if math.isclose(best_score, second_score, rel_tol=_TIE_REL_TOL, abs_tol=_TIE_ABS_TOL):
            return None, {
                "method": "social_goal_posterior",
                "reason": "posterior_tie",
                "goal_to_choice": unique_goal_to_choice,
                "posterior": posterior,
            }

    return unique_goal_to_choice[best_goal], {
        "method": "social_goal_posterior",
        "reason": "posterior_selected",
        "selected_goal": best_goal,
        "question_polarity": polarity,
        "goal_to_choice": unique_goal_to_choice,
        "posterior": posterior,
        "confidence": posterior_info.get("confidence"),
        "evidence_summary": posterior_info.get("evidence_summary"),
        "key_facts": posterior_info.get("key_facts", []),
        "heuristic_evidence": posterior_info.get("heuristic_evidence"),
    }


def _collect_rule_evidence(
    question_label,
    prompt,
    info,
    main_person,
    other_person,
    option_texts,
    description="",
    visual_summary="",
):
    try:
        return _infer_model_question_evidence(
            question_label,
            prompt,
            info,
            main_person,
            other_person,
            option_texts,
            description=description,
            visual_summary=visual_summary,
        )
    except Exception as exc:
        return {
            "method": "model_question_evidence",
            "question_label": question_label,
            "evidence_source": "model_allowed_inputs",
            "evidence_types": ["model_evidence_failed"],
            "evidence_strength": "none",
            "confidence": 0.0,
            "option_support": {},
            "target_choice": None,
            "inferred_goal": None,
            "reason": str(exc),
            "key_facts": [],
            "counter_evidence": [],
        }


def _load_model_evidence_cache():
    if not _USE_MODEL_EVIDENCE_FILE or not _MODEL_EVIDENCE_FILE:
        return {}
    evidence_path = Path(_MODEL_EVIDENCE_FILE)
    if not evidence_path.exists():
        print(f"Warning: model evidence file does not exist: {evidence_path}")
        return {}
    with evidence_path.open("r") as file:
        data = json.load(file)
    if isinstance(data, dict) and isinstance(data.get("episodes"), dict):
        return data["episodes"]
    if isinstance(data, dict):
        return data
    return {}


def _get_cached_model_evidence(evidence_cache, episode_id, question_id):
    if not evidence_cache:
        return None
    episode_record = evidence_cache.get(str(episode_id))
    if not isinstance(episode_record, dict):
        return None

    candidates = []
    for key in ("predictions", "evidence", "questions"):
        question_records = episode_record.get(key)
        if isinstance(question_records, dict):
            candidates.append(question_records.get(str(question_id)))
    candidates.append(episode_record.get(str(question_id)))

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("method") == "model_question_evidence":
            evidence = dict(candidate)
        elif isinstance(candidate.get("model_evidence"), dict):
            evidence = dict(candidate["model_evidence"])
        elif isinstance(candidate.get("evidence"), dict):
            evidence = dict(candidate["evidence"])
        else:
            continue
        evidence["evidence_cache_source"] = _MODEL_EVIDENCE_FILE
        return evidence
    return None


def _normalize_question_label(value):
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _QUESTION_LABELS:
        return normalized
    return None


def _question_label_from_evidence_record(record):
    if not isinstance(record, dict):
        return None

    for key in ("question_label", "model_question_type", "question_type"):
        label = _normalize_question_label(record.get(key))
        if label:
            return label

    for key in ("model_evidence", "evidence"):
        nested = record.get(key)
        label = _question_label_from_evidence_record(nested)
        if label:
            return label

    return None


def _get_cached_question_label(evidence_cache, episode_id, question_id):
    if not evidence_cache:
        return None, None
    episode_record = evidence_cache.get(str(episode_id))
    if not isinstance(episode_record, dict):
        return None, None

    candidates = []
    for bucket in ("evidence", "questions", "predictions", "skipped"):
        question_records = episode_record.get(bucket)
        if isinstance(question_records, dict):
            candidates.append((bucket, question_records.get(str(question_id))))
    candidates.append(("episode_direct", episode_record.get(str(question_id))))

    for source, candidate in candidates:
        label = _question_label_from_evidence_record(candidate)
        if label:
            return label, f"model_evidence_file:{source}"
    return None, None


def _resolve_question_label(
    prompt,
    evidence_cache=None,
    episode_id=None,
    question_id=None,
    rule_evidence=None,
    fallback_label=None,
):
    label = _question_label_from_evidence_record(rule_evidence)
    if label:
        return label, "model_evidence_record"

    label, source = _get_cached_question_label(evidence_cache, episode_id, question_id)
    if label:
        return label, source

    label = _normalize_question_label(fallback_label)
    if label:
        return label, "existing_prediction"

    return _infer_question_label_from_prompt(prompt), "prompt_heuristic"


def _evidence_passes_prior_filter(rule_evidence, question_label):
    if not rule_evidence or rule_evidence.get("method") != "model_question_evidence":
        return False, "not_model_question_evidence"
    if rule_evidence.get("evidence_source") != "model_allowed_inputs":
        return False, "invalid_evidence_source"

    confidence = _confidence_to_score(rule_evidence.get("confidence", 0.0))
    if confidence < _EVIDENCE_PRIOR_MIN_CONFIDENCE:
        return False, "low_confidence"

    strength = str(rule_evidence.get("evidence_strength") or "").strip().lower()
    if _EVIDENCE_PRIOR_STRENGTHS and strength not in _EVIDENCE_PRIOR_STRENGTHS:
        return False, "weak_evidence_strength"

    if question_label == "social_goal":
        if rule_evidence.get("inferred_goal") not in _SOCIAL_GOAL_LABELS:
            return False, "missing_social_goal"
        return True, "passed"

    if question_label in {"belief", "belief_of_goal"}:
        option_support = rule_evidence.get("option_support")
        target_choice = str(rule_evidence.get("target_choice") or "").strip().upper()
        if isinstance(option_support, dict) and option_support:
            return True, "passed"
        if target_choice in {"A", "B", "C"}:
            return True, "passed"
        return False, "missing_option_support"

    return False, "unsupported_question_label"


def _apply_option_score_rule_prior(question_label, prompt, option_scores, rule_evidence):
    if not rule_evidence or question_label not in {"belief", "belief_of_goal"}:
        return dict(option_scores), None
    if rule_evidence.get("method") != "model_question_evidence":
        return dict(option_scores), None

    adjusted_scores = {choice: float(score) for choice, score in option_scores.items()}
    if question_label == "belief_of_goal" and not _ENABLE_BELIEF_OF_GOAL_EVIDENCE_PRIOR:
        return adjusted_scores, None

    passed_filter, filter_reason = _evidence_passes_prior_filter(rule_evidence, question_label)
    if not passed_filter:
        return adjusted_scores, {
            "method": "option_score_model_evidence_prior",
            "applied": False,
            "reason": filter_reason,
            "question_label": question_label,
            "confidence": _confidence_to_score(rule_evidence.get("confidence", 0.0)),
            "evidence_strength": rule_evidence.get("evidence_strength"),
            "evidence_types": rule_evidence.get("evidence_types", []),
            "evidence_cache_source": rule_evidence.get("evidence_cache_source"),
        }

    option_support = rule_evidence.get("option_support") or {}
    confidence = _confidence_to_score(rule_evidence.get("confidence", 0.0))
    if confidence < _MODEL_EVIDENCE_CONFIDENCE_THRESHOLD or not option_support:
        return adjusted_scores, None

    prior_weights = {}
    is_least_likely = "LEAST LIKELY" in prompt.upper()
    for choice in adjusted_scores:
        support = _confidence_to_score(option_support.get(choice, 0.5))
        directional_support = 1.0 - support if is_least_likely else support
        base_weight = 0.70 + 0.80 * directional_support
        prior_weights[choice] = 1.0 + confidence * (base_weight - 1.0)
        adjusted_scores[choice] *= prior_weights[choice]

    return adjusted_scores, {
        "method": "option_score_model_evidence_prior",
        "applied": True,
        "reason": "model_option_evidence_prior",
        "question_label": question_label,
        "prior_weights": prior_weights,
        "question_polarity": "least" if is_least_likely else "most",
        "option_support_interpretation": "final_answer_support",
        "confidence": confidence,
        "evidence_types": rule_evidence.get("evidence_types", []),
        "evidence_strength": rule_evidence.get("evidence_strength"),
        "evidence_cache_source": rule_evidence.get("evidence_cache_source"),
        "prior_filter": {
            "min_confidence": _EVIDENCE_PRIOR_MIN_CONFIDENCE,
            "allowed_strengths": sorted(_EVIDENCE_PRIOR_STRENGTHS),
        },
    }


def _apply_social_goal_rule_prior(posterior_info, rule_evidence, prompt=""):
    return posterior_info, None


def _target_score(option_scores, prompt):
    if not option_scores:
        raise ValueError("No choice probabilities were provided.")
    values = list(option_scores.values())
    if "LEAST LIKELY" in prompt.upper():
        return min(values)
    return max(values)


def _top_two_score_margin(option_scores, prompt=""):
    values = sorted(
        (float(score) for score in option_scores.values()),
        reverse=("LEAST LIKELY" not in prompt.upper()),
    )
    if len(values) < 2:
        return float("inf")
    return abs(values[0] - values[1])


def _tied_choices(option_scores, prompt):
    target = _target_score(option_scores, prompt)
    return [
        choice
        for choice, score in option_scores.items()
        if math.isclose(float(score), float(target), rel_tol=_TIE_REL_TOL, abs_tol=_TIE_ABS_TOL)
    ]


def _resolve_tied_choice(prompt, description, tied_choices, option_texts):
    candidate_lines = "\n".join(
        f"{choice}) {option_texts.get(choice, '').strip()}".rstrip()
        for choice in tied_choices
    )
    model = get_default_model()
    messages = [
        {
            "role": "system",
            "content": "Resolve a multiple-choice tie using the scenario description and question only. Answer with one allowed letter.",
        },
        {
            "role": "user",
            "content": (
                f"Scenario description:\n{description or 'Unknown'}\n\n"
                f"Question:\n{prompt}\n\n"
                f"Only choose among these tied candidates: {', '.join(tied_choices)}\n"
                f"Candidate texts:\n{candidate_lines}\n\n"
                "Respond with one letter only."
            ),
        },
    ]
    return model.choose_from_letters(messages, tied_choices)


def get_choice(option_scores, prompt, description=""):
    tied_choices = _tied_choices(option_scores, prompt)
    if len(tied_choices) == 1:
        return tied_choices[0], None

    _, option_texts = _parse_question_options(prompt)
    try:
        choice, probabilities = _resolve_tied_choice(prompt, description, tied_choices, option_texts)
        return choice, {
            "tied_choices": tied_choices,
            "candidate_texts": {choice: option_texts.get(choice, "") for choice in tied_choices},
            "tie_break_probabilities": {label: float(score) for label, score in probabilities.items()},
        }
    except Exception as exc:
        raise RuntimeError("Tie-break model failed; refusing heuristic fallback.") from exc


def _choose_final_prediction(base_choice, rule_evidence=None, rule_prior=None):
    if rule_evidence is None and rule_prior is None:
        return base_choice, False, None

    decision = {
        "method": (rule_prior or {}).get("method"),
        "base_choice": base_choice,
        "model_evidence_method": (rule_evidence or {}).get("method"),
    }
    if rule_prior:
        decision.update(rule_prior)
    if rule_evidence:
        decision["model_evidence"] = rule_evidence

    changed_choice = bool((rule_prior or {}).get("changed_choice"))
    decision["changed_choice"] = changed_choice
    return base_choice, changed_choice, decision


def _results_file() -> Path:
    default_path = get_results_file_default()
    return Path(os.getenv("MUMATOM_RESULTS_FILE", default_path))


def _build_config():
    backend = get_backend_name()
    local_defaults = get_local_model_defaults()
    api_defaults = get_qwen_api_defaults()
    return {
        "backend": backend,
        "thinking_switch": get_enable_thinking_default(),
        "questions_file": str((Path(__file__).resolve().parent.parent / "Files" / "questions.json")),
        "texts_file": str((Path(__file__).resolve().parent.parent / "Files" / "texts.json")),
        "visual_prompt_file": str((Path(__file__).resolve().parent.parent / "Files" / "actions_extracted.json")),
        "model_id": os.getenv("MUMATOM_MODEL_ID", str(local_defaults["model_id"])),
        "model_dir": os.getenv("MUMATOM_MODEL_DIR", str(local_defaults["model_dir"])),
        "model_cache_dir": os.getenv("MUMATOM_MODEL_CACHE_DIR", str(local_defaults["cache_dir"])),
        "revision": os.getenv("MUMATOM_MODEL_REVISION") or local_defaults["revision"],
        "torch_dtype": os.getenv("MUMATOM_TORCH_DTYPE", str(local_defaults["torch_dtype"])),
        "device_map": os.getenv("MUMATOM_DEVICE_MAP", str(local_defaults["device_map"])),
        "enable_thinking": os.getenv(
            "MUMATOM_ENABLE_THINKING",
            "1" if local_defaults["enable_thinking"] else "0",
        )
        == "1",
        "qwen_api_base_url": os.getenv("QWEN_API_BASE_URL", str(api_defaults["base_url"]))
        if backend == "qwen_api"
        else None,
        "qwen_api_model": os.getenv("QWEN_API_MODEL", str(api_defaults["model_id"])) if backend == "qwen_api" else None,
        "qwen_api_timeout": float(os.getenv("QWEN_API_TIMEOUT", str(api_defaults["timeout"])))
        if backend == "qwen_api"
        else None,
        "qwen_api_enable_thinking": os.getenv(
            "QWEN_API_ENABLE_THINKING",
            "1" if api_defaults["enable_thinking"] else "0",
        )
        == "1"
        if backend == "qwen_api"
        else None,
        "qwen_api_seed": int(os.getenv("QWEN_API_SEED", str(api_defaults["seed"])))
        if backend == "qwen_api"
        else None,
        "model_evidence_confidence_threshold": _MODEL_EVIDENCE_CONFIDENCE_THRESHOLD,
        "model_evidence_max_tokens": _MODEL_EVIDENCE_MAX_TOKENS,
        "model_evidence_file": _MODEL_EVIDENCE_FILE or None,
        "use_model_evidence_file": _USE_MODEL_EVIDENCE_FILE,
        "evidence_prior_min_confidence": _EVIDENCE_PRIOR_MIN_CONFIDENCE,
        "evidence_prior_strengths": sorted(_EVIDENCE_PRIOR_STRENGTHS),
        "enable_belief_of_goal_evidence_prior": _ENABLE_BELIEF_OF_GOAL_EVIDENCE_PRIOR,
        "enable_social_goal_evidence_prior": False,
        "enable_direct_choice_rerank": False,
        "direct_choice_max_margin": 0.10,
        "direct_choice_min_confidence": 0.42,
        "eval_with_answers": False,
        "eval_answers_file": None,
        "results_file": str(_results_file()),
    }


def _default_results():
    return {
        "config": _build_config(),
        "episodes": {},
        "summary": {
            "episodes_finished": 0,
            "total_predictions": 0,
        },
    }


def _save_results(results):
    results_path = _results_file()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def _load_json_file(path):
    with Path(path).open("r") as file:
        return json.load(file)


def _load_results():
    results_path = _results_file()
    if not results_path.exists():
        return _default_results()

    try:
        with results_path.open("r") as file:
            loaded = json.load(file)
    except json.JSONDecodeError:
        print(f"Warning: failed to parse existing results file {results_path}, starting from scratch.")
        return _default_results()

    results = _default_results()
    if isinstance(loaded, dict):
        if isinstance(loaded.get("episodes"), dict):
            results["episodes"] = loaded["episodes"]
        if isinstance(loaded.get("summary"), dict):
            results["summary"] = loaded["summary"]
    return results


def _refresh_results_summary(results):
    episodes = results.setdefault("episodes", {})
    total_predictions = 0
    for episode_record in episodes.values():
        predictions = episode_record.get("predictions")
        if not isinstance(predictions, dict):
            predictions = {}
            episode_record["predictions"] = predictions

        for prediction in predictions.values():
            if isinstance(prediction, dict):
                prediction.pop("gold", None)
                prediction.pop("correct", None)
                prediction.pop("rule_override", None)
                prediction.pop("rule_evidence", None)
                prediction.pop("rule_prior", None)
                prediction.pop("rule_applied", None)
                prediction.pop("rule_decision", None)
        completed_predictions = [prediction for prediction in predictions.values() if _is_question_complete(prediction)]
        episode_total = len(completed_predictions)
        episode_record["total"] = episode_total
        episode_record.pop("correct", None)
        episode_record.pop("accuracy", None)
        total_predictions += episode_total

    results["summary"] = {
        "episodes_finished": len(episodes),
        "total_predictions": total_predictions,
    }
    return total_predictions


def _answer_letter(answer):
    if isinstance(answer, list) and answer:
        answer = answer[0]
    text = str(answer or "").strip()
    match = re.match(r"^\s*([A-C])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-C])\)", text)
    return match.group(1) if match else None


def _choice_from_scores_without_model(option_scores, prompt, fallback=None):
    if not option_scores:
        return fallback, None
    values = {choice: float(score) for choice, score in option_scores.items()}
    target = min(values.values()) if "LEAST LIKELY" in prompt.upper() else max(values.values())
    tied_choices = [
        choice
        for choice, score in values.items()
        if math.isclose(float(score), float(target), rel_tol=_TIE_REL_TOL, abs_tol=_TIE_ABS_TOL)
    ]
    if len(tied_choices) == 1:
        return tied_choices[0], None
    if fallback in tied_choices:
        return fallback, {
            "method": "score_tie_keep_fallback",
            "tied_choices": tied_choices,
            "fallback_choice": fallback,
        }
    fallback_choice = sorted(tied_choices)[0]
    return fallback_choice, {
        "method": "score_tie_alphabetical",
        "tied_choices": tied_choices,
        "fallback_choice": fallback_choice,
    }


def _strip_internal_evidence_artifacts(prediction):
    cleaned = deepcopy(prediction)
    for key in (
        "rule_override",
        "rule_evidence",
        "rule_prior",
        "rule_applied",
        "rule_decision",
    ):
        cleaned.pop(key, None)

    posterior = cleaned.get("social_goal_posterior")
    if isinstance(posterior, dict):
        posterior = dict(posterior)
        for key in (
            "rule_prior",
            "heuristic_evidence",
            "model_evidence",
        ):
            posterior.pop(key, None)
        cleaned["social_goal_posterior"] = posterior
    return cleaned


def _summarize_predictions_with_gold(results, summary_key):
    total = 0
    correct = 0
    changed = 0
    evidence_available = 0
    prior_applied = 0
    by_label = {}
    for episode_record in results.get("episodes", {}).values():
        predictions = episode_record.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        for prediction in predictions.values():
            if not isinstance(prediction, dict):
                continue
            gold = _answer_letter(prediction.get("gold"))
            pred = _answer_letter(prediction.get("prediction"))
            if gold is None or pred is None:
                continue
            label = (
                prediction.get("question_label")
                or _infer_question_label_from_prompt(prediction.get("question", ""))
                or "unknown"
            )
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
    results[summary_key] = summary
    return summary


def _summarize_base_option_predictions(results):
    option_results = {"episodes": {}}
    for episode_id, episode_record in results.get("episodes", {}).items():
        predictions = episode_record.get("predictions", {})
        option_predictions = {}
        if isinstance(predictions, dict):
            for question_id, prediction in predictions.items():
                if not isinstance(prediction, dict):
                    continue
                prompt = prediction.get("question", "")
                pred = _answer_letter(prediction.get("score_based_prediction"))
                if pred is None and isinstance(prediction.get("option_scores"), dict):
                    pred, _ = _choice_from_scores_without_model(
                        prediction["option_scores"],
                        prompt,
                        fallback=_answer_letter(prediction.get("prediction")),
                    )
                option_prediction = {
                    "question": prompt,
                    "question_label": (
                        prediction.get("question_label")
                        or _infer_question_label_from_prompt(prompt)
                    ),
                    "prediction": pred,
                    "gold": prediction.get("gold"),
                }
                option_predictions[question_id] = option_prediction
        option_results["episodes"][episode_id] = {"predictions": option_predictions}
    return _summarize_predictions_with_gold(option_results, "summary")


def _refresh_scored_results_summary(results):
    total_correct = 0
    total_questions = 0
    for episode_record in results.setdefault("episodes", {}).values():
        predictions = episode_record.get("predictions")
        if not isinstance(predictions, dict):
            predictions = {}
            episode_record["predictions"] = predictions
        completed_predictions = [
            prediction
            for prediction in predictions.values()
            if isinstance(prediction, dict)
            and prediction.get("prediction") is not None
            and prediction.get("gold") is not None
        ]
        episode_total = len(completed_predictions)
        episode_correct = sum(1 for prediction in completed_predictions if prediction.get("correct"))
        episode_record["correct"] = episode_correct
        episode_record["total"] = episode_total
        episode_record["accuracy"] = episode_correct / episode_total if episode_total else 0.0
        total_correct += episode_correct
        total_questions += episode_total
    results["summary"] = {
        "episodes_finished": len(results.get("episodes", {})),
        "total_correct": total_correct,
        "total_questions": total_questions,
        "accuracy": total_correct / total_questions if total_questions else 0.0,
    }
    return total_correct, total_questions


def _is_completed_scored_prediction(prediction):
    return (
        isinstance(prediction, dict)
        and prediction.get("prediction") is not None
        and prediction.get("gold") is not None
        and "model_evidence" in prediction
    )


def _default_episode_list(question_data):
    return sorted(int(episode_id) for episode_id in question_data.keys())


def _run_limp_pipeline_with_model_evidence():
    model_evidence_cache = _load_model_evidence_cache()
    if _MODEL_EVIDENCE_FILE:
        print(
            "Model evidence cache: "
            f"{len(model_evidence_cache)} episode(s) loaded from {_MODEL_EVIDENCE_FILE}"
        )

    with open("../Files/questions.json", "r") as file:
        question_data = json.load(file)
    with open("../Files/texts.json", "r") as file:
        text_data = json.load(file)

    episode_override = _parse_episode_override()
    episode_list = episode_override if episode_override is not None else _default_episode_list(question_data)

    results = _load_results()
    results["config"] = _build_config()
    results["config"]["pipeline"] = "limp_model_evidence"
    results["config"]["uses_external_results_file"] = False
    _refresh_scored_results_summary(results)
    _save_results(results)

    for episode in tqdm(episode_list, "Answering questions (LIMP + model evidence)"):
        episode_key = str(episode)
        try:
            questions = question_data[episode_key]
            text = text_data[episode_key]
            episode_description = _clean_description(questions.get("description", ""))
            question_prompts = questions["questions"]
            episode_record = _prepare_episode_record(
                episode,
                text,
                existing_record=results.setdefault("episodes", {}).get(episode_key),
            )
            results["episodes"][episode_key] = episode_record

            completed_question_ids = {
                question_id
                for question_id, prediction in episode_record["predictions"].items()
                if _is_completed_scored_prediction(prediction)
            }
            if len(completed_question_ids) == len(question_prompts):
                if episode_record.pop("error", None) is not None:
                    _refresh_scored_results_summary(results)
                    _save_results(results)
                print(f"Episode {episode} already finished, skip.")
                continue

            print("Episode ", episode)
            if completed_question_ids:
                print(f"Resume from existing results: skip {len(completed_question_ids)} completed question(s).")
            social_goal_posterior_cache = {}
            social_goal_evidence_cache = {}

            for question_id, prompt in question_prompts.items():
                if _is_completed_scored_prediction(episode_record["predictions"].get(question_id)):
                    print(f"Question {question_id} already finished, skip.")
                    continue

                print("Question ", question_id)
                name_list = extract_name_from_question(prompt)
                main_person = name_list[0]
                other_person = name_list[1] if len(name_list) > 1 else None
                name_alignment, text_names = _build_name_alignment(name_list, text, episode_description)
                if name_alignment and any(name_alignment.get(name, name) != name for name in name_list):
                    print("Name alignment:", name_alignment)
                info = {}
                have_utterance = False
                for name in name_list:
                    source_name = name_alignment.get(name, name)
                    person_info = text_parsing.parse_text_info(text, source_name)
                    if person_info["utterance"] is not None:
                        have_utterance = True
                    info[name] = person_info
                utterance = None
                if have_utterance:
                    utterance = {name: info[name]["utterance"] for name in info.keys()}
                print(utterance)

                action_target_name = None
                if episode > 4000:
                    action_target_name = name_list[1] if len(name_list) > 1 else main_person
                else:
                    if info[main_person]["action"] is None:
                        action_target_name = main_person
                    elif len(name_list) > 1:
                        action_target_name = name_list[1]

                if action_target_name is not None and not info[action_target_name]["action"]:
                    info[action_target_name]["action"] = visual_action_extraction.get_action(
                        episode,
                        person_name=name_alignment.get(action_target_name, action_target_name),
                        additional_information=text,
                    )

                visual_action_result = _load_visual_action_result(episode)
                episode_record["names"] = name_list
                episode_record["text_names"] = text_names
                episode_record["name_alignment"] = name_alignment
                episode_record["description"] = episode_description
                episode_record["visual_action_result"] = visual_action_result
                episode_record["visual_summary"] = _build_visual_summary(visual_action_result)
                episode_record["info"] = _normalize_info(info)

                question_label, question_label_source = _resolve_question_label(
                    prompt,
                    evidence_cache=model_evidence_cache,
                    episode_id=episode_key,
                    question_id=question_id,
                )
                init_state, latent_var_options = text_parsing.latent_variable_extraction(
                    info,
                    prompt,
                        question_label,
                )
                episode_record["init_state"] = _normalize_init_state(init_state)

                _, option_texts = _parse_question_options(prompt)
                prob_list = []
                choices = list(latent_var_options.keys())
                for choice, latent_var in latent_var_options.items():
                    probability = compute_prob_GPT.compute_prob(
                        init_state,
                        latent_var,
                        info,
                        main_person,
                        prompt,
                        question_context={
                            "description": episode_description,
                            "question": prompt,
                            "question_label": question_label,
                            "choice_label": choice,
                            "choice_text": option_texts.get(choice, ""),
                            "episode_id": episode_key,
                            "question_id": question_id,
                            "visual_summary": episode_record["visual_summary"],
                        },
                    )
                    prob_list.append(probability)
                option_scores = {choice: float(score) for choice, score in zip(choices, prob_list)}
                final_prob = scipy.special.softmax(prob_list)
                print(final_prob)
                score_margin = _top_two_score_margin(option_scores)
                score_based_choice, tie_break = get_choice(option_scores, prompt, episode_description)
                base_model_choice = score_based_choice
                base_prediction_source = "option_scores"
                prior_adjusted_option_scores = None
                social_goal_posterior = None
                social_goal_selection = None
                social_goal_evidence = None

                if question_label == "social_goal" and other_person is not None:
                    evidence_cache_key = (main_person, other_person)
                    if evidence_cache_key not in social_goal_evidence_cache:
                        social_goal_evidence_cache[evidence_cache_key] = _collect_rule_evidence(
                            question_label,
                            prompt,
                            info,
                            main_person,
                            other_person,
                            option_texts,
                            description=episode_description,
                            visual_summary=episode_record["visual_summary"],
                        )
                    social_goal_evidence = social_goal_evidence_cache[evidence_cache_key]

                model_evidence = _get_cached_model_evidence(model_evidence_cache, episode_key, question_id)
                evidence_prior = None
                if question_label == "social_goal" and other_person is not None:
                    posterior_cache_key = (main_person, other_person)
                    if posterior_cache_key not in social_goal_posterior_cache:
                        social_goal_posterior_cache[posterior_cache_key] = _infer_social_goal_posterior(
                            info,
                            main_person,
                            other_person,
                            prompt,
                            description=episode_description,
                            visual_summary=episode_record["visual_summary"],
                            social_goal_evidence=_social_goal_evidence_from_rule_evidence(social_goal_evidence),
                        )
                    social_goal_posterior = social_goal_posterior_cache[posterior_cache_key]
                    social_goal_posterior, evidence_prior = _apply_social_goal_rule_prior(
                        social_goal_posterior,
                        model_evidence,
                        prompt=prompt,
                    )
                    posterior_choice, social_goal_selection = _select_social_goal_choice_from_posterior(
                        prompt,
                        option_texts,
                        social_goal_posterior,
                    )
                    if evidence_prior is not None:
                        prior_before_choice = base_model_choice
                        prior_after_choice = posterior_choice if posterior_choice is not None else prior_before_choice
                        evidence_prior["pre_prior_choice"] = prior_before_choice
                        evidence_prior["post_prior_choice"] = prior_after_choice
                        evidence_prior["changed_choice"] = (
                            bool(evidence_prior.get("applied"))
                            and prior_after_choice != prior_before_choice
                        )
                    if posterior_choice is not None:
                        base_model_choice = posterior_choice
                        base_prediction_source = (
                            "social_goal_posterior_with_model_evidence_prior"
                            if (evidence_prior or {}).get("changed_choice")
                            else "social_goal_posterior"
                        )
                        print("Social-goal posterior base:", social_goal_selection)
                else:
                    prior_adjusted_option_scores, evidence_prior = _apply_option_score_rule_prior(
                        question_label,
                        prompt,
                        option_scores,
                        model_evidence,
                    )
                    if evidence_prior is not None:
                        prior_choice, prior_tie_break = get_choice(prior_adjusted_option_scores, prompt, episode_description)
                        evidence_prior["pre_prior_choice"] = base_model_choice
                        evidence_prior["post_prior_choice"] = prior_choice
                        evidence_prior["changed_choice"] = prior_choice != base_model_choice
                        if prior_choice != base_model_choice:
                            base_model_choice = prior_choice
                            base_prediction_source = "option_scores_with_model_evidence_prior"
                        if prior_tie_break is not None:
                            tie_break = prior_tie_break

                if tie_break is not None:
                    print("Tie break:", tie_break)
                model_choice, evidence_applied, evidence_decision = _choose_final_prediction(
                    base_model_choice,
                    rule_evidence=model_evidence,
                    rule_prior=evidence_prior,
                )
                if model_evidence is not None:
                    print("Model evidence:", model_evidence)
                if evidence_prior is not None:
                    if evidence_prior.get("changed_choice"):
                        print("Evidence prior changed base:", evidence_prior)
                    else:
                        print("Evidence prior kept base:", evidence_prior)
                if evidence_decision is not None:
                    print("Evidence decision:", evidence_decision)

                gold = questions.get("answers", {}).get(question_id, [None])[0]
                print("Model choose ", model_choice)
                if gold is not None:
                    print("Correct answer ", gold)

                is_correct = model_choice == gold if gold is not None else None
                episode_record["predictions"][question_id] = {
                    "question": prompt,
                    "description": episode_description,
                    "question_label": question_label,
                    "question_label_source": question_label_source,
                    "main_person": main_person,
                    "other_person": other_person,
                    "latent_variables": _normalize_latent_variables(latent_var_options),
                    "option_scores": option_scores,
                    "softmax_scores": {choice: float(score) for choice, score in zip(choices, final_prob.tolist())},
                    "score_margin": score_margin,
                    "tie_break": tie_break,
                    "score_based_prediction": score_based_choice,
                    "base_prediction": base_model_choice,
                    "base_prediction_source": base_prediction_source,
                    "prior_adjusted_option_scores": prior_adjusted_option_scores,
                    "social_goal_posterior": _strip_internal_evidence_artifacts({"social_goal_posterior": social_goal_posterior}).get("social_goal_posterior"),
                    "social_goal_selection": social_goal_selection,
                    "social_goal_evidence_hint_used": bool(social_goal_evidence),
                    "model_evidence": model_evidence,
                    "evidence_prior": evidence_prior,
                    "evidence_prior_applied": evidence_applied,
                    "evidence_decision": evidence_decision,
                    "prediction": model_choice,
                }
                if gold is not None:
                    episode_record["predictions"][question_id]["gold"] = gold
                    episode_record["predictions"][question_id]["correct"] = is_correct
                episode_record.pop("error", None)
                _refresh_scored_results_summary(results)
                _save_results(results)
        except Exception as e:
            episode_record = _prepare_episode_record(
                episode,
                text_data.get(episode_key, ""),
                existing_record=results.setdefault("episodes", {}).get(episode_key),
            )
            results["episodes"][episode_key] = episode_record
            episode_record["error"] = str(e)
            _refresh_scored_results_summary(results)
            _save_results(results)
            raise e

    _refresh_scored_results_summary(results)
    print("Total accuracy rate: ", results["summary"].get("accuracy"))
    _save_results(results)
    return True


def _load_visual_action_result(episode_id):
    actions_file = Path(__file__).resolve().parent.parent / "Files" / "actions_extracted.json"
    if not actions_file.exists():
        return None
    with actions_file.open("r") as file:
        data = json.load(file)
    entry = data.get(str(episode_id))
    if not isinstance(entry, dict):
        return None
    return {
        "episode_id": episode_id,
        "video_path": entry.get("video_path"),
        "prompt": entry.get("prompt", ""),
        "sampled_frame_indices": entry.get("sampled_frame_indices", []),
        "sampled_timestamps": entry.get("sampled_timestamps", []),
        "action_summary": entry.get("action", ""),
        "actions": entry.get("actions_list", []),
        "observations": entry.get("observations", []),
    }


def _infer_visual_action_target_name(visual_action_result, name_list, name_alignment):
    if not visual_action_result or not name_list:
        return None
    search_text = "\n".join(
        str(visual_action_result.get(key, "") or "")
        for key in ("prompt", "action_summary")
    )
    candidates = []
    for pattern in (r"\bFor agent ([A-Z][a-z]+)\b", r"\bFor ([A-Z][a-z]+)\b"):
        candidates.extend(re.findall(pattern, search_text))
    for candidate in candidates:
        if candidate in name_list:
            return candidate
        for question_name, source_name in (name_alignment or {}).items():
            if candidate == source_name and question_name in name_list:
                return question_name
    return None


def _build_visual_summary(visual_action_result):
    if not visual_action_result:
        return ""
    parts = []
    action_summary = str(visual_action_result.get("action_summary", "")).strip()
    if action_summary:
        parts.append(f"Video action summary: {action_summary}")
    actions = [str(item).strip() for item in visual_action_result.get("actions", []) if str(item).strip()]
    if actions:
        parts.append("Video actions:\n" + "\n".join(f"- {action}" for action in actions))
    observations = [str(item).strip() for item in visual_action_result.get("observations", []) if str(item).strip()]
    if observations:
        parts.append("Video observations:\n" + "\n".join(f"- {item}" for item in observations))
    return "\n\n".join(parts)
def _normalize_info(info):
    normalized = {}
    for name, person_info in info.items():
        normalized[name] = {
            "actions": person_info["action"] or [],
            "utterances": person_info["utterance"] or [],
        }
    return normalized


def _normalize_init_state(init_state):
    if init_state is None:
        return []
    if isinstance(init_state, list):
        return [str(item).strip() for item in init_state if str(item).strip()]
    text = str(init_state).strip()
    if not text:
        return []
    if "\n" in text:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if ";" in text:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [text]


def _normalize_latent_variables(latent_var_options):
    normalized = {}
    for choice, latent_var in latent_var_options.items():
        try:
            parsed = compute_prob_GPT.parse_latent_var(latent_var)
            normalized[choice] = {
                "belief": parsed["Belief"],
                "social_goal": parsed["Social Goal"],
                "believed_goal": parsed["Believed Goal"],
            }
        except Exception:
            normalized[choice] = {"raw": latent_var}
    return normalized


def _prepare_episode_record(episode_id, text, existing_record=None):
    episode_record = dict(existing_record) if isinstance(existing_record, dict) else {}
    episode_record["episode_id"] = episode_id
    episode_record["names"] = episode_record.get("names") if isinstance(episode_record.get("names"), list) else []
    episode_record["text_names"] = episode_record.get("text_names") if isinstance(episode_record.get("text_names"), list) else []
    episode_record["name_alignment"] = (
        episode_record.get("name_alignment") if isinstance(episode_record.get("name_alignment"), dict) else {}
    )
    episode_record["text"] = text
    episode_record["description"] = str(episode_record.get("description", ""))
    episode_record["visual_summary"] = str(episode_record.get("visual_summary", ""))
    episode_record["info"] = episode_record.get("info") if isinstance(episode_record.get("info"), dict) else {}
    episode_record["init_state"] = episode_record.get("init_state") if isinstance(episode_record.get("init_state"), list) else []
    episode_record["predictions"] = episode_record.get("predictions") if isinstance(episode_record.get("predictions"), dict) else {}
    episode_record["total"] = int(episode_record.get("total", 0) or 0)
    episode_record.pop("correct", None)
    episode_record.pop("accuracy", None)
    return episode_record


def _is_question_complete(prediction):
    return (
        isinstance(prediction, dict)
        and prediction.get("prediction") is not None
        and "model_evidence" in prediction
    )


def _parse_episode_override():
    raw = os.getenv("MUMATOM_EPISODES", "").strip()
    if not raw:
        return None
    try:
        if raw.startswith("["):
            parsed = ast.literal_eval(raw)
            return [int(item) for item in parsed]
    except Exception:
        pass
    return [int(part.strip()) for part in raw.split(",") if part.strip()]

if __name__ == "__main__":
    _run_limp_pipeline_with_model_evidence()

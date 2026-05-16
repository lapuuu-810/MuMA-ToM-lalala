from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from model_backend import get_default_model
from runtime_hparams import get_enable_thinking_default


FILES_DIR = Path(__file__).resolve().parent.parent / "Files"
DEFAULT_ACTIONS_FILE = FILES_DIR / "actions_extracted.json"
DEFAULT_QUESTIONS_FILE = FILES_DIR / "questions.json"

TRANSFER_VERB_PATTERN = re.compile(
    r"\b(?:pick(?:s|ed)? up|grab(?:s|bed)?|place(?:s|d)?|put(?:s)?|take(?:s)?|took|retrieve(?:s|d)?|carry|carries|carried|move(?:s|d)?)\b",
    re.IGNORECASE,
)
GENERIC_OBJECT_PATTERN = re.compile(r"\b(?:some object|an object|a object|the object|object)\b", re.IGNORECASE)
ACTION_SPLIT_PATTERN = re.compile(
    r",\s*|;\s*|\.\s*|(?:\band\b|\bthen\b)\s+(?=(?:walk|walks|walked|move|moves|moved|head|heads|headed|go|goes|went|approach|approaches|approached|open|opens|opened|close|closes|closed|grab|grabs|grabbed|place|places|placed|put|puts|take|takes|took|get|gets|got|retrieve|retrieves|retrieved|carry|carries|carried|pick|picks|picked)\b)",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    with path.open("w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def clean_description(description: Any) -> str:
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


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def normalize_action(text: Any) -> str:
    return " ".join(str(text or "").strip().strip(" ,.;").split())


def extract_target_person(prompt_text: str) -> str:
    patterns = [
        r"\bFor\s+([A-Z][A-Za-z_-]+)\b",
        r"Person's name:\s*([A-Z][A-Za-z_-]+)",
        r"person is\s+([A-Z][A-Za-z_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt_text)
        if match:
            return match.group(1).strip()
    return ""


def dedupe_phrases(phrases: list[str], limit: int = 8) -> list[str]:
    stopwords = {
        "it",
        "him",
        "her",
        "them",
        "object",
        "some object",
        "an object",
        "the object",
        "goal",
        "current location",
        "desired location",
        "earlier",
        "later",
    }
    seen: set[str] = set()
    result: list[str] = []
    for phrase in phrases:
        normalized = normalize_text(re.sub(r"^(?:a|an|the)\s+", "", phrase))
        normalized = normalized.strip(" .,;:")
        if not normalized or normalized in stopwords or len(normalized) < 3:
            continue
        if normalized.startswith("to ") or normalized.startswith("toward ") or normalized.startswith("towards "):
            continue
        if " had placed " in normalized or " placed earlier" in normalized:
            continue
        if len(normalized.split()) > 5:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
        if len(result) >= limit:
            break
    return result


def target_person_context(description: str, target_person: str) -> str:
    if not target_person:
        return description
    sentences = re.split(r"(?<=[.!?])\s+", description)
    selected: list[str] = []
    target_seen = False
    for sentence in sentences:
        sentence_text = sentence.strip()
        if not sentence_text:
            continue
        lowered = sentence_text.lower()
        if target_person.lower() in lowered:
            selected.append(sentence_text)
            target_seen = True
            continue
        if target_seen and re.match(r"^(?:he|she|they|then|afterward|later)\b", sentence_text, re.IGNORECASE):
            selected.append(sentence_text)
    return " ".join(selected) or description


def extract_target_objects(description: str, target_person: str) -> list[str]:
    context = target_person_context(description, target_person)
    object_patterns = [
        r"\b(?:grab|grabs|grabbed|pick up|picks up|picked up|take|takes|took|place|places|placed|put|puts|move|moves|moved|carry|carries|carried|retrieve|retrieves|retrieved) (?:a |an |the )?([a-z0-9][a-z0-9\- ]*?)(?= (?:from|inside|in|on|onto|to|at|towards|toward)\b|,|\.|;| and |$)",
    ]
    objects: list[str] = []
    for pattern in object_patterns:
        objects.extend(match.group(1) for match in re.finditer(pattern, context, re.IGNORECASE))
    return dedupe_phrases(objects)


def entry_action_text(entry: dict[str, Any]) -> str:
    actions_list = entry.get("actions_list")
    if isinstance(actions_list, list):
        actions_text = " ".join(str(action) for action in actions_list)
    else:
        actions_text = str(actions_list or "")
    return " ".join([str(entry.get("action", "")), actions_text])


def action_list_from_entry(entry: dict[str, Any]) -> list[str]:
    actions = entry.get("actions_list")
    if isinstance(actions, list) and actions:
        return [normalize_action(action) for action in actions if normalize_action(action)]

    action_text = normalize_action(entry.get("action", ""))
    if not action_text:
        return []
    split_actions = [normalize_action(chunk) for chunk in ACTION_SPLIT_PATTERN.split(action_text)]
    return [action for action in split_actions if action]


def has_target_transfer_info(entry: dict[str, Any], target_objects: list[str]) -> bool:
    text = normalize_text(entry_action_text(entry))
    if not text:
        return False
    has_transfer = bool(TRANSFER_VERB_PATTERN.search(text))
    if not has_transfer:
        return False
    if target_objects:
        return any(normalize_text(target_object) in text for target_object in target_objects)
    return not GENERIC_OBJECT_PATTERN.search(text)


def infer_missing_target_actions_json(
    model: Any,
    *,
    episode_id: str,
    description: str,
    prompt_text: str,
    target_person: str,
    target_objects: list[str],
    current_action: str,
    current_actions_list: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "You compare action extractions with scenario descriptions. Return JSON only.",
        },
        {
            "role": "user",
            "content": (
                "Find only the missing key target-object actions in the current extracted action list.\n"
                "Use the scenario description as evidence. Do not use outside knowledge.\n"
                "Do not rewrite existing actions. Only propose missing actions involving the key target object, such as going to the object, picking/grabbing it, carrying/moving it, putting/placing it, or opening/closing a container only when needed for the target object's movement.\n"
                "If the current extraction already contains the target object and its key transfer actions, return needs_update=false and empty lists.\n"
                "Use short action phrases consistent with the current actions_list style. Omit the person's name if current actions_list omits the name.\n"
                "insert_after_existing_action_indices uses -1 for before the first existing action, otherwise the index of the existing action after which to insert. If uncertain, choose the best chronological position.\n\n"
                "Return valid JSON with exactly this schema:\n"
                "{\n"
                "  \"needs_update\": true,\n"
                "  \"target_object\": \"...\",\n"
                "  \"missing_action_summary\": \"single-line summary of only the missing actions\",\n"
                "  \"missing_actions_list\": [\"short missing action phrase\"],\n"
                "  \"insert_after_existing_action_indices\": [-1],\n"
                "  \"observations\": [\"short observation about the inserted target movement\"]\n"
                "}\n\n"
                f"Episode id: {episode_id}\n"
                f"Target person inferred from extraction prompt: {target_person or 'unknown'}\n"
                f"Target object candidates from description: {json.dumps(target_objects, ensure_ascii=False)}\n\n"
                f"Original extraction prompt:\n{prompt_text}\n\n"
                f"Scenario description:\n{description}\n\n"
                f"Current action field:\n{current_action}\n\n"
                f"Current actions_list:\n{json.dumps(current_actions_list, ensure_ascii=False)}"
            ),
        },
    ]
    data = model.generate_json(
        messages,
        max_new_tokens=max_new_tokens,
        enable_thinking=get_enable_thinking_default(),
    )
    return data if isinstance(data, dict) else {}


def structure_action_text(model: Any, raw_action_text: str, max_new_tokens: int = 256) -> dict[str, Any]:
    if not raw_action_text:
        return {"action_summary": "", "actions_list": [], "observations": []}
    data = model.generate_json(
        [
            {
                "role": "system",
                "content": "You convert action descriptions into structured JSON. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Convert the following action description into valid JSON with this schema:\n"
                    "{\n"
                    "  \"action_summary\": \"single-line chronological action summary\",\n"
                    "  \"actions_list\": [\"...\"],\n"
                    "  \"observations\": [\"...\"]\n"
                    "}\n"
                    "Rules:\n"
                    "- Keep actions_list chronological.\n"
                    "- Use short action phrases in actions_list.\n"
                    "- Preserve target object names and source/destination locations.\n"
                    "- Do not add new actions beyond the input.\n\n"
                    f"Input: {raw_action_text}"
                ),
            },
        ],
        max_new_tokens=max_new_tokens,
        enable_thinking=get_enable_thinking_default(),
    )
    if not isinstance(data, dict):
        return {"action_summary": raw_action_text, "actions_list": [], "observations": []}
    return {
        "action_summary": str(data.get("action_summary", "")).strip() or raw_action_text,
        "actions_list": [normalize_action(item) for item in data.get("actions_list", []) if normalize_action(item)],
        "observations": [str(item).strip() for item in data.get("observations", []) if str(item).strip()],
    }


def normalize_insert_indices(indices: Any, action_count: int, missing_count: int) -> list[int]:
    if not isinstance(indices, list):
        return [action_count - 1] * missing_count
    normalized: list[int] = []
    for index in indices[:missing_count]:
        try:
            value = int(index)
        except (TypeError, ValueError):
            value = action_count - 1
        normalized.append(max(-1, min(value, action_count - 1)))
    while len(normalized) < missing_count:
        normalized.append(normalized[-1] if normalized else action_count - 1)
    return normalized


def insert_missing_actions(existing_actions: list[str], missing_actions: list[str], insert_indices: list[int]) -> list[str]:
    seen = {normalize_text(action) for action in existing_actions}
    slots: dict[int, list[str]] = {}
    for action, index in zip(missing_actions, insert_indices):
        normalized_action = normalize_action(action)
        if not normalized_action:
            continue
        key = normalize_text(normalized_action)
        if key in seen:
            continue
        seen.add(key)
        slots.setdefault(index, []).append(normalized_action)

    result: list[str] = []
    result.extend(slots.get(-1, []))
    for index, action in enumerate(existing_actions):
        result.append(action)
        result.extend(slots.get(index, []))
    if not existing_actions:
        result.extend(slots.get(-1, []))
    return result


def summarize_actions(actions: list[str], target_person: str) -> str:
    if not actions:
        return ""
    if len(actions) == 1:
        body = actions[0]
    else:
        body = ", ".join(actions[:-1]) + ", and " + actions[-1]
    if target_person and not body.lower().startswith(target_person.lower()):
        return f"{target_person} {body}"
    return body


def merge_observations(existing_observations: Any, new_observations: list[str]) -> list[str]:
    result = [str(item).strip() for item in existing_observations if str(item).strip()] if isinstance(existing_observations, list) else []
    seen = {normalize_text(item) for item in result}
    for observation in new_observations:
        normalized = normalize_text(observation)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(observation)
    return result


def sorted_episode_keys(actions_data: dict[str, Any]) -> list[str]:
    return sorted(actions_data.keys(), key=lambda value: int(value) if str(value).isdigit() else str(value))


def supplement_target_actions(
    *,
    actions_file: Path = DEFAULT_ACTIONS_FILE,
    questions_file: Path = DEFAULT_QUESTIONS_FILE,
    episode_ids: list[int] | None = None,
    dry_run: bool = False,
    max_new_tokens: int = 512,
) -> None:
    actions_data = load_json(actions_file)
    questions_data = load_json(questions_file)
    episode_keys = [str(episode_id) for episode_id in episode_ids] if episode_ids else sorted_episode_keys(actions_data)
    model = get_default_model()
    changed = 0
    skipped = 0

    for episode_key in episode_keys:
        entry = actions_data.get(episode_key)
        question_entry = questions_data.get(episode_key)
        if not isinstance(entry, dict) or not isinstance(question_entry, dict):
            print(f"Skip episode {episode_key}: missing action or question entry")
            skipped += 1
            continue

        description = clean_description(question_entry.get("description", ""))
        if not description:
            print(f"Skip episode {episode_key}: missing description")
            skipped += 1
            continue

        current_actions = action_list_from_entry(entry)
        current_action = str(entry.get("action", "")).strip()
        prompt_text = str(entry.get("prompt", "")).strip()
        target_person = extract_target_person(prompt_text)
        target_objects = extract_target_objects(description, target_person)

        if has_target_transfer_info(entry, target_objects):
            print(f"Skip episode {episode_key}: target transfer already present")
            skipped += 1
            continue

        inferred = infer_missing_target_actions_json(
            model,
            episode_id=episode_key,
            description=description,
            prompt_text=prompt_text,
            target_person=target_person,
            target_objects=target_objects,
            current_action=current_action,
            current_actions_list=current_actions,
            max_new_tokens=max_new_tokens,
        )
        if not inferred.get("needs_update"):
            print(f"Skip episode {episode_key}: model found no missing target actions")
            skipped += 1
            continue

        missing_actions = [normalize_action(action) for action in inferred.get("missing_actions_list", []) if normalize_action(action)]
        missing_summary = normalize_action(inferred.get("missing_action_summary", ""))
        if not missing_actions and missing_summary:
            missing_actions = [missing_summary]
        if not missing_actions:
            print(f"Skip episode {episode_key}: model returned no missing action list")
            skipped += 1
            continue

        structured_missing = structure_action_text(model, "; ".join(missing_actions))
        normalized_missing_actions = structured_missing["actions_list"] or missing_actions
        insert_indices = normalize_insert_indices(
            inferred.get("insert_after_existing_action_indices"),
            action_count=len(current_actions),
            missing_count=len(normalized_missing_actions),
        )
        updated_actions = insert_missing_actions(current_actions, normalized_missing_actions, insert_indices)
        if len(updated_actions) == len(current_actions):
            print(f"Skip episode {episode_key}: normalized missing actions duplicate existing actions")
            skipped += 1
            continue

        updated_action_summary = summarize_actions(updated_actions, target_person)
        updated_observations = merge_observations(
            entry.get("observations", []),
            [str(item).strip() for item in inferred.get("observations", []) if str(item).strip()] + structured_missing["observations"],
        )

        print(f"Supplement episode {episode_key}: insert {len(updated_actions) - len(current_actions)} target actions")
        print("  target objects:", ", ".join(target_objects) if target_objects else "unknown")
        print("  inserted:", normalized_missing_actions)
        changed += 1
        if not dry_run:
            entry["action"] = updated_action_summary
            entry["actions_list"] = updated_actions
            entry["observations"] = updated_observations

    if dry_run:
        print(f"Dry run complete. Would update {changed} episodes; skipped {skipped}.")
        return

    if changed:
        save_json(actions_file, actions_data)
    print(f"Done. Updated {changed} episodes; skipped {skipped}.")


def parse_episode_ids(raw_values: list[str] | None) -> list[int] | None:
    if not raw_values:
        return None
    episode_ids: list[int] = []
    for raw_value in raw_values:
        for chunk in raw_value.split(","):
            chunk = chunk.strip()
            if chunk:
                episode_ids.append(int(chunk))
    return episode_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert missing target-object actions into actions_extracted.json using questions.json descriptions.")
    parser.add_argument("--actions-file", type=Path, default=DEFAULT_ACTIONS_FILE)
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--episodes", nargs="*", help="Episode ids, e.g. --episodes 138 153 or --episodes 138,153. Default: all episodes.")
    parser.add_argument("--dry-run", action="store_true", help="Run model inference and print changes without writing actions_extracted.json.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    supplement_target_actions(
        actions_file=args.actions_file,
        questions_file=args.questions_file,
        episode_ids=parse_episode_ids(args.episodes),
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

import ast
import json
import re

from model_backend import get_default_model
from runtime_hparams import get_enable_thinking_default


def _chat(prompt, max_new_tokens):
    return get_default_model().chat(
        [
            {"role": "system", "content": prompt},
        ],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        enable_thinking=get_enable_thinking_default(),
    ).strip()


def _normalize_action_list(raw_text):
    actions_match = re.search(r"Actions:\s*(\[[^\]]*\])", raw_text, re.DOTALL)
    action = actions_match.group(1) if actions_match else raw_text.strip()
    try:
        parsed = ast.literal_eval(action)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError, TypeError):
        pass

    repaired = get_default_model().generate_json(
        [
            {
                "role": "system",
                "content": (
                    "Return valid JSON only with schema {\"actions\": [\"...\"]}. "
                    "Convert the provided action summary into a valid list of action strings."
                ),
            },
            {
                "role": "user",
                "content": raw_text,
            },
        ],
        max_new_tokens=256,
        enable_thinking=get_enable_thinking_default(),
    )
    if isinstance(repaired, dict):
        repaired = repaired.get("actions")
    if isinstance(repaired, list):
        return [str(item).strip() for item in repaired if str(item).strip()]
    raise ValueError(f"Failed to parse action list from model output: {raw_text}")


def _extract_target_name(prompt_text, action_text):
    prompt_match = re.search(r"\b(?:For|for)\s+(?:agent\s+)?([A-Z][a-z]+)\b", prompt_text)
    if prompt_match:
        return prompt_match.group(1)

    action_match = re.match(r"\s*([A-Z][a-z]+)\b", action_text)
    if action_match:
        return action_match.group(1)

    return None


def get_action(episode_id, person_name=None, additional_information=None):
    with open("../Files/actions_extracted.json", "r") as file:
        data = json.load(file)
    entry = data[str(episode_id)]
    with open("../Files/texts.json", "r") as file:
        text = json.load(file)[str(episode_id)]

    if additional_information is None:
        additional_information = text.split("\n")[0]

    if episode_id < 4000 and additional_information == text.split("\n")[0]:
        prompt = f"""
            Read a piece of text, select the object that the person is picking up and moving around, only include the object name in your answer.

            Text: {text}
        """
        additional_information = _chat(prompt, max_new_tokens=48)
        print(additional_information)

    name = person_name or _extract_target_name(entry.get("prompt", ""), entry.get("action", ""))
    if not name:
        prompt = f"""Read a piece of text, select a person's name from the text. Only output person's name
        Input text: {entry["action"]}
        """
        name = _chat(prompt, max_new_tokens=32)
        print(name)
    prompt = """
    Input text: {}
    Additional_information: {}
    Person's name: {}
    You will read some text describe a person's action. The name of the person is given. Only summarize his/her action and ignore actions of other person. Reorganize the person's actions.
    Possible actions include: walk towards somewhere, grab something from somewhere, open some container, close some container, put something somewhere. Only summarize these actions and their synonyms in this form and abandon mismatch actions. Omit peron's name. When mentioning location name, try to infer room the location is inside and include it in the action in form "[container] in [room_name]"
    Check objects mentioned in the Additional Information section. Replace any object mentioned in action with the object appeared in that section
    Formulate your final answer in the following form.
    Actions:
    ["action1", "action2", ....]
    """
    print(prompt.format(entry["action"], additional_information, name))
    actions = _chat(prompt.format(entry["action"], additional_information, name), max_new_tokens=384)
    action_prediction = _normalize_action_list(actions)
    print(action_prediction)
    return action_prediction

import ast
import re

from model_backend import get_default_model
from runtime_hparams import get_enable_thinking_default


latent_variable_prompt = """
    You will read a question about agents' mind and ideas, and the initial state of the environment from which agents' are interacting in. Agents' knowledge & belief are about this initial state, but not necessarily changed state after some actions. For each choice, extract one set of second person's belief (make sure to turn it into some statement about the environment state), second person's social goal toward first peron's actions (help, hinder or some similar words of indepedent), and second person's believed first person's physical goal (some arrangement of objects). Organize the answer in this way: A: Belief: contents; Social goal: contents; Believed Goal: contents. B: Belief: contents; Social goal: contents; Believed Goal: contents. C: Belief: contents; Social goal: contents; Believed Goal: contents. Do not include any other information or extra contents. Make sure your answer follow the format requirement, use ";" to separate variables within each choice and end response with ".". Separate contents of "A", "B" and "C" with "."

    Question: {}
"""

init_state_prompt = """
    You will read one or two person's actions in a list like form. From the actions taken, extract the initial state of the environment before any people act.
    Check each grab action or synonyms. Describe it in the form "There is a [object grabbed] [on/inside location of grabbing].
    Preserve the full location phrase exactly whenever it is available. Do not shorten location qualifiers such as room names, "other", "another room", or similar descriptors.
    If two locations share the same head noun but have different qualifiers, keep them distinct.
    Only include environment states statements. Do not include any other information or extra contents.

    Actions: {}
"""


OPTION_PATTERN = re.compile(r"^\s*([A-C])\)\s*(.+?)(?=^\s*[A-C]\)\s*|\Z)", re.MULTILINE | re.DOTALL)
ARTICLE_PATTERN = re.compile(r"^(?:a|an|the|some)\s+", re.IGNORECASE)


def _chat(prompt, max_new_tokens):
    return get_default_model().chat(
        [
            {"role": "system", "content": prompt},
        ],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        enable_thinking=get_enable_thinking_default(),
    ).strip()


def _extract_section_text(label, raw_text):
    pattern = rf"{label}:\s*(.*?)(?=\n\s*(?:Actions|Utterance):|\Z)"
    match = re.search(pattern, raw_text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_literal_list(section_text):
    if not section_text:
        return []
    try:
        parsed = ast.literal_eval(section_text)
    except (ValueError, SyntaxError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item).strip() for item in parsed if str(item).strip()]


def _repair_info_lists(raw_text):
    repaired = get_default_model().generate_json(
        [
            {
                "role": "system",
                "content": (
                    "Return valid JSON only with schema {\"actions\": [\"...\"], \"utterances\": [\"...\"]}. "
                    "Convert the provided content into two clean string lists. If a section is missing, use an empty list."
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
    if not isinstance(repaired, dict):
        return [], []

    actions = repaired.get("actions", [])
    utterances = repaired.get("utterances", [])
    if not isinstance(actions, list):
        actions = []
    if not isinstance(utterances, list):
        utterances = []

    actions = [str(item).strip() for item in actions if str(item).strip()]
    utterances = [str(item).strip() for item in utterances if str(item).strip()]
    return actions, utterances


def _normalize_spaces(text):
    return " ".join(str(text).strip().split())


def _sentence(text):
    text = _normalize_spaces(text).rstrip(".")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def _strip_articles(text):
    return ARTICLE_PATTERN.sub("", _normalize_spaces(text))


def _normalize_init_state_list(init_state):
    if init_state is None:
        return []
    if isinstance(init_state, list):
        items = init_state
    else:
        text = str(init_state).strip()
        if not text:
            return []
        if "\n" in text:
            items = text.splitlines()
        elif ";" in text:
            items = text.split(";")
        else:
            items = [text]
    return [_sentence(item) for item in items if _normalize_spaces(item)]


def _parse_question_options(question):
    matches = list(OPTION_PATTERN.finditer(question))
    options = {match.group(1): _normalize_spaces(match.group(2)) for match in matches}
    stem = question[: matches[0].start()].strip() if matches else question.strip()
    return _normalize_spaces(stem), options


def _infer_question_label(question_label, stem, options):
    if question_label in {"belief", "social_goal", "belief_of_goal"}:
        return question_label

    option_blob = " ".join(options.values()).lower()
    stem_lower = stem.lower()
    if "based on the actions of the agents" in stem_lower or "wants to place" in option_blob or "doesn't know" in option_blob:
        return "belief_of_goal"
    if "believed that there was" in option_blob:
        return "belief"
    if "has been trying to" in option_blob or "was indifferent" in option_blob or "prevent" in option_blob:
        return "social_goal"
    return None


def _extract_goal_object_from_text(text):
    normalized = _normalize_spaces(text)
    patterns = [
        r"where (?:the |a |an )?([a-z0-9][a-z0-9\- ]*?) might be",
        r"locate (?:the |a |an )?([a-z0-9][a-z0-9\- ]*?)(?:[.?!]|$)",
        r"find(?:ing)? (?:the |a |an )?([a-z0-9][a-z0-9\- ]*?)(?:[.?!]|$)",
        r"get (?:the |a |an )?([a-z0-9][a-z0-9\- ]*?)(?:[.?!]|$)",
        r"wants to place (?:the |a |an )?([a-z0-9][a-z0-9\- ]*?) (?:inside|in|on|onto|to)\b",
        r"believed that there was (?:a |an |the )?([a-z0-9][a-z0-9\- ]*?) (?:inside|in|on)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return _strip_articles(match.group(1).rstrip("."))
    return None


def _extract_object_from_action(action):
    normalized = _normalize_spaces(action)
    patterns = [
        r"\bgrab(?:bed)? (?:the |a |an )?(.+?) from\b",
        r"\bput (?:the |a |an )?(.+?) (?:inside|in|on|onto|to)\b",
        r"\bplace(?:d)? (?:the |a |an )?(.+?) (?:inside|in|on|onto|to)\b",
        r"\bmove(?:d)? (?:the |a |an )?(.+?) (?:inside|in|on|onto|to)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return _strip_articles(match.group(1).rstrip("."))
    return None


def _infer_goal_object_from_info(info, fallback_text=""):
    names = list(info.keys())
    candidate_texts = []
    if len(names) > 1:
        candidate_texts.extend(info.get(names[1], {}).get("utterance") or [])
        candidate_texts.extend(info.get(names[0], {}).get("utterance") or [])
    if fallback_text:
        candidate_texts.append(fallback_text)

    for text in candidate_texts:
        object_name = _extract_goal_object_from_text(text)
        if object_name:
            return object_name

    for name in names[::-1]:
        for action in reversed(info.get(name, {}).get("action") or []):
            object_name = _extract_object_from_action(action)
            if object_name:
                return object_name
    return None


def _infer_desired_location_from_actions(actions, object_name):
    target_object = _strip_articles(object_name).lower()
    patterns = [
        r"\b(?:put|place(?:d)?|move(?:d)?) (?:the |a |an )?(?P<object>.+?) (?P<prep>inside|in|on|onto|to) (?P<location>.+)$",
    ]
    for action in reversed(actions or []):
        normalized = _normalize_spaces(action).rstrip(".")
        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue
            action_object = _strip_articles(match.group("object")).lower()
            if target_object and action_object != target_object:
                continue
            prep = match.group("prep").lower()
            location = _normalize_spaces(match.group("location"))
            return f"{prep} {location}"
    return None


def _infer_goal_statement(info, fallback_text=""):
    names = list(info.keys())
    other_name = names[1] if len(names) > 1 else "Other person"
    goal_object = _infer_goal_object_from_info(info, fallback_text)

    for utterance in info.get(other_name, {}).get("utterance") or []:
        if goal_object and re.search(r"where .* might be|locate|find", utterance, re.IGNORECASE):
            return _sentence(f"{other_name} wants to find the {goal_object}")

    if goal_object:
        desired_location = _infer_desired_location_from_actions(info.get(other_name, {}).get("action") or [], goal_object)
        if desired_location:
            return _sentence(f"{other_name} wants to place the {goal_object} {desired_location}")
        return _sentence(f"{other_name} wants to get the {goal_object}")

    return _sentence(f"{other_name} wants to complete their goal")


def _select_relevant_state(init_state, info, fallback_text=""):
    states = _normalize_init_state_list(init_state)
    goal_object = _infer_goal_object_from_info(info, fallback_text)
    if goal_object:
        for state in states:
            if goal_object.lower() in state.lower():
                return state
    return states[0] if states else ""


def _parse_social_goal_from_text(text):
    normalized = _normalize_spaces(text).lower()
    if any(token in normalized for token in ["doesn't know", "does not know", "without thinking about what", "indifferent", "no particular inclination"]):
        return "indifferent"
    if "prevent" in normalized or "hinder" in normalized:
        return "hinder"
    if "help" in normalized:
        return "help"
    return "unknown"


def _belief_from_option_text(option_text):
    match = re.search(r"believed that (.*?)(?::|$)", option_text, re.IGNORECASE)
    if not match:
        return ""
    clause = match.group(1).strip().rstrip(".")
    clause = re.sub(r"^there was\b", "There is", clause, flags=re.IGNORECASE)
    clause = re.sub(r"^there is\b", "There is", clause, flags=re.IGNORECASE)
    return _sentence(clause)


def _format_latent_choice(belief, social_goal, believed_goal):
    belief_text = _sentence(belief) if belief else "Unknown."
    believed_goal_text = _sentence(believed_goal) if believed_goal else "Unknown."
    social_goal_text = _normalize_spaces(social_goal).lower() or "unknown"
    return f"Belief: {belief_text}; Social goal: {social_goal_text}; Believed Goal: {believed_goal_text}"


def _build_belief_latent_choices(info, stem, options):
    social_goal = _parse_social_goal_from_text(stem)
    believed_goal = _infer_goal_statement(info, stem + " " + " ".join(options.values()))
    if social_goal == "unknown" or not believed_goal:
        return None

    choices = {}
    for label, option_text in options.items():
        belief = _belief_from_option_text(option_text)
        if not belief:
            return None
        choices[label] = _format_latent_choice(belief, social_goal, believed_goal)
    return choices


def _build_social_goal_latent_choices(info, stem, options, init_state):
    belief = _select_relevant_state(init_state, info, stem + " " + " ".join(options.values()))
    believed_goal = _infer_goal_statement(info, stem + " " + " ".join(options.values()))
    if not belief or not believed_goal:
        return None

    choices = {}
    for label, option_text in options.items():
        social_goal = _parse_social_goal_from_text(option_text)
        if social_goal == "unknown":
            return None
        choices[label] = _format_latent_choice(belief, social_goal, believed_goal)
    return choices


def _parse_belief_of_goal_option(option_text, info):
    names = list(info.keys())
    other_name = names[1] if len(names) > 1 else "Other person"
    normalized = _normalize_spaces(option_text)
    normalized_lower = normalized.lower()

    if any(token in normalized_lower for token in ["doesn't know", "does not know", "without thinking about what"]):
        match = re.search(r"(.+?)(?: and moves|\.)", normalized, re.IGNORECASE)
        belief_text = match.group(1) if match else normalized
        belief_text = re.sub(r"\bdoesn't\b", "does not", belief_text, flags=re.IGNORECASE)
        return _sentence(belief_text), "indifferent", _sentence(f"{other_name}'s goal is unknown")

    first_clause, _, second_clause = normalized.partition(":")
    belief_match = re.search(r"believed that (.*)$", first_clause, re.IGNORECASE)
    belief = _sentence(belief_match.group(1) if belief_match else first_clause)
    social_goal = _parse_social_goal_from_text(second_clause or normalized)

    wants_match = re.search(
        r"believed that .*? wants to place (?:the |a |an )?(.+?) ((?:inside|in|on|onto|to) .+)$",
        first_clause,
        re.IGNORECASE,
    )
    if wants_match:
        object_name = _strip_articles(wants_match.group(1))
        location = _normalize_spaces(wants_match.group(2))
        believed_goal = _sentence(f"{other_name} wants to place the {object_name} {location}")
        return belief, social_goal, believed_goal

    placed_match = re.search(
        r"believed that .*? placed (?:the |a |an )?(.+?) at (?:his|her|their) desired location",
        first_clause,
        re.IGNORECASE,
    )
    if placed_match:
        object_name = _strip_articles(placed_match.group(1))
        desired_location = _infer_desired_location_from_actions(info.get(other_name, {}).get("action") or [], object_name)
        if desired_location:
            believed_goal = _sentence(f"{other_name} wants to place the {object_name} {desired_location}")
        else:
            believed_goal = _sentence(f"{other_name} wants the {object_name} at the current location")
        return belief, social_goal, believed_goal

    return belief, social_goal, _infer_goal_statement(info, normalized)


def _build_belief_of_goal_latent_choices(info, options):
    choices = {}
    for label, option_text in options.items():
        belief, social_goal, believed_goal = _parse_belief_of_goal_option(option_text, info)
        if not belief or social_goal == "unknown" or not believed_goal:
            return None
        choices[label] = _format_latent_choice(belief, social_goal, believed_goal)
    return choices


def _build_deterministic_latent_choices(info, question, init_state, question_label=None):
    stem, options = _parse_question_options(question)
    if set(options) != {"A", "B", "C"}:
        return None

    resolved_label = _infer_question_label(question_label, stem, options)
    if resolved_label == "belief":
        return _build_belief_latent_choices(info, stem, options)
    if resolved_label == "social_goal":
        return _build_social_goal_latent_choices(info, stem, options, init_state)
    if resolved_label == "belief_of_goal":
        return _build_belief_of_goal_latent_choices(info, options)
    return None


def _model_latent_variable_extraction(info, question, init_state):
    names = list(info.keys())
    if info[names[1]]["action"] is not None and info[names[1]]["utterance"] is None:
        prompt = f"""
        Consider the action of {names[1]} before {names[0]} act. Check where {names[1]} has put the object to help you determine {names[1]}'s desired location for the object.
        Actions: {info[names[1]]["action"]}
        """ + latent_variable_prompt
        latent_variables = _chat(prompt.format(question), max_new_tokens=384)
    else:
        prompt = latent_variable_prompt + """
        State: {}
        """
        latent_variables = _chat(prompt.format(question, init_state), max_new_tokens=384)

    def extract_contents(label, input_string):
        pattern = rf"{label}: (.*?)(?=[A-Z]:|$)"
        match = re.search(pattern, input_string, re.DOTALL)
        return match.group(1).strip() if match else None

    return {
        "A": extract_contents("A", latent_variables),
        "B": extract_contents("B", latent_variables),
        "C": extract_contents("C", latent_variables),
    }


def parse_text_info(text, name):  # parse any kind of action and utterance text
    info_extraction_prompt = """
        You will read a piece of text describing actions of some number of people with distinctive names. You will also have a name, which is the name of the person whom you should pay attention to. Summarize the person's actions and utterance separately in a chronological order. Only include the actions and utterance directly taken by the person in the text, and exclude any previous actions mentioned indirectly. If you cannot find either utterance or actions of the person in the text, leave the corresponding section blank. When reading words like "it", replace it with inferred object or location to make actions clearer. Do not include agent's communication as part of it. Organize your answer in this form:
        Actions:
        ["action one", "action two", "action three", ...]
        ...
        Utterance:
        ["utterance one", "utterance two", "utterance three", ...]
        ...

        Text: {}

        Name: {}
    """
    info = _chat(info_extraction_prompt.format(text, name), max_new_tokens=512)

    actions = _parse_literal_list(_extract_section_text("Actions", info))
    utterance = _parse_literal_list(_extract_section_text("Utterance", info))
    if actions is None or utterance is None:
        repaired_actions, repaired_utterances = _repair_info_lists(info)
        if actions is None:
            actions = repaired_actions
        if utterance is None:
            utterance = repaired_utterances

    action_list = actions if actions else None
    utterance_list = utterance if utterance else None
    return {"action": action_list, "utterance": utterance_list}


def latent_variable_extraction(info, question, question_label=None):
    action_str = ""
    print(info)
    for name in info.keys():
        if info[name]["action"] is not None:
            action_str += f"{name}'s actions:\n"
            for index, action in enumerate(info[name]["action"]):
                action_str += f"{index + 1}: {action}\n"
    init_state = _chat(init_state_prompt.format(action_str), max_new_tokens=256)

    deterministic_choices = _build_deterministic_latent_choices(
        info=info,
        question=question,
        init_state=init_state,
        question_label=question_label,
    )
    if deterministic_choices and all(deterministic_choices.values()):
        return init_state, deterministic_choices

    return init_state, _model_latent_variable_extraction(info, question, init_state)


if __name__ == "__main__":
    pass

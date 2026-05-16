import json
import math
import re

from model_backend import get_default_model
from runtime_hparams import get_enable_thinking_default


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "the",
    "there",
    "is",
    "inside",
    "in",
    "on",
    "onto",
    "at",
    "to",
    "towards",
    "toward",
    "into",
    "from",
    "of",
}
_ABSTRACT_LOCATION_PHRASES = {
    "current location",
    "desired location",
    "goal location",
    "target location",
    "unknown",
    "elsewhere",
}


_BELIEF_OF_GOAL_HINT = (
    "Do not infer that the acting agent knows the other agent's goal only because the final object location "
    "happens to align with or oppose that goal. Help or hinder should be likely only when the action pattern "
    "is better explained by already knowing the other's goal than by an ordinary relocation."
)

_UNKNOWN_GOAL_HINT = (
    "The acting agent does not know the other agent's goal. Do not assume hidden goal knowledge. "
    "An action can still be likely if it reflects an ordinary relocation or the actor's own preferences, "
    "but it should be unlikely only if it specifically requires knowing the other agent's goal in advance."
)


def parse_latent_var(latent_var):
    belief = re.search(r"Belief:\s*(.*?)(?=\; Social goal)", latent_var).group(1)
    social_goal = re.search(r"Social goal:\s*(.*?)(?=\; Believed Goal)", latent_var).group(1)
    believed_goal = re.search(r"Believed Goal:\s*(.*)", latent_var).group(1)
    return {"Belief": belief, "Social Goal": social_goal, "Believed Goal": believed_goal}


def _score_a_probability(evaluation_prompt: str) -> float:
    model = get_default_model()
    probabilities = model.score_candidates(
        [{"role": "system", "content": evaluation_prompt}],
        ["A", "B"],
    )
    return probabilities["A"]


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _strip_leading_articles(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", _normalize_text(text))


def _is_concrete_location(text: str) -> bool:
    normalized = _strip_leading_articles(text).strip(" .,:;")
    if not normalized:
        return False
    if normalized in _ABSTRACT_LOCATION_PHRASES:
        return False
    return not any(
        normalized.startswith(prefix)
        for prefix in ("current location", "desired location", "goal location", "target location")
    )


def _normalize_target_locations(raw_locations) -> list[str]:
    if raw_locations is None:
        return []
    if isinstance(raw_locations, str):
        raw_locations = [raw_locations]
    if not isinstance(raw_locations, list):
        return []

    locations: list[str] = []
    for location in raw_locations:
        normalized = _strip_leading_articles(str(location))
        if normalized and _is_concrete_location(normalized) and normalized not in locations:
            locations.append(normalized)
    return locations


def _extract_target_object(belief: str, believed_goal: str) -> str | None:
    patterns = [
        (believed_goal, r"wants to find (?:a |an |the )?(.+?)(?:\.|$)"),
        (believed_goal, r"wants to get (?:a |an |the )?(.+?)(?:\.|$)"),
        (believed_goal, r"wants to place (?:a |an |the )?(.+?) (?:inside|in|on|onto|at|to)\b"),
        (belief, r"there is (?:a |an |the )?(.+?) (?:inside|in|on|onto|at)\b"),
    ]
    for source_text, pattern in patterns:
        match = re.search(pattern, _normalize_text(source_text), re.IGNORECASE)
        if match:
            return _strip_leading_articles(match.group(1))
    return None


def _extract_target_locations(belief: str, believed_goal: str) -> list[str]:
    patterns = [
        (belief, r"there is (?:a |an |the )?.+? (?:inside|in|on|onto|at) (.+?)(?:\.|$)"),
        (believed_goal, r"wants to place (?:a |an |the )?.+? (?:inside|in|on|onto|at|to) (.+?)(?:\.|$)"),
    ]
    raw_locations: list[str] = []
    for source_text, pattern in patterns:
        match = re.search(pattern, _normalize_text(source_text), re.IGNORECASE)
        if match:
            raw_locations.append(match.group(1))
    return _normalize_target_locations(raw_locations)


def _phrase_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(_normalize_text(text))
        if token not in _STOPWORDS
    }


def _action_matches_phrase(action: str, phrase: str) -> bool:
    normalized_action = _normalize_text(action)
    normalized_phrase = _normalize_text(phrase)
    if normalized_phrase and normalized_phrase in normalized_action:
        return True
    phrase_tokens = _phrase_tokens(phrase)
    if not phrase_tokens:
        return False
    action_tokens = _phrase_tokens(action)
    return phrase_tokens.issubset(action_tokens)


def _extract_action_locations(action: str) -> list[str]:
    normalized_action = _normalize_text(action)
    locations: list[str] = []
    patterns = [
        r"\bfrom (.+)$",
        r"\b(?:inside|in|on|onto|at|to) (.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_action, re.IGNORECASE)
        if match:
            location = _strip_leading_articles(match.group(1).strip(' .,:;'))
            if location and location not in locations:
                locations.append(location)
    return locations


def _action_matches_location(action: str, location: str) -> bool:
    normalized_location = _strip_leading_articles(location).strip(' .,:;')
    if not normalized_location:
        return False
    action_locations = _extract_action_locations(action)
    if not action_locations:
        return False

    location_tokens = list(_phrase_tokens(normalized_location))
    location_token_count = len(location_tokens)
    for action_location in action_locations:
        if action_location == normalized_location:
            return True
        if location_token_count <= 1 and _action_matches_phrase(action_location, normalized_location):
            return True
    return False


def _merge_action_lists(*action_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for actions in action_lists:
        for action in actions:
            if action not in merged:
                merged.append(action)
    return merged


def _filter_relevant_actions(
    actions: list[str] | None,
    target_object: str | None,
    target_locations: list[str],
    include_object_actions_with_location_matches: bool = False,
) -> list[str]:
    if not actions:
        return []

    location_matched_actions: list[str] = []
    object_matched_actions: list[str] = []
    for action in actions:
        if any(_action_matches_location(action, location) for location in target_locations):
            location_matched_actions.append(action)
            continue
        if target_object and _action_matches_phrase(action, target_object):
            object_matched_actions.append(action)

    if location_matched_actions:
        if include_object_actions_with_location_matches:
            return _merge_action_lists(location_matched_actions, object_matched_actions)
        return location_matched_actions
    return object_matched_actions


def _first_utterance(utterances: list[str] | None) -> str:
    if not utterances:
        return ""
    return str(utterances[0]).strip()


def _actions_block(name: str, actions: list[str] | None) -> str:
    lines = [f"{name}'s actions:"]
    if actions:
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.append("- None")
    return "\n".join(lines)


def _is_unknown_goal_case(belief: str, believed_goal: str) -> bool:
    normalized_belief = _normalize_text(belief)
    normalized_goal = _normalize_text(believed_goal)
    return (
        "goal is unknown" in normalized_goal
        or ("does not know" in normalized_belief and "goal" in normalized_belief)
        or ("doesn't know" in normalized_belief and "goal" in normalized_belief)
    )


def _all_actions(actions: list[str] | None) -> list[str]:
    if not actions:
        return []
    return [str(action).strip() for action in actions if str(action).strip()]


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    clipped = [min(max(float(value), 1e-6), 1.0) for value in values]
    return math.exp(sum(math.log(value) for value in clipped) / len(clipped))


def _extract_targets_with_model(
    *,
    belief: str,
    social_goal: str,
    believed_goal: str,
    info,
    main_person: str,
    other_name: str,
    question_context: dict,
) -> tuple[str | None, list[str]]:
    model = get_default_model()
    description = str(question_context.get("description", "") or "").strip()
    question = str(question_context.get("question", "") or "").strip()
    choice_label = str(question_context.get("choice_label", "") or "").strip()
    choice_text = str(question_context.get("choice_text", "") or "").strip()

    messages = [
        {
            "role": "system",
            "content": "You extract a target object and concrete target locations for action scoring. Return JSON only.",
        },
        {
            "role": "user",
            "content": (
                "Infer the key physical target object and concrete target locations for the candidate below.\n"
                "Use the scenario description, the question, the candidate answer text, and the latent variables.\n"
                "Target locations should be concrete places like containers, surfaces, or room-qualified locations.\n"
                "Do not return abstract phrases such as current location, desired location, goal location, target location, or unknown.\n"
                "If no concrete object or location can be inferred, use null or an empty list.\n\n"
                f"Scenario description:\n{description or 'Unknown'}\n\n"
                f"Question:\n{question or 'Unknown'}\n\n"
                f"Candidate choice:\n{choice_label}) {choice_text}\n\n"
                "Latent variables:\n"
                f"Belief: {belief}\n"
                f"Social goal: {social_goal}\n"
                f"Believed Goal: {believed_goal}\n\n"
                f"{_actions_block(main_person, info.get(main_person, {}).get('action'))}\n\n"
                f"{_actions_block(other_name, info.get(other_name, {}).get('action'))}\n\n"
                'Return valid JSON with exactly this schema: {"target_object": null, "target_locations": []}'
            ),
        },
    ]

    data = model.generate_json(
        messages,
        max_new_tokens=256,
        enable_thinking=get_enable_thinking_default(),
    )
    if not isinstance(data, dict):
        return None, []

    target_object = data.get("target_object")
    if target_object is None:
        normalized_object = None
    else:
        normalized_object = _strip_leading_articles(str(target_object))
        if normalized_object in {"", "none", "null", "unknown"}:
            normalized_object = None

    target_locations = _normalize_target_locations(data.get("target_locations"))
    return normalized_object, target_locations


def compute_prob_sequence(
    name_agent_0,
    name_agent_1,
    info,
    main_actions,
    other_actions,
    a1_social_goal,
    a1_belief,
    a1_belief_of_goal,
    question_label="",
    unknown_goal_case=False,
    question_context=None,
):
    question_context = question_context or {}
    description = str(question_context.get("description", "") or "").strip()
    question = str(question_context.get("question", "") or "").strip()
    choice_text = str(question_context.get("choice_text", "") or "").strip()
    visual_summary = str(question_context.get("visual_summary", "") or "").strip()
    other_utterance = _first_utterance(info.get(name_agent_0, {}).get("utterance"))
    main_utterance = _first_utterance(info.get(name_agent_1, {}).get("utterance"))

    evaluation_prompt = f"""
    Decide whether the full observed interaction is jointly consistent with the candidate mental-state hypothesis.
    Respond with only either A or B:

    Candidate hypothesis for {name_agent_1}:
    - social goal: {a1_social_goal}
    - belief: {a1_belief}
    - belief of {name_agent_0}'s goal: {a1_belief_of_goal}

    Scenario description: {description or "Unknown"}
    Question: {question or "Unknown"}
    Candidate answer text: {choice_text or "Unknown"}

    {name_agent_0}'s utterance: {other_utterance or "None"}
    {name_agent_1}'s utterance: {main_utterance or "None"}

    {_actions_block(name_agent_0, other_actions)}

    {_actions_block(name_agent_1, main_actions)}
    """
    if visual_summary:
        evaluation_prompt += f"\nOptional visual summary:\n{visual_summary}\n"
    evaluation_prompt += """
    Judge the interaction as a whole instead of over-weighting one individual action.
    Prefer A when the utterance and the action trajectory together make sense under the candidate hypothesis.
    Prefer B when the overall sequence conflicts with the candidate hypothesis.
    """
    if question_label == "belief_of_goal":
        evaluation_prompt += f"\n{_BELIEF_OF_GOAL_HINT}\n"
    if unknown_goal_case:
        evaluation_prompt += f"\n{_UNKNOWN_GOAL_HINT}\n"
    evaluation_prompt += """
    A) Likely
    B) Unlikely
    """
    return _score_a_probability(evaluation_prompt)


def compute_prob(init_state, latent_var, info, main_person, prompt, question_context=None):
    latent_vars = parse_latent_var(latent_var)
    belief = latent_vars["Belief"]
    social_goal = latent_vars["Social Goal"]
    believed_goal = latent_vars["Believed Goal"]
    names = list(info.keys())
    other_name = [name for name in names if name != main_person][0]
    question_label = str((question_context or {}).get("question_label") or "").strip().lower()
    is_belief_of_goal = question_label == "belief_of_goal"
    unknown_goal_case = is_belief_of_goal and _is_unknown_goal_case(belief, believed_goal)

    if unknown_goal_case:
        target_object = None
        target_locations = []
        main_actions = []
        other_actions = []
    else:
        target_object = _extract_target_object(belief, believed_goal)
        target_locations = _extract_target_locations(belief, believed_goal)
        include_full_object_trajectory = question_label == "belief_of_goal"
        main_actions = _filter_relevant_actions(
            info[main_person]["action"],
            target_object,
            target_locations,
            include_object_actions_with_location_matches=include_full_object_trajectory,
        )
        other_actions = _filter_relevant_actions(
            info[other_name]["action"],
            target_object,
            target_locations,
            include_object_actions_with_location_matches=include_full_object_trajectory,
        )

    needs_model_target_fallback = bool(
        question_context
        and not unknown_goal_case
        and (
            target_object is None
            or not target_locations
            or (info[main_person].get("action") and not main_actions)
        )
    )
    if needs_model_target_fallback:
        fallback_object, fallback_locations = _extract_targets_with_model(
            belief=belief,
            social_goal=social_goal,
            believed_goal=believed_goal,
            info=info,
            main_person=main_person,
            other_name=other_name,
            question_context=question_context,
        )
        if target_object is None and fallback_object:
            target_object = fallback_object
        if fallback_locations:
            merged_locations = list(target_locations)
            for location in fallback_locations:
                if location not in merged_locations:
                    merged_locations.append(location)
            target_locations = merged_locations
        include_full_object_trajectory = question_label == "belief_of_goal"
        main_actions = _filter_relevant_actions(
            info[main_person]["action"],
            target_object,
            target_locations,
            include_object_actions_with_location_matches=include_full_object_trajectory,
        )
        other_actions = _filter_relevant_actions(
            info[other_name]["action"],
            target_object,
            target_locations,
            include_object_actions_with_location_matches=include_full_object_trajectory,
        )
        print(
            "Model target fallback:",
            json.dumps(
                {
                    "target_object": fallback_object,
                    "target_locations": fallback_locations,
                },
                ensure_ascii=False,
            ),
        )

    main_utterance = _first_utterance(info[main_person].get("utterance"))
    other_utterance = _first_utterance(info[other_name].get("utterance"))

    utterance_probability = None
    if main_utterance:
        utterance_probability = compute_prob_utterance(
            other_name,
            main_person,
            other_utterance,
            main_utterance,
            social_goal,
            belief,
            believed_goal,
            None,
            exclude=["Believed_Goal"],
        )
        probability = utterance_probability
    else:
        probability = 1.0

    print(f"Target object: {target_object}")
    print(f"Target locations: {target_locations}")
    print(f"Relevant actions used for {main_person}: {main_actions}")

    action_step_probabilities: list[float] = []
    for index, action in enumerate(main_actions):
        previous_actions = f"{other_name}'s actions:\n"
        for action1 in other_actions:
            previous_actions += action1
            previous_actions += "\n"
        previous_actions += f"{main_person}'s actions:\n"
        for i in range(index):
            previous_actions += main_actions[i]
            previous_actions += "\n"
        prob = compute_prob_action(
            other_name,
            main_person,
            init_state,
            previous_actions,
            action,
            social_goal,
            belief,
            believed_goal,
            question_label=question_label,
            unknown_goal_case=unknown_goal_case,
        )
        print(f"Probability of step {index}: {prob}")
        action_step_probabilities.append(prob)

    action_product = 1.0
    for prob in action_step_probabilities:
        action_product *= prob

    sequence_probability = None
    if question_label == "social_goal":
        sequence_probability = compute_prob_sequence(
            other_name,
            main_person,
            info,
            main_actions,
            other_actions,
            social_goal,
            belief,
            believed_goal,
            question_label=question_label,
            unknown_goal_case=unknown_goal_case,
            question_context=question_context,
        )
        probability = (utterance_probability if utterance_probability is not None else 1.0) * _geometric_mean(
            action_step_probabilities
        ) * sequence_probability
    else:
        for prob in action_step_probabilities:
            probability *= prob

    print(f"Utterance probability: {utterance_probability if utterance_probability is not None else 1.0}")
    print(f"Action product: {action_product}")
    if sequence_probability is not None:
        print(f"Sequence probability: {sequence_probability}")
    print(f"Raw total score: {probability}")
    return probability


def compute_prob_utterance(name_agent_0, name_agent_1, utterance_agent_0, utterance_agent_1, a1_social_goal, a1_belief, a1_belief_of_goal, init_state, exclude=[]):
    evaluation_prompt = f"""
    {name_agent_1}'s social goal: {a1_social_goal}
    {name_agent_1}'s belief: {a1_belief}
    """
    if "Believed_Goal" not in exclude:
        evaluation_prompt += f"{name_agent_1}'s belief of {name_agent_0}'s goal: {a1_belief_of_goal}\n"
    evaluation_prompt += f"{name_agent_0}'s Utterance': {utterance_agent_0}\n"
    if init_state is not None:
        evaluation_prompt += f"Initial state of environment: {init_state}\n"
    evaluation_prompt += f"""
    Based on the information, decide if it is likely for {name_agent_1} to say this word given conditions above. Compare the utterance and the belief of {name_agent_1}. 
    When trying to hinder, {name_agent_1} is likely to give different information with belief. For example, saying that some object is there when {name_agent_1} believe that there is some other things or nothing there, or the object is at a different place.
    Respond with only either A or B:
    {name_agent_1}'s Utterance: {utterance_agent_1}
    A) Likely
    B) Unlikely
    """
    return _score_a_probability(evaluation_prompt)


def compute_prob_action(
    name_agent_0,
    name_agent_1,
    init_state,
    previous_actions,
    a1_action,
    a1_social_goal,
    a1_belief,
    a1_belief_of_goal,
    question_label="",
    unknown_goal_case=False,
):
    evaluation_prompt = f"""
    Decide if {name_agent_1}'s action is likely with the information provided, respond with only either A or B:
    {name_agent_1}'s social goal: {a1_social_goal}
    {name_agent_1}'s belief: {a1_belief}
    {name_agent_1}'s belief of {name_agent_0}'s goal: {a1_belief_of_goal}
    Initial state: {init_state}
    Check {name_agent_0}'s action to get the location of object when {name_agent_1} starts to act. 
    When {name_agent_1} tries to hinder, it's likely to grab object from its believed goal location for other agent, and unlikely to move objects to the believed goal location
    When {name_agent_1} tries to help, it's likely to grab object from somewhere else and put it to believed goal location, and unlikely to grab object from believed goal location
    Walking towards or grabbing from some unrelated location should be considered likely
"""
    if question_label == "belief_of_goal":
        evaluation_prompt += f"\n    {_BELIEF_OF_GOAL_HINT}\n"
    if unknown_goal_case:
        evaluation_prompt += f"\n    {_UNKNOWN_GOAL_HINT}\n"
    evaluation_prompt += f"""
    Previous Actions: {previous_actions}
    {name_agent_1}'s Action: {a1_action}
    A) Likely
    B) Unlikely
    """
    return _score_a_probability(evaluation_prompt)

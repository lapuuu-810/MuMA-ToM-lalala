from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from typing import Any, Iterable, Sequence

import requests

from runtime_hparams import get_enable_thinking_default, get_qwen_api_defaults


_DEFAULT_MODEL: "QwenAPIModel | None" = None


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    decoder = json.JSONDecoder()
    for start, char in enumerate(stripped):
        if char not in "[ {":
            continue
        try:
            _, end = decoder.raw_decode(stripped[start:])
            return stripped[start : start + end]
        except json.JSONDecodeError:
            continue
    return None


def _flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "text"))
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)
        elif block_type in {"image", "image_url"}:
            parts.append("[IMAGE]")
        elif block_type in {"video", "video_url"}:
            parts.append("[VIDEO]")
    return "\n".join(parts).strip()


def _normalize_message_role(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [{"role": message["role"], "content": message.get("content", "")} for message in messages]
    if normalized and not any(message["role"] == "user" for message in normalized):
        normalized[-1] = {
            **normalized[-1],
            "role": "user",
        }
    return normalized


def _image_to_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    format_name = getattr(image, "format", None) or "PNG"
    image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = f"image/{format_name.lower()}"
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content

    normalized: list[dict[str, Any]] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("type", "text"))
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                normalized.append({"type": "text", "text": text})
            continue

        if block_type == "image":
            image = block.get("image")
            if image is not None:
                normalized.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_to_data_url(image),
                        },
                    }
                )
            continue

        if block_type == "image_url":
            image_url = block.get("image_url", {})
            if isinstance(image_url, dict):
                url = str(image_url.get("url", "")).strip()
            else:
                url = str(image_url).strip()
            if url:
                normalized.append({"type": "image_url", "image_url": {"url": url}})
            continue

        if block_type == "video_url":
            video_url = block.get("video_url", {})
            if isinstance(video_url, dict):
                url = str(video_url.get("url", "")).strip()
            else:
                url = str(video_url).strip()
            if url:
                normalized.append({"type": "video_url", "video_url": {"url": url}})
            continue

    return normalized


class QwenAPIModel:
    def __init__(
        self,
        api_key: str,
        model_id: str,
        base_url: str,
        timeout: float = 120.0,
        enable_thinking: bool = False,
        seed: int | None = 1234,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "Missing Qwen API key. Set QWEN_API_KEY or DASHSCOPE_API_KEY before running."
            )

        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.session = requests.Session()

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text.strip()
            if len(body) > 800:
                body = body[:800] + "..."
            raise RuntimeError(
                f"Qwen API request failed with status {response.status_code}: {body}"
            ) from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Qwen API response: {data!r}")
        return data

    def _prepare_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for message in _normalize_message_role(messages):
            normalized.append(
                {
                    "role": message["role"],
                    "content": _normalize_content(message.get("content", "")),
                }
            )
        return normalized

    def _text_only_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = []
        for message in _normalize_message_role(messages):
            normalized.append(
                {
                    "role": message["role"],
                    "content": _flatten_text_content(message.get("content", "")),
                }
            )
        return normalized

    @staticmethod
    def _extract_response_text(response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"Qwen API returned no choices: {response!r}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> str:
        resolved_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if resolved_thinking is not None:
            payload["enable_thinking"] = bool(resolved_thinking)

        response = self._post_chat_completions(payload)
        return self._extract_response_text(response)

    def generate_json(
        self,
        messages: Sequence[dict[str, Any]],
        max_new_tokens: int = 384,
        repair_attempts: int = 2,
        enable_thinking: bool | None = None,
    ) -> object:
        current_messages = list(messages)
        last_text = ""
        for _ in range(repair_attempts + 1):
            last_text = self.chat(
                current_messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            blob = _extract_json_blob(last_text)
            if blob is not None:
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    pass

            current_messages = [
                {
                    "role": "system",
                    "content": "You are a JSON repair tool. Return valid JSON only, with no markdown.",
                },
                {
                    "role": "user",
                    "content": (
                        "Convert the following content into valid JSON while preserving the meaning.\n\n"
                        f"{last_text}"
                    ),
                },
            ]
        raise ValueError(f"Failed to parse JSON from model output: {last_text}")

    def _score_single_token_candidates(
        self,
        messages: Sequence[dict[str, Any]],
        candidates: Sequence[str],
    ) -> dict[str, float]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._text_only_messages(messages),
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": min(5, max(1, len(candidates))),
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        response = self._post_chat_completions(payload)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"Qwen API returned no choices: {response!r}")

        logprob_content = ((choices[0].get("logprobs") or {}).get("content") or [])
        if not logprob_content:
            raise RuntimeError(
                "Qwen API did not return token logprobs. Use a model that supports logprobs, "
                "or update QWEN_API_MODEL to a snapshot model that supports it."
            )

        candidate_logprobs = {candidate: -60.0 for candidate in candidates}
        top_logprobs = logprob_content[0].get("top_logprobs") or []
        for item in top_logprobs:
            token = str(item.get("token", "")).strip()
            logprob = item.get("logprob")
            if token in candidate_logprobs and logprob is not None:
                candidate_logprobs[token] = float(logprob)

        generated_token = str(logprob_content[0].get("token", "")).strip()
        generated_logprob = logprob_content[0].get("logprob")
        if generated_token in candidate_logprobs and generated_logprob is not None:
            candidate_logprobs[generated_token] = max(
                candidate_logprobs[generated_token],
                float(generated_logprob),
            )

        probabilities = _softmax([candidate_logprobs[candidate] for candidate in candidates])
        return dict(zip(candidates, probabilities))

    def _score_with_json(
        self,
        messages: Sequence[dict[str, Any]],
        candidates: Sequence[str],
    ) -> dict[str, float]:
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in self._text_only_messages(messages)
        )
        schema = ", ".join(f'"{candidate}": probability' for candidate in candidates)
        data = self.generate_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. Estimate a normalized probability distribution over the "
                        "provided candidates based on the prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Prompt:\n{prompt}\n\n"
                        f"Candidates: {list(candidates)}\n"
                        f"Return JSON with schema {{{schema}}}. Values must be numbers >= 0 and sum to 1."
                    ),
                },
            ],
            max_new_tokens=256,
            enable_thinking=get_enable_thinking_default(),
        )
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object for candidate scores, got: {data!r}")

        raw_scores: list[float] = []
        for candidate in candidates:
            value = data.get(candidate, 0.0)
            try:
                raw_scores.append(max(0.0, float(value)))
            except (TypeError, ValueError):
                raw_scores.append(0.0)

        total = sum(raw_scores)
        if total <= 0:
            uniform = 1.0 / len(candidates)
            return {candidate: uniform for candidate in candidates}
        return {
            candidate: score / total
            for candidate, score in zip(candidates, raw_scores)
        }

    def score_candidates(
        self,
        messages: Sequence[dict[str, Any]],
        candidates: Sequence[str],
    ) -> dict[str, float]:
        normalized_candidates = [str(candidate).strip() for candidate in candidates]
        if not normalized_candidates:
            return {}

        if all(len(candidate) == 1 for candidate in normalized_candidates):
            try:
                return self._score_single_token_candidates(messages, normalized_candidates)
            except Exception:
                pass
        return self._score_with_json(messages, normalized_candidates)

    def choose_from_letters(
        self,
        messages: Sequence[dict[str, Any]],
        letters: Iterable[str],
    ) -> tuple[str, dict[str, float]]:
        candidates = list(letters)
        probabilities = self.score_candidates(messages, candidates)
        choice = max(probabilities.items(), key=lambda item: item[1])[0]
        return choice, probabilities


def get_default_model() -> QwenAPIModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        defaults = get_qwen_api_defaults()
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
        base_url = os.getenv("QWEN_API_BASE_URL", str(defaults["base_url"]))
        model_id = os.getenv("QWEN_API_MODEL", str(defaults["model_id"]))
        timeout = float(os.getenv("QWEN_API_TIMEOUT", str(defaults["timeout"])))
        enable_thinking = os.getenv(
            "QWEN_API_ENABLE_THINKING",
            "1" if defaults["enable_thinking"] else "0",
        ) == "1"
        seed = os.getenv("QWEN_API_SEED")
        resolved_seed = int(seed) if seed not in {None, ""} else int(defaults["seed"])

        _DEFAULT_MODEL = QwenAPIModel(
            api_key=api_key,
            model_id=model_id,
            base_url=base_url,
            timeout=timeout,
            enable_thinking=enable_thinking,
            seed=resolved_seed,
        )
    return _DEFAULT_MODEL

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from runtime_hparams import get_local_model_defaults


_DEFAULT_MODEL: "LocalChatModel | None" = None


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
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(stripped[start:])
            return stripped[start : start + end]
        except json.JSONDecodeError:
            continue
    return None


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

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
                normalized.append({"type": "image", "image": image})
            continue

        if block_type == "image_url":
            image_url = block.get("image_url", {})
            image = image_url.get("url") if isinstance(image_url, dict) else image_url
            if image:
                normalized.append({"type": "image", "image": image})
            continue

        if block_type == "video":
            video = block.get("video")
            if video is not None:
                normalized.append({"type": "video", "video": video})
            continue

        if block_type == "video_url":
            video_url = block.get("video_url", {})
            video = video_url.get("url") if isinstance(video_url, dict) else video_url
            if video:
                normalized.append({"type": "video", "video": video})

    return normalized


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


class LocalChatModel:
    def __init__(
        self,
        model_id: str,
        model_dir: str | None = None,
        cache_dir: str | None = None,
        revision: str | None = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        enable_thinking: bool = False,
    ) -> None:
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.model_dir = self._resolve_model_dir(
            model_id=model_id,
            model_dir=model_dir,
            cache_dir=cache_dir,
            revision=revision,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        resolved_device_map = self._resolve_device_map(device_map)
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if resolved_device_map is not None:
            model_kwargs["device_map"] = resolved_device_map
        dtype = self._resolve_torch_dtype(torch_dtype)
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_dir,
            **model_kwargs,
        )
        self.model.eval()
        self.input_device = self._infer_input_device()

    @staticmethod
    def _resolve_device_map(device_map: Any) -> Any:
        if device_map is None:
            return None
        if isinstance(device_map, dict):
            return device_map
        if not isinstance(device_map, str):
            return device_map

        normalized = device_map.strip()
        if not normalized:
            return None
        if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return normalized
        if normalized.startswith("{"):
            return json.loads(normalized)
        if normalized.isdigit():
            return {"": int(normalized)}
        if normalized.startswith("cuda:") or normalized in {"cpu", "mps"}:
            return {"": normalized}
        return normalized

    @staticmethod
    def _resolve_model_dir(
        model_id: str,
        model_dir: str | None,
        cache_dir: str | None,
        revision: str | None,
    ) -> str:
        if model_dir:
            path = Path(model_dir)
            if path.exists():
                return str(path)

        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope is not installed. Please run `python -m pip install modelscope` first."
            ) from exc

        kwargs: dict[str, str] = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if revision:
            kwargs["revision"] = revision
        return snapshot_download(model_id, **kwargs)

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str) -> torch.dtype | str:
        if torch_dtype == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if torch_dtype not in mapping:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        return mapping[torch_dtype]

    def _infer_input_device(self) -> torch.device:
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                    return torch.device(device)
        return next(self.model.parameters()).device

    def _normalize_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            normalized.append(
                {
                    "role": message["role"],
                    "content": _normalize_content(message.get("content", "")),
                }
            )
        if normalized and not any(message["role"] == "user" for message in normalized):
            normalized[-1] = {
                **normalized[-1],
                "role": "user",
            }
        return normalized

    def _text_only_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = [
            {
                "role": message["role"],
                "content": _flatten_text_content(message.get("content", "")),
            }
            for message in messages
        ]
        if normalized and not any(message["role"] == "user" for message in normalized):
            normalized[-1] = {
                **normalized[-1],
                "role": "user",
            }
        return normalized

    def _prepare_inputs(
        self,
        messages: Sequence[dict[str, Any]],
        enable_thinking: bool | None,
    ) -> dict[str, Any]:
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        try:
            encoded = self.processor.apply_chat_template(
                self._normalize_messages(messages),
                **template_kwargs,
            )
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            encoded = self.processor.apply_chat_template(
                self._normalize_messages(messages),
                **template_kwargs,
            )

        return {
            key: value.to(self.input_device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> str:
        resolved_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        inputs = self._prepare_inputs(messages, enable_thinking=resolved_thinking)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = [
            output[input_ids.shape[0] :]
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

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

    def score_candidates(
        self,
        messages: Sequence[dict[str, Any]],
        candidates: Sequence[str],
    ) -> dict[str, float]:
        prompt_text = self.tokenizer.apply_chat_template(
            self._text_only_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        prompt_length = prompt_ids.shape[0]

        scores: list[float] = []
        for candidate in candidates:
            full_text = prompt_text + candidate
            full_ids = self.tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.input_device)
            if full_ids.shape[1] <= prompt_length:
                raise ValueError(f"Candidate produced no continuation tokens: {candidate!r}")

            with torch.no_grad():
                logits = self.model(full_ids).logits[0]
                log_probs = torch.log_softmax(logits[:-1], dim=-1)

            total_log_prob = 0.0
            for position in range(prompt_length, full_ids.shape[1]):
                token_id = full_ids[0, position]
                total_log_prob += float(log_probs[position - 1, token_id].item())
            scores.append(total_log_prob)

        probabilities = _softmax(scores)
        return dict(zip(candidates, probabilities))

    def choose_from_letters(
        self,
        messages: Sequence[dict[str, Any]],
        letters: Iterable[str],
    ) -> tuple[str, dict[str, float]]:
        candidates = list(letters)
        probabilities = self.score_candidates(messages, candidates)
        choice = max(probabilities.items(), key=lambda item: item[1])[0]
        return choice, probabilities


def get_default_model() -> LocalChatModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        defaults = get_local_model_defaults()
        model_dir = os.getenv(
            "MUMATOM_MODEL_DIR",
            str(defaults["model_dir"]),
        )
        model_id = os.getenv("MUMATOM_MODEL_ID", str(defaults["model_id"]))
        cache_dir = os.getenv("MUMATOM_MODEL_CACHE_DIR", str(defaults["cache_dir"]))
        revision = os.getenv("MUMATOM_MODEL_REVISION")
        if revision is None:
            revision = defaults["revision"]
        torch_dtype = os.getenv("MUMATOM_TORCH_DTYPE", str(defaults["torch_dtype"]))
        device_map = os.getenv("MUMATOM_DEVICE_MAP", str(defaults["device_map"]))
        enable_thinking = os.getenv(
            "MUMATOM_ENABLE_THINKING",
            "1" if defaults["enable_thinking"] else "0",
        ) == "1"

        resolved_model_dir = model_dir if Path(model_dir).exists() else None
        _DEFAULT_MODEL = LocalChatModel(
            model_id=model_id,
            model_dir=resolved_model_dir,
            cache_dir=cache_dir,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            enable_thinking=enable_thinking,
        )
    return _DEFAULT_MODEL

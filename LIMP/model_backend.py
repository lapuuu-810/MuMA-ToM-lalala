from __future__ import annotations

import os

from runtime_hparams import get_backend_default


def get_backend_name() -> str:
    backend = os.getenv("MUMATOM_BACKEND", get_backend_default()).strip().lower()
    if backend in {"local", "hf", "huggingface"}:
        return "local"
    if backend in {"qwen_api", "api", "dashscope"}:
        return "qwen_api"
    raise ValueError(
        "Unsupported MUMATOM_BACKEND. Expected one of: local, qwen_api."
    )


def get_default_model():
    backend = get_backend_name()
    if backend == "local":
        from local_backend import get_default_model as _get_local_model

        return _get_local_model()

    from qwen_api_backend import get_default_model as _get_qwen_api_model

    return _get_qwen_api_model()

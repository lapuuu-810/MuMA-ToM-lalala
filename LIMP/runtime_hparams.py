from __future__ import annotations

from copy import deepcopy


# ============================================================
# User-editable model hyperparameters
# Edit this section when you want to switch backend/model setup.
# ============================================================

BACKEND = "local"  #qwen_api，local

# RESULTS_FILE = "/data/LPP/cvpr/muti_agent/MuMA-ToM_my/local_runs/qwen3.5-4b-ep42/results.json"
RESULTS_FILE = "/data/LPP/cvpr/muti_agent/MuMA-ToM_my/local_runs/qwen3_5_27b/results_new_16_test.json"
MODEL_EVIDENCE_FILE = "/data/LPP/cvpr/muti_agent/MuMA-ToM_my/output_evidence/qwen3_5_27b/model_evidence_strong.json"
ENABLE_THINKING = False
#export DASHSCOPE_API_KEY="*****"
LOCAL_MODEL_DEFAULTS = {
    "model_id": "Qwen/Qwen3.5-27B",
    "model_dir": "/data/LPP/cvpr/model/Qwen/Qwen3.5-27B",
    "cache_dir": "/data/.cache/modelscope",
    "revision": None,
    "torch_dtype": "float16",
    "device_map": "auto",
    "enable_thinking": ENABLE_THINKING,
}

QWEN_API_DEFAULTS = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model_id": "qwen3.5-plus",
    "timeout": 120.0,
    "enable_thinking": ENABLE_THINKING,
    "seed": 1000,
}


def get_backend_default() -> str:
    return BACKEND


def get_results_file_default() -> str:
    return RESULTS_FILE


def get_model_evidence_file_default() -> str:
    return MODEL_EVIDENCE_FILE


def get_enable_thinking_default() -> bool:
    return ENABLE_THINKING


def get_local_model_defaults() -> dict[str, object]:
    return deepcopy(LOCAL_MODEL_DEFAULTS)


def get_qwen_api_defaults() -> dict[str, object]:
    return deepcopy(QWEN_API_DEFAULTS)

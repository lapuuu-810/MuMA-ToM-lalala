# MuMA-ToM-lalala

本仓库基于原始项目 [MuMA-ToM](https://github.com/SCAI-JHU/MuMA-ToM.git) 修改而来。

原始 MuMA-ToM 提供了一个多模态多智能体 Theory of Mind 推理 benchmark，并提出了基于语言模型的 LIMP 方法。  
本仓库在原始 LIMP 代码基础上，加入了本地 Qwen3.5-27B 推理、动作文件生成与补全、模型证据生成、结果保存和提交格式转换等功能。

本仓库最终复现实验结果为：

```text
Accuracy = 0.9433333333333334
```

---

## 1. Repository

Original repository:

```text
https://github.com/SCAI-JHU/MuMA-ToM.git
```

Our repository:

```text
https://github.com/lapuuu-810/MuMA-ToM-lalala.git
```

---

## 2. Main Changes

相比原始仓库，本仓库主要做了以下修改：

1. 将原始 LIMP 推理流程适配到本地 Qwen3.5-27B 模型；
2. 增加统一模型后端，支持本地模型和 Qwen API；
3. 增加视频动作自动生成脚本；
4. 增加目标动作补全脚本；
5. 增加模型证据生成脚本；
6. 在最终推理中使用置信度过滤后的强证据；
7. 增加结果保存、快速测试、准确率评估和提交格式转换功能。

整体流程如下：

```text
Questions + Texts + Videos
        |
        v
Generate Visual Actions
        |
        v
Supplement Target Actions
        |
        v
Generate Model Evidence
        |
        v
Run LIMP Inference
        |
        v
Save Results / Convert Submission
```

---

## 3. Added / Updated Files

### Updated Files

| File | Description |
|---|---|
| `LIMP/LIMP.py` | 主推理脚本，加入本地 Qwen 后端、模型证据读取、结果保存和 episode 筛选等功能。 |
| `LIMP/compute_prob_GPT.py` | 更新候选答案打分逻辑，使其适配新的模型后端。 |
| `LIMP/text_parsing.py` | 更新文本解析和隐变量抽取逻辑。 |
| `LIMP/visual_action_extraction.py` | 更新视觉动作读取逻辑。 |
| `README.md` | 更新项目说明和运行流程。 |

### Added Files

| File | Description |
|---|---|
| `LIMP/runtime_hparams.py` | 统一管理模型路径、结果路径、证据路径等运行参数。 |
| `LIMP/model_backend.py` | 模型后端统一入口。 |
| `LIMP/local_backend.py` | 本地 Qwen3.5-27B 推理后端。 |
| `LIMP/qwen_api_backend.py` | Qwen API 推理后端。 |
| `LIMP/fill_actions_extracted.py` | 从视频中采样帧并生成视觉动作描述。 |
| `LIMP/supplement_target_actions.py` | 补充缺失的目标物体动作。 |
| `LIMP/generate_model_evidence.py` | 生成模型证据文件。 |
| `LIMP/apply_model_evidence_offline.py` | 离线应用模型证据。 |
| `LIMP/local_limp_benchmark.py` | 本地 benchmark / debug 脚本。 |
| `LIMP/evaluate_results_accuracy.py` | 计算本地结果准确率。 |
| `LIMP/convert_results_to_submission.py` | 将结果转换为提交格式。 |
| `mmtom_environment.txt` | 运行环境记录。 |
| `output_evidence/qwen3_5_27b/model_evidence_strong.json` | 生成的强模型证据文件。 |
| `local_runs/qwen3_5_27b/results_new_16_943.json` | 最终结果文件。 |
| `local_runs/qwen3_5_27b/results_new_16_943_submission.json` | 提交格式结果文件。 |

---

## 4. Environment

推荐使用 Conda 环境：

```bash
conda create -n mmtom python=3.10 -y
conda activate mmtom
```

安装主要依赖：

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate modelscope
pip install decord pillow scipy tqdm requests
```

环境记录文件为：

```text
mmtom_environment.txt
```

---

## 5. Model and Data Paths

所有命令默认从项目根目录 `MuMA-ToM-lalala/` 开始执行。

进入 LIMP 目录：

```bash
cd LIMP
```

默认本地模型路径建议设置为：

```text
../models/Qwen/Qwen3.5-27B
```

如果模型路径不同，可以通过环境变量指定：

```bash
export MUMATOM_MODEL_DIR=../models/Qwen/Qwen3.5-27B
```

需要准备的数据文件如下：

```text
../Files/questions.json
../Files/texts.json
../Files/actions_extracted.json
```

视频文件默认建议放在：

```text
../videos
```

如果视频目录不同，需要在 `LIMP/fill_actions_extracted.py` 中修改对应的视频路径配置。

推荐的项目结构如下：

```text
MuMA-ToM-lalala/
├── LIMP/
├── Files/
│   ├── questions.json
│   ├── texts.json
│   └── actions_extracted.json
├── videos/
├── models/
│   └── Qwen/
│       └── Qwen3.5-27B/
├── output_evidence/
│   └── qwen3_5_27b/
├── local_runs/
│   └── qwen3_5_27b/
├── mmtom_environment.txt
└── README.md
```

---

## 6. Running Pipeline

完整运行流程包括三个步骤：

1. 生成动作文件；
2. 生成模型证据；
3. 运行最终 LIMP 推理。

下面每一步的命令都使用相对路径。

---

### Step 1: Generate Action File

生成或更新：

```text
../Files/actions_extracted.json
```

运行：

```bash
cd LIMP

python fill_actions_extracted.py
python supplement_target_actions.py
```

其中：

- `fill_actions_extracted.py` 用于从视频采样帧并生成视觉动作描述；
- `supplement_target_actions.py` 用于补充缺失的目标物体动作。

---

### Step 2: Generate Model Evidence

生成：

```text
../output_evidence/qwen3_5_27b/model_evidence_strong.json
```

如果当前已经在 `LIMP/` 目录下，运行：

```bash
python generate_model_evidence.py \
  --output ../output_evidence/qwen3_5_27b/model_evidence_strong.json \
  --questions ../Files/questions.json \
  --texts ../Files/texts.json \
  --min-confidence 0.70 \
  --strengths strong
```

---

### Step 3: Run Final LIMP Inference

生成最终结果：

```text
../local_runs/qwen3_5_27b/results_new_16_943.json
```

如果当前已经在 `LIMP/` 目录下，运行：

```bash
MUMATOM_RESULTS_FILE=../local_runs/qwen3_5_27b/results_new_16_943.json \
python LIMP.py
```

完整运行预期准确率：

```text
Accuracy = 0.9433333333333334
```

---

## 7. Quick Test

可以只运行 episode `4005` 进行快速测试。

从项目根目录运行：

```bash
cd LIMP

MUMATOM_EPISODES=4005 \
MUMATOM_RESULTS_FILE=../local_runs/qwen3_5_27b/results_new_16_943_repro_4005.json \
python LIMP.py
```

预期结果：

```text
episode 4005: 4/4 correct
Question 1: B
Question 2: A
Question 3: B
Question 4: C
```

---

## 8. Convert to Submission Format

将内部结果转换为提交格式。

如果当前已经在 `LIMP/` 目录下，运行：

```bash
python convert_results_to_submission.py \
  --input ../local_runs/qwen3_5_27b/results_new_16_943.json \
  --output ../local_runs/qwen3_5_27b/results_new_16_943_submission.json
```

提交文件格式如下：

```json
[
  {
    "scenario_id": 4005,
    "question_id": 1,
    "answer": "B"
  }
]
```

---

## 9. Complete Reproduction Commands

完整复现流程如下。下面命令默认从项目根目录执行，可以直接复制运行：

```bash
cd LIMP

python fill_actions_extracted.py
python supplement_target_actions.py

python generate_model_evidence.py \
  --output ../output_evidence/qwen3_5_27b/model_evidence_strong.json \
  --questions ../Files/questions.json \
  --texts ../Files/texts.json \
  --min-confidence 0.70 \
  --strengths strong

MUMATOM_RESULTS_FILE=../local_runs/qwen3_5_27b/results_new_16_943.json \
python LIMP.py

python convert_results_to_submission.py \
  --input ../local_runs/qwen3_5_27b/results_new_16_943.json \
  --output ../local_runs/qwen3_5_27b/results_new_16_943_submission.json
```

最终生成：

```text
local_runs/qwen3_5_27b/results_new_16_943.json
local_runs/qwen3_5_27b/results_new_16_943_submission.json
```

---

## 10. Runtime Variables

常用环境变量如下：

| Variable | Description |
|---|---|
| `MUMATOM_BACKEND` | 模型后端，支持 `local` 或 `qwen_api`。 |
| `MUMATOM_MODEL_DIR` | 本地 Qwen3.5-27B 模型路径。 |
| `MUMATOM_RESULTS_FILE` | 最终结果保存路径。 |
| `MUMATOM_EPISODES` | 指定运行的 episode。 |
| `MUMATOM_USE_MODEL_EVIDENCE_FILE` | 是否使用模型证据文件。 |
| `MUMATOM_MODEL_EVIDENCE_FILE` | 模型证据文件路径。 |
| `MUMATOM_EVIDENCE_PRIOR_MIN_CONFIDENCE` | 使用证据的最低置信度。 |
| `MUMATOM_EVIDENCE_PRIOR_STRENGTHS` | 使用的证据强度。 |

示例：

```bash
cd LIMP

MUMATOM_BACKEND=local \
MUMATOM_MODEL_DIR=../models/Qwen/Qwen3.5-27B \
MUMATOM_RESULTS_FILE=../local_runs/qwen3_5_27b/results_new_16_943.json \
python LIMP.py
```

如果使用 Qwen API 后端，可以设置：

```bash
cd LIMP

MUMATOM_BACKEND=qwen_api \
python LIMP.py
```

---

## 11. Citation

If you use this project, please cite the original MuMA-ToM paper:

```bibtex
@article{shi2024muma,
  title={MuMA-ToM: Multi-modal Multi-Agent Theory of Mind},
  author={Shi, Haojun and Ye, Suyu and Fang, Xinyu and Jin, Chuanyang and Isik, Leyla and Kuo, Yen-Ling and Shu, Tianmin},
  journal={arXiv preprint arXiv:2408.12574},
  year={2024}
}
```

---

## Acknowledgement

This repository is built upon the original MuMA-ToM codebase. We thank the authors of MuMA-ToM for releasing the benchmark and baseline implementation.

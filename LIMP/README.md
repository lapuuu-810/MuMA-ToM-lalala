# MuMA-ToM LIMP 运行流程

```bash
cd /data/LPP/cvpr/muti_agent/MuMA-ToM_my/LIMP
```

默认模型路径：`/data/LPP/cvpr/model/Qwen/Qwen3.5-27B`

## 1. 生成动作文件

生成 `/data/LPP/cvpr/muti_agent/MuMA-ToM_my/Files/actions_extracted.json`：

```bash
python fill_actions_extracted.py
python supplement_target_actions.py
```

## 2. 生成模型证据

生成 `/data/LPP/cvpr/muti_agent/MuMA-ToM_my/output_evidence/qwen3_5_27b/model_evidence_strong.json`：

```bash
python generate_model_evidence.py \
  --output /data/LPP/cvpr/muti_agent/MuMA-ToM_my/output_evidence/qwen3_5_27b/model_evidence_strong.json \
  --questions /data/LPP/cvpr/muti_agent/MuMA-ToM_my/Files/questions.json \
  --texts /data/LPP/cvpr/muti_agent/MuMA-ToM_my/Files/texts.json \
  --min-confidence 0.70 \
  --strengths strong
```


## 3. 生成最终结果

生成 `/data/LPP/cvpr/muti_agent/MuMA-ToM_my/local_runs/qwen3_5_27b/results_new_16_943.json`：

```bash
MUMATOM_RESULTS_FILE=/data/LPP/cvpr/muti_agent/MuMA-ToM_my/local_runs/qwen3_5_27b/results_new_16_943.json \
python LIMP.py
```

完整运行预期 accuracy 为 `0.9433333333333334`。

## 快速测试

```bash
MUMATOM_EPISODES=4005 \
MUMATOM_RESULTS_FILE=/tmp/results_new_16_943_repro_4005.json \
python LIMP.py
```

预期 episode `4005` 为 `4/4`，预测 `1=B, 2=A, 3=B, 4=C`。

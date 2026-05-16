# MuMA-ToM-lalala: Qwen3.5-27B Enhanced LIMP Pipeline

本仓库基于原始项目 [MuMA-ToM: Multi-modal Multi-Agent Theory of Mind](https://github.com/SCAI-JHU/MuMA-ToM) 修改而来。原始项目提出了 MuMA-ToM benchmark 以及 Language model-based Inverse Multi-agent Planning, 即 LIMP 方法，用于多模态、多智能体 Theory of Mind 推理任务。

在原始 LIMP 框架的基础上，本仓库主要加入了基于 **Qwen3.5-27B** 的本地推理后端、动作文件自动生成与补全、模型证据生成、置信度过滤的证据先验、离线结果处理、结果格式转换和复现实验脚本。

本仓库的完整运行流程可以复现如下结果：

```text
Accuracy: 0.9433333333333334

# Step5：A5 到 A3 的源码迁移资料

分支：`a5toa3`。来源是 `step5-minimal-r3-deepep-20261110` 在 2026-11-19 的代码快照。
本分支提供四仓固定提交、项目脚本和环境依赖记录，供逐步重建 A3 环境。
**这是迁移起点，不是已通过 A3 验收的安装包。** 原始 A5 脚本保持原文归档；没有自动执行安装、训练或编译。

## Conda 分步验证

使用 `/mnt/share/z00988734/.conda/step5` 的完整命令见 [三步验证文档](docs/CONDA-VERIFY.md)。
检查脚本位于 `scripts/checks/`，可在四仓安装前分别检查 CANN、基础 NPU 计算和 GBSA/GBSAG。
安装 torch/torch_npu 的版本需先根据目标 A3 的 CANN 确认；没有默认安装最新版。

## 获取四仓

```sh
git clone --branch a5toa3 https://github.com/SelfDLer/Step5.git
cd Step5
git submodule update --init src/verl src/FSDPTurbo src/vllm src/vllm-ascend
```

四仓是固定 SHA 的 submodule，分支来源均为 `step5-a5`。首次检出通常是 detached HEAD，这能保证版本准确。
复现时不要使用 `git submodule update --remote`，以免随分支移动。

| 仓库 | 托管位置 | 固定提交 |
|---|---|---|
| verl | [SelfDLer/verl](https://github.com/SelfDLer/verl/tree/step5-a5) | `db76ca612ed3b3febf7d35c26a9448c5f0ff09a7` |
| FSDPTurbo | [SelfDLer/FSDPTurbo](https://gitcode.com/SelfDLer/FSDPTurbo/tree/step5-a5) | `14bffd579e36dfca3fbf550be3bc6427b65e2153` |
| vLLM | [SelfDLer/vllm_step_yunyao](https://github.com/SelfDLer/vllm_step_yunyao/tree/step5-a5) | `144a421dff083f73db81ab5bcf9fb0e60f91125d` |
| vLLM Ascend | [SelfDLer/vllm-ascend-step-yunyao](https://github.com/SelfDLer/vllm-ascend-step-yunyao/tree/step5-a5) | `bd4a6ce33465e9364a6e2e06f163a96fb0e132ef` |

FSDPTurbo 按指定地址保存在 GitCode，因此获取全部四仓需要能访问 GitCode。若 A3 严格只能访问 GitHub，仍需另行准备该固定提交的 GitHub 镜像；本分支没有擅自替换它的托管位置。

## 文件范围

- `reference/a5/scripts/`：原项目启动脚本、奖励逻辑、ByteLevel tokenizer 适配和两个 MC2 诊断工具。
- `reference/a5/cann_ops/dispatch/`：A5 dispatch 特化的 C++/头文件、编译脚本及接口参数描述；不包含编译产物。
- `reference/a5-runtime/`：原共享环境的激活脚本与 Python 导入钩子源码，供排查隐藏依赖；不得直接安装进 A3 venv。
- `manifests/sources.json`：四仓 SHA、原文件路径与校验摘要。
- `manifests/a5-installed-packages.json`：原环境已安装包的元数据版本清单，只用于对照，不是可直接用于 A3 的 requirements。
- `docs/A3-MIGRATION.md`：逐步重建顺序和尚需适配的项目。

没有加入 DeepEP 子模块或源码，没有加入已编译算子、wheel、虚拟环境、权重、数据集、日志、profiling、dump 或训练结果。
该分支从全新 Git 历史导出，未继承旧总仓包含的实验结果历史。

## 使用前必读

归档文件保留旧相对路径和 A5 绝对路径。它们位于 `reference/`，**不要直接作为本分支的运行入口执行**。
尤其是 `activate.sh` 和 `sitecustomize.py` 会指向旧环境原生扩展，`rl_352e.sh` 还要求 DeepEP 及 Ascend 950 专用算子挂载。
这些文件为解释 A5 行为而保存，不表示 A3 也应采用这些路径或挂载。

当前目标是关闭 DeepEP 验证精度。原快照的 `run_rl_rmpad.sh` 默认已是 DeepEP dispatcher，迁移时必须显式改用：

```sh
export STEP5_EP_DISPATCHER=custom_native_eager_fused_forward
```

这只改变专家计算路径，不关闭 R3，也不能替代 GBSA/GBSAG。GBSA 前向和 GBSAG 反向仍需 A3 对应包。
完整迁移步骤见 [A3-MIGRATION.md](docs/A3-MIGRATION.md)。

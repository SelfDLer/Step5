# A3 手动重建说明

## 已保存和未验证的边界

保存的是 A5 当前四仓源码和项目辅助源码。四仓已有 Git 提交不代表同一组合已经在 A3 验证。
本次只完成源码归档及静态检查；没有在 A3 安装、编译或执行训练。
不迁移 A5 venv、缓存和原生扩展。权重、tokenizer、processor 和数据由 A3 本地另行提供。

## 逐步执行顺序

1. 确认 A3 驱动/固件和 CANN 的完整版本。激活管理员或同事提供的 CANN 与算子环境；不要套用归档里的 A5 CANN 路径。
2. 按根 README 获取四仓固定提交。确认 A3 可以访问 FSDPTurbo 的 GitCode 地址。构建还可能拉取依赖，需要可访问的包源或本地依赖缓存。
3. 新建独立 Python 环境。参考 A5 版本清单选择与 A3 CANN 配套的 torch/torch_npu、Triton 等版本。不要把 A5 包清单直接批量安装为 A3 锁定版本。
4. 先验证 torch_npu 单卡计算和基础多卡 HCCL。之后验证现有 GBSA、GBSAG、metadata 及 Python wrapper 的接口、布局、BF16/head_dim=192 和梯度。
5. 按各固定版本仓库的构建说明依次安装 vLLM、vLLM Ascend、FSDPTurbo、verl。涉及 C++/Rust/AscendC 的扩展在 A3 重新构建；不要让安装器意外升级已选定的 torch/torch_npu。
6. 从归档脚本提取需要的启动逻辑，创建适合 A3 的正式入口。移除旧共享环境导入钩子，改用目标 venv 中实际安装的包；将 MODEL、TRAIN_FILES、VAL_FILES、RUN_DIR、IP、网卡和并行度改为 A3 配置。
7. 显式使用 `custom_native_eager_fused_forward`，保留 R3。确认关闭后没有无条件 DeepEP 导入。不要执行依赖 DeepEP 及 A5 dispatch 挂载的 `rl_352e.sh`。
8. 先检查配置解析和小规模前向/反向，再做固定输入的训推精度比对。如果比较 A5/A3，两边统一 dispatcher、权重、输入及精度开关；设备拓扑差异需要记录。

每一步分别完成和检查，不提供未经 A3 验证的一键安装命令。环境脚本手动 source，不修改 `.bashrc` 或 `.zshrc`。
后续修改激活脚本时需要分别在 Bash 与 zsh 检查；要求 Bash 的脚本显式用 `bash script.sh`。

## 需要特别核对的旧环境依赖

- 原 `scripts/activate.sh` source 另一个共享环境，并不独立安装四仓。
- 原 Python `.pth`/`sitecustomize.py` 会从旧目录加载 vLLM Rust 扩展、vLLM Ascend 原生扩展并重定向 OPP；仅安装四个 Python 包不能复现这个导入过程。
- 原环境还引用 FLA NPU 算子、GBSA vendor、可选 batch-invariant 环境和 HYBM runtime view；关闭 DeepEP 不会自动取消这些依赖。它们需逐项确认在 A3 是否需要，不能仅凭版本清单认定运行环境相同。
- 原脚本默认包含融合 QK norm/RoPE、indexer norm/RoPE、SwiGLU 等精度相关开关，应在对照中明确固定。
- `reference/a5/cann_ops/dispatch` 是 Ascend 950 的编译参考源码，不是 A3 通用补丁。其 `input.json` 仅包含接口/shape 描述，不含样本数据或 kernel 二进制。
- A5 多机启动器和 MC2 复现工具绑定特定节点、专家数量及采集文件，不能直接用于 A3。诊断所需历史数据未上传。
- 原大规模运行中的 RL Insight 服务和外部 Python 路径未打包。精度验证可先用 console 日志，再单独接入可选可视化。

## 包清单的解释

`manifests/a5-installed-packages.json` 从原 venv 的 `.dist-info/METADATA` 只读提取，不包含 wheel、源码包、token 或包索引凭据。
editable 源码以 `sources.json` 和 submodule SHA 为准；安装元数据中的 vLLM 版本字符串不替代实际源码 SHA。
该清单也可能包含与本次路径无关的可选包，包括 DeepEP。记录其版本不表示本次上传或安装 DeepEP。

# A3 单机 16 卡：生成随机权重，测试 RL

在已通过四仓验证的 A3 环境中，从任意工作目录执行。下面使用示例绝对路径，每一项都可以单独修改。若要更新已检出的 `a5toa3` 分支，可执行 `git -C "/mnt/share/z00988734/workspace/step5-a3/Step5" pull --ff-only`，仓库位置按实际填写。

## 1. 指定输出目录，生成 BF16 权重

```bash
export GEN_SCRIPT='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/rl/make_random_step5_16e.py'
export LAYOUT='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/rl/step5_16e_layout.json'
export TARGET_CONFIG='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/rl/step5_16e_config.json'
export ASSETS='/你的路径/尚未下载完的Step5-BF16目录'
export MODEL='/你的路径/step5-random-16e-bf16'
python "$GEN_SCRIPT" --assets "$ASSETS" --output "$MODEL" --layout "$LAYOUT" --target-config "$TARGET_CONFIG"
```

`ASSETS` 可以是尚在下载的全尺寸 Step5 HF 目录。需要完整的 `config.json`、`tokenizer.json`、`tokenizer_config.json`；如聊天模板单独存放，也请下载 `chat_template.jinja`。生成器检查输入的模型类型、词表和图像 token ID，仅读取配置与 tokenizer 相关文件。原始权重分片和原权重索引均无需下载。

输出采用仓库自带的 sft100 基线结构：16 专家、top-8、92 层文本、47 层视觉。即使输入全尺寸配置的专家数不同，输出也固定为 16。默认 seed 为 42，可用 `--seed` 更改。将 `MODEL` 指向 `ASSETS` 外部的新目录，预留至少 80 GiB。生成前可在同一条 Python 命令末尾加 `--plan`，只检查输入和预计大小。

## 2. 用该 checkpoint 在 A3 测试一个 GRPO step

如果已有环境报 `ModuleNotFoundError: No module named 'qwen_vl_utils'`，在已激活的 Step5 环境中执行以下两行，再用新的 `RUN_DIR` 重跑。新安装时，四仓安装脚本的 `deps` 阶段会安装该包。

```bash
python -m pip install 'qwen-vl-utils==0.0.14'
python -c 'from qwen_vl_utils import fetch_image, fetch_video, process_vision_info; from importlib.metadata import version; print(version("qwen-vl-utils"))'
```

Geo3K reward 使用 `mathruler==0.1.0` 判分。旧环境可用 `python -m pip show mathruler` 检查；若缺失，在当前 Step5 环境补装，并用 verl 的实际判分函数验证。第二条命令应打印 `1.0`。

```bash
python -m pip install 'mathruler==0.1.0'
python -c 'from verl.utils.reward_score.geo3k import acc_reward; print(acc_reward(r"\boxed{3}", "3"))'
```

```bash
export RL_SCRIPT='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/rl/run_rl_rmpad_a3_fp32.sh'
export BYTELEVEL_HELPER='/mnt/share/z00988734/workspace/step5-a3/Step5/reference/a5/scripts/rl_bytelevel_model.py'
export REWARD_MODULE='/mnt/share/z00988734/workspace/step5-a3/Step5/reference/a5/scripts/step5_geo3k_reward.py'
export MODEL='/你的路径/step5-random-16e-bf16'
export TRAIN_FILES='/你的路径/train.parquet'
export VAL_FILES='/你的路径/validation.parquet'
export RUN_DIR='/你的路径/step5-rl-smoke-001'
export RAY_TMPDIR='/tmp/s5-001'
bash "$RL_SCRIPT" smoke
```

`smoke` 使用当前 `MODEL`，训练和 rollout 从同一份权重启动。单机 16 张空闲 NPU，训练 FSDP16/EP16、rollout TP8、关闭 DeepEP。checkpoint 落盘 BF16，Actor 使用 `model_dtype=fp32`，rollout 为 BF16。单步设置为训练 batch 16、每条 prompt 生成 2 个回答、最大回复长度 128。

`MODEL` 填第一步生成的目录，也可以填另一个完整的 HF checkpoint。`RUN_DIR` 是本次运行的工作和输出目录。给每次运行指定新的 `RUN_DIR` 和较短的本机 `RAY_TMPDIR`（Ray 用它创建 Unix socket）。`BYTELEVEL_HELPER` 和 `REWARD_MODULE` 指向迁移仓内的 tokenizer 适配脚本和 reward 模块，两者都在 A3 本机读取。

查看 `RUN_DIR` 中的 `config.txt` 和 `train.log`，以第 1 个训练 step 完成并正常退出为通过。随机模型的 reward 或 advantage 可能为零。

同一 RL 脚本也支持 `config`（仅导出性能基线配置）和 `train`（按原性能基线训练，默认 100 步）。

### Ray 等待 16 NPU、48 CPU

verl 为每张 NPU 申请 3 个 Ray 逻辑 CPU，单机 16 卡需要同时放置 16 个 bundle，共 48 个 CPU。其他 Ray 进程还可能需要 CPU。旧脚本把整个节点限制为 32 个逻辑 CPU。当前脚本使用 Ray 自动检测的节点 CPU 数量。

若旧作业仍在等待，在同一作业环境的另一终端运行 `ray status`，查看节点的 CPU、NPU 总量、已用量和待调度请求。结束自己的前台作业（Ctrl+C），在原终端使用新的输出目录和 Ray 临时目录重跑。以下命令也适用于旧脚本：

```bash
export RUN_DIR='/你的路径/step5-rl-smoke-002'
export RAY_TMPDIR='/tmp/s5-002'
bash "$RL_SCRIPT" smoke ray_kwargs.ray_init.num_cpus=null
```

若自动检测后单节点仍无法同时提供 16 个空闲 NPU 和 48 个空闲逻辑 CPU，请检查作业或容器的 CPU 配额、NPU 分配和其他任务占用；在资源分配足够后重跑。

### vLLM 开发版本被报为低于 0.18.0

此分支的 verl 先判断当前进程的 `is_npu_available`。已安装 vLLM 且 NPU 可用时，它直接加载 vLLM。若版本号显示为 `0.1.dev…empty` 并报 “SGLang also not Found”，请在报错所用的 A3 环境和终端检查：

```bash
python -P - <<'PY'
import os, sys
from importlib.metadata import version
from verl.utils import device
print("python:", sys.executable, "verl device module:", device.__file__)
print("vllm:", version("vllm"), "verl NPU:", device.is_npu_available)
for key in ("TORCH_DEVICE_BACKEND_AUTOLOAD", "ASCEND_RT_VISIBLE_DEVICES"):
    print(key, repr(os.getenv(key)))
import torch_npu
import torch
print("after explicit torch_npu import:", torch.npu.is_available(), "count:", torch.npu.device_count())
PY
```

* 若 `verl NPU: False`，先看显式导入是否报缺少 CANN 动态库，并确认 Conda、CANN 和可见卡列表。若设置了 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，执行 `unset TORCH_DEVICE_BACKEND_AUTOLOAD`，再用新的 Python 进程复查。单机 16 卡启动前应看到 `verl NPU: True` 和 `count: 16`。
* 若主进程检查通过，可用 `python -P -c 'import verl.third_party.vllm as v; print(v.__file__)'` 验证相同导入。若只有 Ray worker 报错，旧基线的共享激活脚本设置了 `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`，使 Ray 保留启动器的可见卡列表。当前 A3 单机独占 16 卡入口已设置此变量并传给 worker。使用更新后的脚本时，结束旧作业、换新的 `RUN_DIR` 和 `RAY_TMPDIR` 后重跑；使用旧脚本时，在启动前手动执行 `export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`。

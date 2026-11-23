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

### Rollout worker 报错缺少 `flash_attn_npu_3`

固定版本的 vllm-ascend 在 `VLLM_STEP4_ENABLE_FA3` 未设置时默认取 `1`，初始化 attention backend 时强制检查 `flash_attn_npu_3`。基线设置为 `0`，由 vllm-ascend 按模型和硬件选择后端。更新后的启动器会在本机和 Ray worker 中显式设置 `0`。

保留已有的 `MODEL` 和数据路径，结束上一次作业，用新的绝对路径重跑：

```bash
export RUN_DIR='/你的路径/step5-rl-smoke-004'
export RAY_TMPDIR='/tmp/s5-004'
bash "$RL_SCRIPT" smoke
```

若暂时使用旧脚本，请将最后一行换成下面这条，同时指定启动进程与 Ray worker 的值：

```bash
VLLM_STEP4_ENABLE_FA3=0 bash "$RL_SCRIPT" smoke '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_STEP4_ENABLE_FA3="0"'
```

### 单机 FP32 基线的激活环境对照

以下对照覆盖 sft100 单机入口实际加载的项目 `scripts/activate.sh`、共享运行时的 `activate.sh`、batch-invariant 和 HYBM 子脚本，以及 verl 的 PPO Ray 默认环境。表中 A3 一列描述代码行为。A3 作业所在 shell、容器或 CANN 9.3.0 的 `set_env.sh` 也可能已设置同名变量。

| 项目 | 已跑通基线 | A3 单机入口及影响 |
|---|---|---|
| Python 包搜索 | `PYTHONNOUSERSITE=1` | 启动器未显式设置。可能影响是否从用户的 site-packages 加载包。 |
| CPU 线程 | `OMP_NUM_THREADS` 默认 `4`，已有非空值优先 | 启动器未显式设置。做性能对比时记录训练进程和 worker 的实际线程设置。 |
| HCCL 建链 | `HCCL_CONNECT_TIMEOUT` 默认 `1800` 秒，已有非空值优先 | 启动器未显式设置。影响通信建链的等待时间。 |
| HCCL 端口 | `HCCL_NPU_SOCKET_PORT_RANGE=auto` | 父环境未设置时，verl 的 PPO Ray 环境给 worker 配置 NPU 和 host 两个端口范围为 `auto`。 |
| rollout 和可见卡 | `VLLM_STEP4_ENABLE_FA3=0`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` | A3 入口已设置并传给 Ray worker。 |
| 临时目录和缓存 | `TMPDIR=/tmp`。共享运行时指定节点本地的 `XDG_CACHE_HOME`、`TORCH_EXTENSIONS_DIR`、`TRITON_CACHE_DIR`、`VLLM_CACHE_ROOT`；RL 另设本次作业的 `RAY_TMPDIR` | A3 要求用户指定 `RAY_TMPDIR`，其余沿用现场环境或库默认值。对比首次编译耗时要区分冷缓存和热缓存。 |
| CANN、Python 和算子路径 | `CANN_ENV_SCRIPT`、`ASCEND_HOME_PATH`、`PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH`、`ASCEND_CUSTOM_OPP_PATH` 包含基线的 CANN 9.2、bisheng、GBSA、FLA、自定义算子和 Python 库目录 | A3 使用本机 CANN 9.3.0、Conda、四仓 editable 安装和已验证的算子目录。vLLM Ascend 的 bootstrap 会在自己的 vendor 目录存在时添加 OPP 路径。GBSA/GBSAG 仍以 A3 实际执行检查为准。 |
| 旧工程和可选组件 | `STEP5_HOST_ENV`、`STEP5_PYTHON`、`STEP5_RE_ROOT`、`STEP5_R3_ROOT` 用于旧工程定位。可选 batch-invariant 脚本追加 vendor 路径，HYBM 脚本替换 `ASCEND_HOME_PATH`，DeepEP 增加 `SHMEM_ROOT` 及库路径 | 本次 A3 使用独立 Conda 和用户指定的路径，DeepEP、batch invariant 均关闭。上述 A5 私有路径不适用于这个入口。 |

在激活过的 A3 作业终端执行，查看启动前的实际值。训练日志中的 `ray init kwargs` 可另查提交给 Ray 的显式设置。

```bash
python - <<'PY'
import os, site, sys
print("Python:", sys.executable, "user site enabled:", site.ENABLE_USER_SITE)
for key in ("PYTHONNOUSERSITE", "OMP_NUM_THREADS", "HCCL_CONNECT_TIMEOUT",
            "HCCL_NPU_SOCKET_PORT_RANGE", "TMPDIR", "RAY_TMPDIR", "XDG_CACHE_HOME",
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
            "ASCEND_HOME_PATH", "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH", "LD_LIBRARY_PATH"):
    print(key, repr(os.getenv(key)))
PY
```

如果要复用基线的三项通用设置，在同一终端启动 RL 前执行：

```bash
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
```

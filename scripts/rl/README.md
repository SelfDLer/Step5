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

### A3 单卡预检与 indexer Norm + RoPE 编译错误

若 rollout 日志报 `_indexer_qkz_norm_rope_kernel`、`unsupported BlockData producer 'arith.trunci'`，本次失败点是 Triton 编译 int64 位置索引转 int32 的指令。当前 `smoke` 默认给 rollout 设置 `VLLM_STEP4_FUSE_INDEXER_NORM_ROPE=0`，改用模型自带的分步 norm 和 RoPE。训练侧的融合设置、GBSA 和并行配置沿用原值。`train` 默认沿用原融合路径，方便保留性能基线。若要用备用路径测性能，请记录这个差异，并在两端指定相同开关。

若暂用旧脚本，在已激活的环境中保留之前的模型和数据路径，结束旧作业并用新目录重跑：

```bash
export RUN_DIR='/你的路径/step5-rl-smoke-005'
export RAY_TMPDIR='/tmp/s5-005'
VLLM_STEP4_FUSE_INDEXER_NORM_ROPE=0 bash "$RL_SCRIPT" smoke '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_STEP4_FUSE_INDEXER_NORM_ROPE="0"'
```

单卡预检在独立终端执行。指定一张空闲卡、检查脚本和 vllm-ascend 仓库的绝对路径。该检查直接执行融合内核，与 PyTorch 参考比较 Q/K，并检查 Z 原样传递。它不需要模型权重、数据或 Ray。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export INDEXER_CHECK='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/checks/04_check_indexer_rope.py'
export ASCEND_SRC='/mnt/share/z00988734/workspace/step5-a3/Step5/src/vllm-ascend'
python "$INDEXER_CHECK" --ascend-src "$ASCEND_SRC" --positions int64 --lengths 1 33 129
# 独立运行 int32 对照，判断是否与索引类型相关
python "$INDEXER_CHECK" --ascend-src "$ASCEND_SRC" --positions int32 --lengths 1 33 129
ASCEND_OPS="$ASCEND_SRC/tests/e2e/nightly/single_node/ops"
VLLM_USE_MODELSCOPE=false python -m pytest --confcutdir="$ASCEND_OPS" -q -rs "$ASCEND_OPS/singlecard_ops/triton/test_step5_fused_add_rms_norm.py"
```

当前 16 专家模型在 TP8 下每个 rank 的 indexer 有 4 个 Q 头、1 个 K 头，head dim 256、RoPE dim 32。旧预检将独立的 Q/K/Z 切片传给内核，行跨度为 1536 个元素。rollout 从包含主 QKV、indexer Q/K/Z 和 gate 的同一投影中切片，行跨度为 3464。int64 通过只对实际测试的布局、长度、编译器和特化有效。上面两条命令用于手工缩小长度范围。完整排查请用下节默认矩阵。

### 测试实际使用的算子

**第一步，在 A3 一张空闲卡上跑完整预检。** `MODEL` 为本次 RL 使用的 HF 模型绝对路径。在已激活的环境运行下面的命令，按本机情况填写卡号和新的输出目录。脚本会先从当前 vllm-ascend 源码枚举 Step5 自定义 Triton 内核和启动点，再分组执行数值和编译检查。一组失败会继续尝试其他组。单卡终端中的可见卡设置和 16 卡作业终端应各自独立。

```bash
export OP_CHECK='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/checks/05_rl_ops.py'
export ASCEND_SRC='/mnt/share/z00988734/workspace/step5-a3/Step5/src/vllm-ascend'
export FSDP_SRC='/mnt/share/z00988734/workspace/step5-a3/Step5/src/FSDPTurbo'
export OP_OUT='/你的路径/a3-single-op-check-002'
python "$OP_CHECK" preflight --device 0 --ascend-src "$ASCEND_SRC" --fsdp-src "$FSDP_SRC" --model-config "$MODEL/config.json" --tp 8 --out "$OP_OUT"
```

* 训练侧包含 GBSA/GBSAG 前反向（1、2、4 个 KV 头，7/8/9 和 127/128/129 token），带 padding 和 packed TND 的长序列 GBSA、稠密/滑窗 TND 前反向、grouped matmul 梯度、Norm/RoPE 和 SwiGLU 梯度。MoE 用 16 专家、top-8 检查 actor 路由权重、专家 ID 和梯度，覆盖原生选择、全部 replay 和混合 replay。
* rollout 侧检查融合 indexer、主 attention 的 Q/K Norm + RoPE、paged indexer logits、稀疏 decode 元数据、CSA prefill/decode 与交接、summary 状态、融合残差 RMSNorm 和 SwiGLU。单独调用 A3 的 MoE gating 实现，用 NumPy 参考值校验 16 专家、top-8、scale 3 的专家 ID 和权重。每组使用新的 Triton 缓存。int64 和 int32 独立运行，indexer 和 CSA flush 还要求找到对应的完整编译产物。
* indexer 默认跑 1、8、32、33、129、1024、1152、3072 个 token，在 33 token 上还逐个调用固定源码注册的 11 个候选分块并与参考输出比较。日志分别记录生产入口和各个候选的结果。若只有某个候选失败，需要结合真实运行的编译信息判断这次 rollout 是否会选中它。单卡 eager 测试不经过多卡 rollout 的图捕获和回放。
* `results.json` 随每组完成更新。跳过的 pytest 用例计为 `INCOMPLETE`，尚未处理的组显示 `NOT_RUN`。`preflight-kernels.csv` 把每个发现的启动点与本次单卡检查对照。优先看 `NO_PREFLIGHT_EVIDENCE`、`NO_PASSING_CASE`、`AMBIGUOUS_NAME`。`COMPILED_IN_PASSING_CASE` 代表在通过组的缓存里见过该名称；同名可能有多种特化。`kernel-map.csv` 给出源码位置、启动参数、constexpr 以及静态可解析的测试调用。

#### 单独验证 IndexerRegionScore

生产 rollout 通过 `indexer_score_topk` 调用 `cann_ops_transformer` 的 `IndexerRegionScore`，输出可见历史 region 的分数和排序后的 top-k ID。旧预检中的 `indexer_i64/i32` 验证前面的 Norm/RoPE，`paged_indexer_logits` 验证独立的 Triton 打分实现；两者的结果都无法确认这个 ACLNN 算子在 A3 上可用。

更新 `a5toa3` 后，在已激活的 A3 环境单独运行下面三组。每组使用一个新的输出目录；完整 `preflight` 也会执行这三组。仅需一张空闲卡和两个已安装的源码目录。

```bash
git -C /mnt/share/z00988734/workspace/step5-a3/Step5 pull --ff-only
export OP_CHECK='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/checks/05_rl_ops.py'
export ASCEND_SRC='/mnt/share/z00988734/workspace/step5-a3/Step5/src/vllm-ascend'
export FSDP_SRC='/mnt/share/z00988734/workspace/step5-a3/Step5/src/FSDPTurbo'
export IRS_OUT='/你的路径/a3-indexer-region-score-001'
export OP_DEVICE=0
for op_case in indexer_region_score indexer_region_score_long indexer_region_score_vs_triton; do
    python "$OP_CHECK" preflight --only "$op_case" --device "$OP_DEVICE" --ascend-src "$ASCEND_SRC" --fsdp-src "$FSDP_SRC" --out "$IRS_OUT/$op_case" || echo "请查看 $IRS_OUT/$op_case/$op_case.log"
done
```

* `indexer_region_score` 调用迁移分支的生产包装函数，检查 TP8 单 rank 输入、3072-token 配置对应的 384 个 region 容量，以及整批短 prefill 没有完整历史 region 的边界。top-k 为 512。
* `indexer_region_score_long` 复用固定源码中 decode、prefill 和确定性用例。长输入使可见 region 数超过 512，可以检查实际筛选；检查项包括逻辑 ID 升序、因果可见范围和 `-1` 填充。
* `indexer_region_score_vs_triton` 复用固定源码的两项数值对照，只比较因果可见区域的分数和选中 ID。若这一组失败，请按 traceback 区分 ACLNN 调用、Triton 参考实现和结果断言。

#### 旧日志出现 ModelScope 或 snapshot_download

固定版 vllm-ascend 的 `tests/e2e/conftest.py` 在模块顶层导入 `snapshot_download`，供 LoRA 端到端测试的 fixture 使用。若 traceback 停在这行并报 `No module named 'modelscope'`，pytest 在收集测试前就退出了，这一行没有执行下载。旧预检中直接从 E2E 目录收集的三个组是 `rollout_swiglustep`、`decode_sparse_meta` 和 `fused_add_norm`。新版从 `single_node/ops` 开始加载 pytest 配置，保留设备初始化和自定义算子注册。生成在输出目录的单卡用例也限定了配置搜索范围。预检子进程设置 `VLLM_USE_MODELSCOPE=false`，因为这些用例只使用本地配置和测试张量。

先更新已检出的 `a5toa3` 分支，再用下面的命令逐项重跑旧日志里确实缺少 ModelScope 的组。沿用第一步的 `OP_CHECK`、`ASCEND_SRC`、`FSDP_SRC`、`MODEL`，把 `OP_OUT` 指向**旧预检目录**。每组结果写入新的独立目录。

```bash
git -C /mnt/share/z00988734/workspace/step5-a3/Step5 pull --ff-only
export OP_OUT='/你的路径/旧预检目录'
export RETRY_OUT='/你的路径/新的重跑目录'
export OP_DEVICE=0
for log in "$OP_OUT"/*.log; do
    grep -Fq "No module named 'modelscope'" "$log" || continue
    op_case=$(basename "$log" .log)
    python "$OP_CHECK" preflight --only "$op_case" --device "$OP_DEVICE" --ascend-src "$ASCEND_SRC" --fsdp-src "$FSDP_SRC" --model-config "$MODEL/config.json" --tp 8 --out "$RETRY_OUT/$op_case" || echo "请查看 $RETRY_OUT/$op_case/$op_case.log"
done
```

新版 pytest 日志末尾记录退出码和阶段。仅收集、fixture 或报告不完整时显示 `INCOMPLETE`；有用例在调用阶段失败时保留 `FAIL`。判断具体算子需看该用例的 traceback。若日志已经进入 `snapshot_download(...)` 函数体，请按首个调用它的文件和行号另行定位。

**单独定位 CSA decode 编译器退出。** `AutoBlockify disabled ... atomic operations` 表示该优化已被编译器关闭。`bishengir-compile ... --target=Ascend910_9382 ... SIGABRT` 表示编译器子进程异常退出。多进程日志交错，排查时同时保留出错 worker 的完整 traceback 和编译器 stderr。若怀疑 `_csa_decode_stage_flush_kernel`，用空闲卡执行：

```bash
export CSA_OUT='/你的路径/a3-csa-decode-001'
python "$OP_CHECK" preflight --only csa_decode_flush --device 0 --ascend-src "$ASCEND_SRC" --fsdp-src "$FSDP_SRC" --out "$CSA_OUT"
cat "$CSA_OUT/environment.json"
cat "$CSA_OUT/csa_decode_flush.log"
```

该用例也纳入默认预检。它复用固定仓库的参考测试，仅将调用切到 Step5 实现，使用独立冷缓存检查单行 slot 分配、staging、8-token region flush、FP8 输出和 slot 释放。若同样 SIGABRT，保留该目录；在独立 A3 环境中一次只改一个与本机 CANN 配套的 Triton 构建，用相同代码和输入重跑。若通过，继续提取 TP4 异常链中的内核名、实际 `rows/live_rows`、`ACTIVE_CAPACITY`、dtype、stride 和编译器断言，单行测试不能代表这些线上特化。对比版本时同时记录驱动、固件、CANN 完整构建号和日志中实际调用的编译器版本。

**第二步：运行 16 卡真实 GRPO 并采集算子。** 在 16 卡作业终端设置上文所列的 `RL_SCRIPT`、`MODEL`、数据和辅助文件绝对路径。`ops` 沿用 `train` 的 batch、并行和 token 配置，默认只跑一个训练步，跳过定期保存和验证。训练和 rollout 各启用全部 rank 的 NPU profiler。确认输出盘能容纳 16 卡两侧的原始 trace 和解析结果；本模式的时间和显存数据包含采集开销。

```bash
export RUN_DIR='/你的路径/a3-rl-op-coverage-001'
export RAY_TMPDIR='/tmp/s5-op-001'
export OP_CHECK='/mnt/share/z00988734/workspace/step5-a3/Step5/scripts/checks/05_rl_ops.py'
export ASCEND_SRC='/mnt/share/z00988734/workspace/step5-a3/Step5/src/vllm-ascend'
export OP_OUT='/你的路径/a3-single-op-check-002'
# 测融合基线时显式启用，与单卡 indexer 对照；训练失败也继续生成报告。
VLLM_STEP4_FUSE_INDEXER_NORM_ROPE=1 bash "$RL_SCRIPT" ops
python "$OP_CHECK" inventory --run-dir "$RUN_DIR" --ascend-src "$ASCEND_SRC" --preflight-dir "$OP_OUT"
```

若融合 indexer 阻断整步，另选新的 `RUN_DIR` 和 `RAY_TMPDIR`，用 `VLLM_STEP4_FUSE_INDEXER_NORM_ROPE=0 bash "$RL_SCRIPT" ops` 跑分步路径，并对该次运行再执行 inventory。分别保留两次报告。要覆盖视觉路径，训练 batch 必须实际包含图片。要覆盖长序列、不同序列边界和 decode，请用相应样本并确认 rollout 实际生成了多个 token。权重同步、反向、优化器和 TP/EP 通信随真实单步执行。默认关闭的 DeepEP、batch invariant 和样本未触发的分支不计入覆盖。

**读取结果。** `operators.csv` 按 actor/rollout、host/kernel/HCCL、设备和算子列出观察次数；shape、dtype 在 profiler 提供时记录。原始记录分别在 `profiles/actor` 和 `profiles/rollout`。`STEP_TRACED` 要求 trainer 正常退出、日志有第 1 步指标、两侧均有 16 个设备的 kernel，以及 host、HCCL 记录。缺文件、中途失败或只采到一侧会输出 `PARTIAL` 并保留已观察结果。profiler 的执行记录不能替代逐算子的数值参考测试。

`ops` 给本次作业及 Ray worker 设置独立的 `RUN_DIR/triton-cache`。`triton-coverage.csv` 比较本次作业与同一 A3 环境单卡预检的编译键。`COMPLETE + EXACT_KEY` 表示同一特化已在通过的用例组中编译过。`NAME_ONLY` 表示同名但编译键不同，`UNTESTED` 表示通过组未见该内核，`PARTIAL_ARTIFACTS` 表示现场只留下部分文件。编译中止可能早于首次写缓存，此时还要看失败 worker 的完整 traceback。`NO_CACHE_EVIDENCE` 表示无可比较产物。有缺口时 inventory 以非零状态退出并保留 CSV。算子执行以 profiler 为准，缓存可能包括 autotune 尝试的候选。A5 和 A3 的编译目标不同，编译键应在同一台 A3 上对照。

`kernel-map-vs-profile.csv` 把源码内核映射到 rollout 的设备记录。同名歧义、未解析调用和未匹配项都会保留。静态清单仅覆盖当前仓内 `vllm_ascend.models.step5` 的 Triton 定义；外部 CANN 算子和 HCCL 通过数值用例或真实单步采集观察。本模式包含冷编译与 profiler 开销，性能对比使用正常的 `train` 模式。

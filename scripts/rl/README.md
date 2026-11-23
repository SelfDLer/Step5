# A3 单机 16 卡：用 16 专家随机权重跑一个 GRPO step

在已经通过四仓 `verify` 的 A3 环境中执行。随机 checkpoint 以 BF16 保存；Actor 加载时使用 `model_dtype=fp32`，rollout 使用 BF16。默认训练 FSDP16/EP16、rollout TP8（两个副本），关闭 DeepEP。

## 1. 生成本地路径配置

```bash
cd /mnt/share/z00988734/workspace/step5-a3/Step5
git switch a5toa3
git pull --ff-only
bash scripts/rl/random_16e_smoke.sh init
vi .local/random-16e-a3.env
```

在文件里替换成 A3 上的实际路径。取消 `MODEL` 行的注释即可指定生成目录：

```bash
ASSETS='/mnt/share/z00988734/models/step5-original'
MODEL='/mnt/share/z00988734/models/step5-random-16e-bf16-seed42'
TRAIN_FILES='/你的数据目录/train.parquet'
VAL_FILES='/你的数据目录/validation.parquet'
```

`ASSETS` 可以直接指向已有权重的原模型目录。生成器读取与 sft100 16 专家基线匹配的 `config.json`、`tokenizer.json`、`tokenizer_config.json`，并复制已有的聊天模板等配套文件，跳过原权重。`MODEL` 是新随机 checkpoint 的输出，放在 `ASSETS` 外部，首次生成时由脚本创建。留空时默认使用仓库下的 `.local/random-16e-bf16-seed42/`。两份 Parquet 应符合当前 verl 入口和奖励函数的输入格式。

## 2. 生成并验证（逐条执行）

**检查布局并生成权重。** `plan` 只打印生成计划；`generate` 在 CPU 上写入约 78.7 GiB 权重，建议目标文件系统至少有 80 GiB 可用空间。若已生成同一份 BF16 checkpoint，直接执行下一组命令。

```bash
bash scripts/rl/random_16e_smoke.sh plan
bash scripts/rl/random_16e_smoke.sh generate
```

**检查配置并训练一步。** 用同一个新的 `RUN_DIR` 保存配置和日志。在 16 张卡空闲时执行 `train`。

```bash
export RUN_DIR="$PWD/outputs/random-16e-a3-$(date +%Y%m%d-%H%M%S)"
bash scripts/rl/random_16e_smoke.sh config
bash scripts/rl/random_16e_smoke.sh train
```

在 `$RUN_DIR/config.txt` 查看解析后的配置，在 `$RUN_DIR/train.log` 确认第 1 个训练 step 完成并正常退出。随机模型可能得到相同奖励，reward 或 advantage 为零也可能正常。

## 一条命令完成

填好第 1 步的路径后，也可以用下面的命令代替第 2 步。`run` 会生成或复用当前 seed 的随机 checkpoint，依次执行 `config` 和 `train`。若此前已训练过，给本次运行指定新的日志目录。

```bash
RUN_DIR="$PWD/outputs/random-16e-a3-$(date +%Y%m%d-%H%M%S)" bash scripts/rl/random_16e_smoke.sh run
```

## 用已有权重跑 A3 性能基线

随机权重单步测试缩短了回复长度和训练步数。要使用原 FP32 单机对照参数进行性能比较，请直接为性能脚本设置环境变量。这里的 `MODEL` 指向**实际要加载的 HF checkpoint**。

```bash
export MODEL='/你的模型目录/Step5-HF'
export TRAIN_FILES='/你的数据目录/train.parquet'
export VAL_FILES='/你的数据目录/validation.parquet'
export RUN_DIR="$PWD/outputs/a3-fp32-$(date +%Y%m%d-%H%M%S)"
bash scripts/rl/run_rl_rmpad_a3_fp32.sh config
bash scripts/rl/run_rl_rmpad_a3_fp32.sh train
```

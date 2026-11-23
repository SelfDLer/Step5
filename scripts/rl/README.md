# A3 单机 16 卡：生成随机权重，测试 RL

在已通过四仓验证的 A3 环境、Step5 仓库根目录执行。第一次使用请先在 `a5toa3` 分支运行 `git pull --ff-only`。

## 1. 指定输出目录，生成 BF16 权重

```bash
export ASSETS='/你的路径/尚未下载完的Step5-BF16目录'
export MODEL='/你的路径/step5-random-16e-bf16'
python scripts/rl/make_random_step5_16e.py --assets "$ASSETS" --output "$MODEL"
```

`ASSETS` 可以是尚在下载的全尺寸 Step5 HF 目录。需要完整的 `config.json`、`tokenizer.json`、`tokenizer_config.json`；如聊天模板单独存放，也请下载 `chat_template.jinja`。生成器检查输入的模型类型、词表和图像 token ID，仅读取配置与 tokenizer 相关文件。原始权重分片和原权重索引均无需下载。

输出采用仓库自带的 sft100 基线结构：16 专家、top-8、92 层文本、47 层视觉。即使输入全尺寸配置的专家数不同，输出也固定为 16。默认 seed 为 42，可用 `--seed` 更改。将 `MODEL` 指向 `ASSETS` 外部的新目录，预留至少 80 GiB。生成前可在同一条 Python 命令末尾加 `--plan`，只检查输入和预计大小。

## 2. 用该 checkpoint 在 A3 测试一个 GRPO step

```bash
# MODEL 沿用第一步的输出目录；也可以直接指定另一个已完整生成的 HF checkpoint。
export TRAIN_FILES='/你的路径/train.parquet'
export VAL_FILES='/你的路径/validation.parquet'
bash scripts/rl/run_rl_rmpad_a3_fp32.sh smoke
```

`smoke` 使用当前 `MODEL`，训练和 rollout 从同一份权重启动。单机 16 张空闲 NPU，训练 FSDP16/EP16、rollout TP8、关闭 DeepEP。checkpoint 落盘 BF16，Actor 使用 `model_dtype=fp32`，rollout 为 BF16。单步设置为训练 batch 16、每条 prompt 生成 2 个回答、最大回复长度 128。

终端的 `Artifacts:` 会打印输出目录，可通过 `RUN_DIR` 指定新目录。查看其中的 `config.txt` 和 `train.log`，以第 1 个训练 step 完成并正常退出为通过。随机模型的 reward 或 advantage 可能为零。

同一 RL 脚本也支持 `config`（仅导出性能基线配置）和 `train`（按原性能基线训练，默认 100 步）。

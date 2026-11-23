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

# Step5 RL 启动说明

## 352 专家 16 机长跑

正式入口为 scripts/rl_352e.sh。在 A5-01、A5-02 的 16 台服务器上使用同一份共享项目、固定环境和输出目录。head 为 A5-01-01。配置来自完成 32 步的长跑，目标仍为 200 步，profiling 默认关闭。

入口会激活项目环境。Ray 由 start 在私有挂载中启动，普通 ray start 启动的 worker 无法继承这些挂载。在项目根目录执行。配置导出只读模型和数据，写入本轮输出目录。

    run=/absolute/shared/path/to/new-run
    bash scripts/rl_352e.sh config "$run"

## 原生算子

项目的 Python 修复直接从 src 导入。原生组件使用与环境匹配的 Python 3.12、CANN 9.2.0、Ascend 950 和 DeepEP 扩展。对于新检出，按 src/DeepEP/experiments/ep_gmm_fused/README.md 在该子模块构建启用 DEEPEP_BUILD_EP_GMM_FUSED 的 wheel。然后在项目根目录准备一个新的共享产物目录。

    source scripts/activate.sh
    export STEP5_352E_NATIVE_DIR="$PWD/outputs/352e-native"
    mkdir -p "$STEP5_352E_NATIVE_DIR/deepep"
    unzip -j "$WHEEL" 'deep_ep_experimental/ep_gmm_fused/_C*.so' -d "$STEP5_352E_NATIVE_DIR/deepep"
    python src/cann_ops/dispatch/build.py "$STEP5_352E_NATIVE_DIR/dispatch"

WHEEL 指向上述源码构建的实际 wheel 文件。verify 在临时服务内核对 DeepEP 的扩展和 ops.py、dispatch 的 .o 和 .json，随后退出。start 和 train 使用同样的只读挂载。共享 venv 和系统 SDK 保持原安装。服务的文件描述符软上限为 65536，供 128 rank 的 Gloo 连接使用。

## 启动与停止

在各节点分别运行 verify。先在 head 运行 start，再在其余 15 台运行 start，最后在 head 运行 train。每次传入相同的绝对共享输出路径。stop 在执行它的节点上停止该输出目录对应的服务。

    bash scripts/rl_352e.sh verify "$run"
    bash scripts/rl_352e.sh start "$run"
    bash scripts/rl_352e.sh train "$run"
    bash scripts/rl_352e.sh stop "$run"

训练日志为输出目录下的 train.log 和 driver.log。RL Insight 默认地址为 http://197.13.21.1:18080，可通过 RL_INSIGHT_SERVER_URL 指定。长跑设置为 batch 128、输入 1024、输出 2048、TP8/EP32/DP4、训练 EP8/FSDP128/EFSDP16、max_num_seqs 64、sleep level 2、pin memory false 和默认权重发送路径。

## SFT100 单机 8 卡

在项目根目录设置一个全新的共享输出目录，然后依次执行以下命令。single-train 会在带有相同四个只读挂载的服务中启动本地 Ray 和训练进程。

    run="$PWD/outputs/sft100-single-run01"
    bash scripts/rl_352e.sh verify "$run"
    bash scripts/rl_352e.sh single-config "$run"
    bash scripts/rl_352e.sh single-train "$run"

配置文件为 config.txt，主日志为 train.log，第三步 actor rank 0 的一次 micro-batch 前反向在 actor-profile-step3 下。单机使用 SFT100 的 16 专家权重、200 条训练样本和训练内 100 条验证样本。每 10 步验证一次，目标 100 步，关闭 checkpoint。手动停止此作业可在同一节点对相同输出目录执行 stop。

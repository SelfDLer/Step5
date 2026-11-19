# A3 Conda 环境：三步验证命令

使用独立环境 `/mnt/share/z00988734/.conda/step5`，不改 shell 启动文件、不自动激活、不安装 DeepEP。
命令适用于 Bash 和 zsh；检查 CANN 的脚本显式用 Bash 执行。`python=3.12` 后不要加句点。
脚本只打印检查结果，不修改 CANN、模型或数据，不启动完整训练。

基础 NPU 与 GBSA/GBSAG 均已通过时，接着执行 [网络配置与四仓安装准备](NEXT-STEPS.md)，保留已验证的 torch/torch_npu，不重复安装。

## 0. 获取检查脚本

首次获取：

```sh
git clone -b a5toa3 https://github.com/SelfDLer/Step5.git
cd Step5
```

已有克隆则进入该目录，确认工作区干净后更新：

```sh
git switch a5toa3
git pull --ff-only
```

这三步不需要初始化四仓 submodule，不需要权重、数据集或 DeepEP。
不要 source `reference/` 中的 A5 激活脚本。下面第 1 步只有信息查询；第 2、3 步实际计算需要选择一张空闲卡。

## 1. 确认驱动、CANN 和算子版本

在新终端执行，避免继承另一套 Python/CANN 环境。路径采用同事提供的位置；若 A3 的位置不同，先修改该变量。

```sh
export CANN_ENV_SCRIPT=/mnt/share/q30068470/step-1026/ops_1105/ascend-toolkit/set_env.sh
test -r "$CANN_ENV_SCRIPT"
source "$CANN_ENV_SCRIPT"
bash scripts/checks/01_check_cann.sh
```

根据输出的 `npu-smi info -l` 选择实际存在的 NPU ID，再查看该设备的驱动/固件。下面的 `0` 是示例，必要时替换：

```sh
NPU_ID=0 bash scripts/checks/01_check_cann.sh
```

需要确认：

- 设备确实是目标 A3，设备状态正常。
- 驱动/固件版本明确，`Compatibility` 的实际输出没有兼容性问题。
- `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 指向预期环境。
- Toolkit、ops 的完整版本明确。`INVENTORY COMPLETE` 仅表示信息采集完成，不等于版本兼容或算子验证通过。
- 自定义算子没有版本记录时，向提供者确认 GBSA/GBSAG 对应的源码提交和 A3 构建版本。目录名 `ops_1105` 不能代替版本号。

官方参考：[查询驱动、固件和 CANN 组件版本](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/softwareinst/instg/instg_0064.html)。

**先保存这一步输出，再确定匹配的 torch/torch_npu 版本。不要直接安装最新版，也不要机械复制 A5 的版本。**

## 2. 创建 Conda 环境并验证基础 NPU 计算

仅在该环境尚不存在时创建；如果已创建，直接激活，不重复创建或覆盖：

```sh
conda create -y -p /mnt/share/z00988734/.conda/step5 python=3.12
conda activate /mnt/share/z00988734/.conda/step5
```

如果当前 shell 未初始化 conda，但 `conda` 命令可用，可以只在当前终端加载它，然后再次激活。不要为此运行 `conda init`：

```sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/share/z00988734/.conda/step5
```

激活 Conda 后重新 source 目标 CANN。若换了终端，重新设置 `CANN_ENV_SCRIPT`：

```sh
export CANN_ENV_SCRIPT=/mnt/share/q30068470/step-1026/ops_1105/ascend-toolkit/set_env.sh
source "$CANN_ENV_SCRIPT"
which python
python --version
python -m pip --version
python -c 'import sys; print(sys.executable); print(sys.prefix)'
```

`sys.prefix` 应为 `/mnt/share/z00988734/.conda/step5`，Python 为 3.12.x。

**确认第 1 步版本配套后**，用所选版本设置 `TORCH_VERSION` 和 `TORCH_NPU_VERSION`，再安装。这里不提供尚未核对的默认值；变量未设置时命令会退出。安装还需要可用的匹配包源：

```sh
# 示例格式，先把注释中的值换成确认过的完整版本号，再执行 export：
# export TORCH_VERSION=确认过的完整版本号
# export TORCH_NPU_VERSION=确认过的完整版本号
python -m pip install "torch==${TORCH_VERSION:?先确认并设置 TORCH_VERSION}" \
    "torch_npu==${TORCH_NPU_VERSION:?先确认并设置 TORCH_NPU_VERSION}"
```

若提供者给的是配套 wheel，则使用它提供的安装方法，仍在上述 Conda 环境内执行 `python -m pip`。不要从 A5 复制原生扩展。

检查空闲卡，并选择一张。以下假设物理卡 0 空闲；若不是，替换 `ASCEND_RT_VISIBLE_DEVICES` 的值。过滤后脚本的 `--device 0` 指的是逻辑卡 0：

```sh
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0
python scripts/checks/02_check_torch_npu.py \
    --expected-prefix /mnt/share/z00988734/.conda/step5 --device 0
```

通过标志：FP32 和 BF16 各自显示 `forward/backward PASS`，最后显示 `TORCH_NPU_BASIC PASS`。
这是单卡基础矩阵乘法与梯度测试，不验证 HCCL、GBSA 或完整训练。

## 3. 验证 GBSA/GBSAG 接口和小规模精度

继续使用同一 Conda 环境及空闲卡。先只检查 Python 包和接口：

```sh
python scripts/checks/03_check_gbsa.py \
    --expected-prefix /mnt/share/z00988734/.conda/step5 --api-only
```

通过标志：打印实际加载的 `cann_ops_transformer` 路径及三个函数，并显示 `GBSA_API PASS`。
它只确认接口可导入，不证明底层算子能运行。如果缺包，确认同事提供的配套 Python wrapper / wheel / 激活入口；不要随意安装同名包。

再执行真实 NPU 前向、反向并对照 CPU FP32 attention：

```sh
python scripts/checks/03_check_gbsa.py \
    --expected-prefix /mnt/share/z00988734/.conda/step5 \
    --device 0 --lengths 16 129 --kv-heads 4
```

这两例都通过后，可补充 512 token 和不同本地 KV 分组：

```sh
python scripts/checks/03_check_gbsa.py \
    --expected-prefix /mnt/share/z00988734/.conda/step5 \
    --device 0 --lengths 512 --kv-heads 4

for groups in 1 2; do
    python scripts/checks/03_check_gbsa.py \
        --expected-prefix /mnt/share/z00988734/.conda/step5 \
        --device 0 --lengths 16 129 --kv-heads "$groups"
done
```

规格：BF16、head_dim=192、TND、每个 KV head 对应 16 个 Q heads、block shape `[1,8]`、索引容量 513，包含完整块和因果残块。
输出列出前向及 dQ/dK/dV 的 relative L2；默认每项小于 0.02 且无 NaN/Inf，最后显示 `GBSA_SMALL_CASE PASS`。
该阈值是冒烟检查界限，不是正式训练精度指标；不覆盖长序列稀疏 top-k、多样本 packed batch、分布式通信或全模型训推一致性。
脚本按固定 FSDPTurbo `14bffd5` 接口编写，不会为绕过报错而改变接口或回退算子。

## 失败时保留哪些信息

保留第 1 步版本输出、Python/包实际路径、失败命令和完整 traceback。
`unexpected keyword argument` 通常提示 wrapper API 不匹配；`undefined symbol` 或加载库失败需要核对运行时库配套；设备错误需要结合驱动日志定位。
误差超阈值时先保留输出，不要直接放宽阈值。
本仓只提交检查代码和说明，不上传检查日志或结果。A3 的实际通过状态以在目标机器运行的输出为准。

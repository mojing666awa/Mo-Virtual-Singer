# 🎤 Mo Virtual Singer (MVS) Core

> 基于 VITS 架构的虚拟歌手语音合成框架
> **Python 3.10 | PyTorch 2.5.1+cu124 | Windows / Linux**

本项目为开箱即用的 MVS 实现。**VITS 核心 Python 脚本位于项目根目录**，`monotonic_align/` 文件夹仅包含预编译的二进制扩展库，无需额外下载源码或手动编译。

## 📁 项目结构

```text
mvs-core/
├── .github/workflows/
│   └── python-package.yml      # CI：验证环境安装与 MAS 模块导入
│
├── configs/
│   └── mvs_base.json           # 🔧 主配置文件（需按数据修改 n_vocab/sample_rate/segment_size）
│
├── data/                       # 📥 用户数据目录（需手动填充 wav + json）
│
├── monotonic_align/            # ⚙️ MAS 编译产物目录（仅含二进制文件，无 Python 源码）
│   ├── *.pyd                   # ✅ Windows 编译扩展
│   └── *.so                    # ✅ Linux 编译扩展
│
├── commons.py                  # 📦 [VITS 核心] 基础算子：sequence_mask, slice_segments 等
│                               #    ⚠️ 从 jaywalnut310/vits 复制，请勿修改
├── losses.py                   # 💥 [VITS 核心] 损失函数：discriminator_loss, kl_loss, mel_loss
│                               #    ⚠️ 从 jaywalnut310/vits 复制，请勿修改
├── models.py                   # 🏗️ [VITS 核心] 模型定义：SynthesizerTrn, Discriminator
│                               #    ⚠️ 从 jaywalnut310/vits 复制，MVS 扩展在此基础之上进行
├── utils.py                    # 🛠️ [VITS 核心] 工具函数：load_checkpoint, get_hparams
│                               #    ⚠️ 从 jaywalnut310/vits 复制，请勿修改
│
├── dataset.py                  # 🗃️ [MVS 自定义] 数据集加载器：适配 wav+json 标注格式
├── preprocess.py               # 🧹 [MVS 自定义] 数据预处理：音素提取、MIDI 对齐、姿态向量生成
├── train.py                    # 🎯 [MVS 主入口] 训练循环：整合数据加载、模型训练、日志与 Checkpoint
│
├── vits_singer.py              # 🗑️ [已废弃] 老版入口文件，请使用 train.py
├── requirements.txt            # 📦 全局依赖清单
├── LICENSE                     # 📜 开源协议
└── README.md                   # 📖 本文件
---
🔑 关键说明
- 根目录四个核心脚本：commons.py, losses.py, models.py, utils.py 是从 jaywalnut310/vits 复制的必要依赖，直接由 train.py 和 preprocess.py 调用。
- monotonic_align/ 仅含编译产物：该文件夹内只有 .pyd（Windows）或 .so（Linux）二进制文件，不包含任何 Python 源码。MAS 对齐算法已通过 C++ 编译加速，开箱即用。
- 主入口为 train.py：vits_singer.py 为历史遗留文件，当前所有训练与推理均通过 train.py 执行。

⚙️ 环境配置（Windows 优先）
1. 创建环境
```
{
conda create -n mvs-py python=3.10 -y
conda activate mvs-py
}
```

2. 安装 PyTorch (CUDA 12.4)
`bash`pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

3.安装依赖
```
{
# Cython 必须先装（用于后续可能的 MAS 重编译）
pip install Cython==3.0.11 numpy==1.26.4

# 安装其余依赖
pip install -r requirements.txt
}
```

4. 验证 MAS 是否可用
`bash`pn_vocab：必须 = max_phoneme_id + 1 + 128(MIDI) + 32(姿态)
segment_size：显存适配参考：24G → 8192，12G → 4096
Step 4: 启动训练ython -c "from monotonic_align import mas; print('✅ MAS binary loaded successfully')"

>✅ 成功输出表示 monotonic_align/ 中的编译扩展已正确加载。

🚀 训练流程
Step 1: 准备数据
- 将音频放入 data/ 目录：格式为 22050Hz / Mono / 16-bit WAV
- 每个 xxx.wav 需配同名 xxx.json（含音素、MIDI、姿态序列等）
Step 2: 预处理
`bash` python preprocess.py --config configs/mvs_base.json
>输出示例：✅ Done! 1287 valid samples
Step 3: 核对配置
打开 configs/mvs_base.json，重点检查：
```
{
{
  "n_vocab": 709,
  "sample_rate": 22050,
  "segment_size": 8192,
  "max_batch_tokens": 4096
}
}
```
- n_vocab：必须 = max_phoneme_id + 1 + 128(MIDI) + 32(姿态)
- segment_size：显存适配参考：24G → 8192，12G → 4096
Step 4: 启动训练
`bash` python train.py --config configs/mvs_base.json
Step 5: 监控
`bash` tensorboard --logdir logs/mvs_native
🛠️ 常见问题
| 问题 | 解决方案 |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'monotonic_align'` | 确认 `monotonic_align/` 目录下存在对应平台的 `.pyd` 或 `.so` 文件 |
| `ImportError: DLL load failed` (Windows) | 缺少 VC++ 运行时，安装 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| 训练 OOM | 降低 `segment_size` 或 `max_batch_tokens`，启用 `--fp16`（若 GPU 支持） |
| 误用 vits_singer.py | 该文件已废弃，请始终使用 `train.py` 作为入口 |


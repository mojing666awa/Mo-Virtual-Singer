# 🎤 Mo Virtual Singer (MVS) Core

> 基于 VITS 架构的虚拟歌手语音合成框架  
> **Python 3.10 | PyTorch 2.5.1+cu124 | Windows / Linux**

本项目为开箱即用的 MVS 实现，**已内置完整的 VITS 核心模块**（`models.py`, `commons.py`, `losses.py`, `utils.py`）及编译好的 `monotonic_align`，无需额外下载或手动编译。

## 📁 项目结构

mvs-core/
├── .github/workflows/
│   └── python-package.yml      # CI：验证环境安装与 MAS 模块导入
│
├── configs/
│   └── mvs_base.json           # 🔧 主配置文件（需按数据修改 n_vocab/sample_rate/segment_size）
│
├── data/                       # 📥 用户数据目录（需手动填充 wav + json）
│
├── monotonic_align/            # ⚙️ MAS 对齐模块（本地编译目录，非 pip 包）
│   ├── *.pyd / *.so            # ✅ 已编译的 C++ 扩展（Windows/Linux）
│   ├── check.py                # 🔍 MAS 独立验证脚本
│   └── ...                     # 其他 MAS 辅助文件
│
├── commons.py                  # 📦 [VITS 核心] 基础算子：sequence_mask, slice_segments, rand_slice 等
│                               #    ⚠️ 从 jaywalnut310/vits 复制，请勿修改
├── losses.py                   # 💥 [VITS 核心] 损失函数：discriminator_loss, generator_loss, kl_loss, mel_loss
│                               #    ⚠️ 从 jaywalnut310/vits 复制，请勿修改
├── models.py                   # 🏗️ [VITS 核心] 模型定义：SynthesizerTrn, Discriminator, ResidualCouplingBlock
│                               #    ⚠️ 从 jaywalnut310/vits 复制，MVS 扩展在此文件基础上进行
├── utils.py                    # 🛠️ [VITS 核心] 工具函数：load_checkpoint, get_hparams, plot_spectrogram_to_numpy
│                               #    ⚠️ 从 jaywalnut310/vits 复制，请勿修改
│
├── dataset.py                  # 🗃️ [MVS 自定义] 数据集加载器：适配 wav+json 标注格式
├── preprocess.py               # 🧹 [MVS 自定义] 数据预处理：音素提取、MIDI 对齐、姿态向量生成
├── train.py                    # 🎯 [MVS 主入口] 训练循环：整合数据加载、模型训练、日志记录、Checkpoint 保存
│
├── vits_singer.py              # 🗑️ [已废弃] 老版入口文件，请使用 train.py
├── requirements.txt            # 📦 全局依赖清单
├── LICENSE                     # 📜 开源协议
└── README.md                   # 📖 本文件
🔍 关键说明：
monotonic_align/ 是 本地目录，非 pip 包。其内部包含 setup.py 和已编译的 .pyd（Windows）或 .so（Linux）文件。
所有 VITS 核心代码（models.py, commons.py 等）直接位于 monotonic_align/ 目录下，train.py 通过 from monotonic_align.models import SynthesizerTrn 导入。
vits_singer.py 是本项目的主训练/推理入口（替代原版 train.py）。
⚙️ 环境配置（Windows 优先）
1. 创建环境
conda create -n mvs-py python=3.10 -y
conda activate mvs-py
2. 安装 PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
3. 安装依赖
# Cython 必须先装（用于后续可能的 MAS 重编译）
pip install Cython==3.0.11 numpy==1.26.4

# 安装其余依赖
pip install -r requirements.txt
4. 验证 MAS 是否可用
python -c "import sys; sys.path.insert(0, './monotonic_align'); import models; print('✅ MAS loaded from local directory')"

✅ 成功输出表示 monotonic_align/ 中的 .pyd 已正确生成且可导入。

🚀 训练流程
Step 1: 准备数据
将音频放入 data/ 目录：格式为 22050Hz / Mono / 16-bit WAV
每个 xxx.wav 需配同名 xxx.json（含音素、MIDI、姿态序列等）

Step 2: 预处理
python monotonic_align/preprocess.py --config configs/mvs_base.json
输出示例：✅ Done! 1287 valid samples

Step 3: 核对配置
打开 configs/mvs_base.json，重点检查：
{
  "n_vocab": 709,          // ← 必须 = max_phoneme_id + 1 + 128(MIDI) + 32(姿态)
  "sample_rate": 22050,
  "segment_size": 8192,    // 显存适配：24G→8192, 12G→4096
  "max_batch_tokens": 4096
}

Step 4: 启动训练
python monotonic_align/vits_singer.py --config configs/mvs_base.json

Step 5: 监控
tensorboard --logdir logs/mvs_native

🛠️ 常见问题
| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: No module named 'monotonic_align'` | 确保在 `preprocess.py`/`vits_singer.py` 中已添加 `sys.path.insert(0, './monotonic_align')` |
| `ImportError: DLL load failed` (Windows)  | 重新进入 `monotonic_align/` 目录执行：<br>`python setup.py build_ext --inplace`（需 VS Build Tools） |
| 训练 OOM | 降低 `segment_size` 或 `max_batch_tokens`，启用 `--fp16`（若 GPU 支持） |

📄 License
本项目基于 jaywalnut310/vits 修改，遵循 MIT License。

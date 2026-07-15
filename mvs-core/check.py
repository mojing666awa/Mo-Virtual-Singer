import torch
import torchaudio

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0)}")
print(f"✅ Torchaudio: {torchaudio.__version__}")

# 关键检查：BF16 支持（MVS 训练强烈推荐）
print(f"✅ BF16 Supported: {torch.cuda.is_bf16_supported()}")

# 关键检查：torch.compile 可用性（CFM 加速必需）
try:
    @torch.compile(mode="reduce-overhead")
    def test_fn(x): return x * 2
    test_fn(torch.randn(10, device="cuda"))
    print("✅ torch.compile: Working")
except Exception as e:
    print(f"❌ torch.compile Failed: {e}")
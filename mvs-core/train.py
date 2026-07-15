import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.cluster import KMeans
import numpy as np
import os

# ==========================================
# 1. WavLM 语义特征提取器 (冻结参数，仅做特征提取)
# ==========================================
class WavLMExtractor(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wavlm = WavLMModel.from_pretrained(model_name)
        # 训练 MVS 时不更新 WavLM 权重
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def forward(self, audio_path: str, sr=16000):
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)
            waveform = resampler(waveform)
        
        inputs = self.extractor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
        outputs = self.wavlm(**inputs)
        # 返回最后一层隐藏状态 [1, T, 768]
        return outputs.last_hidden_state

# ==========================================
# 2. 发声姿态空间构建 (无监督聚类)
# ==========================================
def build_posture_space(feature_dir: str, n_clusters: int = 32):
    """
    将所有训练音频的 WavLM 特征进行 K-Means 聚类，
    生成离散的 '发声姿态' 码本。
    """
    all_features = []
    feat_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    
    print(f"Loading {len(feat_files)} feature files...")
    for f in feat_files:
        feat = np.load(os.path.join(feature_dir, f))
        all_features.append(feat)
        
    features_np = np.concatenate(all_features, axis=0)
    print(f"Clustering {features_np.shape} into {n_clusters} postures...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_np)
    
    # 保存姿态码本，后续推理和声学模型训练都需要它
    np.save("checkpoints/posture_codebook.npy", kmeans.cluster_centers_)
    print("Posture space built and saved!")
    return kmeans

# ==========================================
# 3. MVS 声学模型占位符 (待实现)
# ==========================================
class MVSAcousticModel(nn.Module):
    def __init__(self, posture_dim=768, n_postures=32):
        super().__init__()
        self.posture_embedding = nn.Embedding(n_postures, posture_dim)
        # TODO: 接入 Flow Matching / Diffusion / VITS 等生成模块
        # 输入: 乐谱(MIDI) + 歌词 + 姿态ID
        # 输出: 梅尔频谱 或 原始波形
        
    def forward(self, midi, lyrics, posture_ids):
        posture_cond = self.posture_embedding(posture_ids)
        # TODO: 实现具体的声学解码逻辑
        raise NotImplementedError("Acoustic decoder not implemented yet.")

# ==========================================
# 4. 训练入口示例
# ==========================================
if __name__ == "__main__":
    # Step 1: 提取并缓存 WavLM 特征 (只需做一次)
    extractor = WavLMExtractor()
    # audio_paths = ["data/vocals/001.wav", "data/vocals/002.wav"]
    # for path in audio_paths:
    #     feat = extractor(path).squeeze(0).cpu().numpy()
    #     np.save(f"data/features/{os.path.basename(path)}.npy", feat)
    
    # Step 2: 构建发声姿态空间
    # build_posture_space("data/features", n_clusters=32)
    
    # Step 3: 初始化声学模型并开始训练循环
    model = MVSAcousticModel()
    print("MVS Core skeleton loaded successfully.")
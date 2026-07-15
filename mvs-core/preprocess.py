import os
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.cluster import KMeans

# ==========================================
# 1. 配置参数 (根据你的数据集修改)
# ==========================================
CONFIG = {
    "raw_audio_dir": "data/raw_vocals",      # UVR5分离后的干净人声目录
    "midi_label_dir": "data/midi_labels",    # 对应的MIDI/歌词标注JSON目录
    "output_dir": "data/mvs_processed",     # 预处理输出目录
    "wavlm_model": "microsoft/wavlm-base-plus",
    "sample_rate": 22050,                   # VITS 标准采样率
    "hop_length": 256,                      # STFT hop length
    "n_postures": 32,                       # 发声姿态聚类数
    "min_audio_len": 1.0,                   # 最短音频秒数(过滤静音/无效片段)
}

# ==========================================
# 2. WavLM 姿态提取与聚类
# ==========================================
class PostureTagger:
    def __init__(self, config):
        self.config = config
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["wavlm_model"])
        self.wavlm = WavLMModel.from_pretrained(config["wavlm_model"])
        self.wavlm.eval()
        self.kmeans = None
        
    @torch.no_grad()
    def extract_features(self, audio_path):
        """提取单条音频的WavLM特征 [T, 768]"""
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:  # WavLM 固定需要 16k
            wav = torchaudio.functional.resample(wav, sr, 16000)
        
        inputs = self.extractor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        outputs = self.wavlm(**inputs)
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()
    
    def build_codebook(self, audio_paths):
        """第一阶段：扫描所有音频构建姿态码本"""
        all_feats = []
        print("🔍 Phase 1: Extracting WavLM features for codebook...")
        for path in tqdm(audio_paths):
            feat = self.extract_features(path)
            # 降采样到声学模型帧率 (16000/320 -> 22050/256 ≈ 86fps)
            # WavLM base 输出 50fps，这里简单用插值对齐
            target_frames = int(len(feat) * (self.config["sample_rate"] / self.config["hop_length"]) / (16000 / 320))
            feat_aligned = torch.nn.functional.interpolate(
                torch.tensor(feat).unsqueeze(0).transpose(1,2), 
                size=target_frames, mode='linear', align_corners=False
            ).squeeze(0).transpose(0,1).numpy()
            all_feats.append(feat_aligned)
            
        feats_cat = np.concatenate(all_feats, axis=0)
        print(f"📊 Clustering {feats_cat.shape[0]} frames into {self.config['n_postures']} postures...")
        self.kmeans = KMeans(n_clusters=self.config["n_postures"], random_state=42, n_init=10)
        self.kmeans.fit(feats_cat)
        
        # 保存码本
        ckpt_dir = Path(self.config["output_dir"]) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.save(ckpt_dir / "posture_codebook.npy", self.kmeans.cluster_centers_)
        print(f"✅ Codebook saved to {ckpt_dir / 'posture_codebook.npy'}")
        
    def tag_audio(self, audio_path):
        """第二阶段：为单条音频打上姿态ID序列"""
        feat = self.extract_features(audio_path)
        target_frames = int(len(feat) * (self.config["sample_rate"] / self.config["hop_length"]) / (16000 / 320))
        feat_aligned = torch.nn.functional.interpolate(
            torch.tensor(feat).unsqueeze(0).transpose(1,2), 
            size=target_frames, mode='linear', align_corners=False
        ).squeeze(0).transpose(0,1).numpy()
        
        posture_ids = self.kmeans.predict(feat_aligned)
        return posture_ids.astype(np.int32)

# ==========================================
# 3. MIDI/歌词标注加载器
# ==========================================
def load_midi_label(json_path, target_frames):
    """
    加载MIDI标注并对齐到声学帧率
    JSON格式示例: {"phonemes": ["a", "i"], "midi_notes": [60, 62], "durations": [0.5, 0.3]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        label = json.load(f)
    
    phoneme2id = {'a': 0, 'i': 1, 'u': 2, 'e': 3, 'o': 4}  # TODO: 替换为完整音素表
    ph_ids = [phoneme2id.get(p, 0) for p in label["phonemes"]]
    midi_notes = label["midi_notes"]
    
    # 将音符级标注展开为帧级标注
    frame_phones = []
    frame_midis = []
    fps = CONFIG["sample_rate"] / CONFIG["hop_length"]
    
    for ph, note, dur in zip(ph_ids, midi_notes, label["durations"]):
        n_frames = max(1, int(dur * fps))
        frame_phones.extend([ph] * n_frames)
        frame_midis.extend([note] * n_frames)
    
    # 截断或填充到目标帧数
    frame_phones = frame_phones[:target_frames] + [0] * max(0, target_frames - len(frame_phones))
    frame_midis = frame_midis[:target_frames] + [0] * max(0, target_frames - len(frame_midis))
    
    return np.array(frame_phones, dtype=np.int32), np.array(frame_midis, dtype=np.int32)

# ==========================================
# 4. 主预处理流程
# ==========================================
def preprocess():
    out_dir = Path(CONFIG["output_dir"])
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    audio_files = sorted([f for f in Path(CONFIG["raw_audio_dir"]).glob("*.wav")])
    tagger = PostureTagger(CONFIG)
    
    # Phase 1: 构建姿态码本
    tagger.build_codebook(audio_files)
    
    # Phase 2: 逐条处理并保存
    metadata = []
    print("\n🎵 Phase 2: Processing individual samples...")
    for audio_path in tqdm(audio_files):
        # 过滤短音频
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        if duration < CONFIG["min_audio_len"]:
            continue
            
        # 重采样到VITS目标采样率
        wav, sr = torchaudio.load(str(audio_path))
        if sr != CONFIG["sample_rate"]:
            wav = torchaudio.functional.resample(wav, sr, CONFIG["sample_rate"])
        
        target_frames = wav.shape[1] // CONFIG["hop_length"]
        
        # 获取姿态ID
        posture_ids = tagger.tag_audio(audio_path)
        # 确保长度一致
        min_len = min(target_frames, len(posture_ids))
        posture_ids = posture_ids[:min_len]
        
        # 加载MIDI/歌词标注
        label_path = Path(CONFIG["midi_label_dir"]) / f"{audio_path.stem}.json"
        if not label_path.exists():
            print(f"⚠️ Missing label for {audio_path.name}, skipping...")
            continue
        ph_ids, midi_notes = load_midi_label(label_path, min_len)
        
        # 保存处理后的数据
        stem = audio_path.stem
        torchaudio.save(out_dir / "audio" / f"{stem}.wav", wav, CONFIG["sample_rate"])
        np.savez_compressed(
            out_dir / "labels" / f"{stem}.npz",
            phoneme_ids=ph_ids,
            midi_notes=midi_notes,
            posture_ids=posture_ids,
            length=min_len
        )
        
        metadata.append({
            "id": stem,
            "length": min_len,
            "duration": round(min_len * CONFIG["hop_length"] / CONFIG["sample_rate"], 3)
        })
    
    # 保存元数据索引
    with open(out_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Preprocessing complete! {len(metadata)} samples processed.")
    print(f"📁 Output directory: {out_dir}")

if __name__ == "__main__":
    preprocess()
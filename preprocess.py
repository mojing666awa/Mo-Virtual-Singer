import os, json, numpy as np, torch, torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.cluster import KMeans

class NativeMVSPreprocessor:
    def __init__(self, config_path="configs/mvs_base.json"):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.sr = self.cfg["sample_rate"]
        self.hop = self.cfg["hop_length"]
        self.out_dir = Path(self.cfg["processed_dir"])
        (self.out_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        print("🔄 Loading WavLM...")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg["wavlm_model"])
        self.wavlm = WavLMModel.from_pretrained(self.cfg["wavlm_model"]).eval()
        self.kmeans = None
        
        with open(self.cfg["phoneme_dict"], 'r') as f:
            self.ph2id = json.load(f)
        # 为MIDI和姿态预留特殊token空间 (假设字典最大ID<600)
        self.midi_offset = 600  
        self.posture_offset = 728 

    @torch.no_grad()
    def _get_wavlm_feat(self, wav_16k):
        inputs = self.extractor(wav_16k.numpy(), sampling_rate=16000, return_tensors="pt")
        return self.wavlm(**inputs).last_hidden_state.squeeze(0)

    def build_codebook(self, audio_paths):
        all_feats = []
        print(f"🔍 Extracting WavLM features from {len(audio_paths)} files...")
        for p in tqdm(audio_paths):
            wav, sr = torchaudio.load(str(p))
            if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
            feat = self._get_wavlm_feat(wav.squeeze(0))
            all_feats.append(feat.cpu().numpy())
        feats = np.concatenate(all_feats, axis=0)
        print(f"📊 Clustering into {self.cfg['n_postures']} postures...")
        self.kmeans = KMeans(n_clusters=self.cfg["n_postures"], random_state=42, n_init=10)
        self.kmeans.fit(feats)
        np.save(self.out_dir / "posture_codebook.npy", self.kmeans.cluster_centers_)
        print(f"✅ Codebook saved")

    def process_sample(self, audio_path):
        wav, sr = torchaudio.load(str(audio_path))
        if sr != self.sr: wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.squeeze(0)
        
        # 提取姿态并插值对齐到帧级
        wav_16k = torchaudio.functional.resample(wav.unsqueeze(0), self.sr, 16000).squeeze(0)
        feat = self._get_wavlm_feat(wav_16k)
        target_frames = wav.shape[0] // self.hop
        feat_interp = torch.nn.functional.interpolate(
            feat.unsqueeze(0).transpose(1,2), size=target_frames, mode='linear', align_corners=False
        ).squeeze(0).transpose(0,1).cpu().numpy()
        posture_ids = self.kmeans.predict(feat_interp).astype(np.int32)
        
        # 加载标注
        label_path = Path(self.cfg["midi_label_dir"]) / f"{audio_path.stem}.json"
        if not label_path.exists(): return None
        with open(label_path, 'r') as f: lbl = json.load(f)
        
        # 🔥 核心：将 音素+MIDI+姿态 编码为原生VITS可识别的整数序列
        merged_tokens = []
        fps = self.sr / self.hop
        for ph, note, dur in zip(lbl["phonemes"], lbl["midi_notes"], lbl["durations"]):
            n_f = max(1, int(dur * fps))
            ph_id = self.ph2id.get(ph, self.ph2id["_spn"])
            midi_token = note + self.midi_offset      # MIDI 0-127 -> 600-727
            post_token = posture_ids[0] + self.posture_offset  # 简化:取该音符段首帧姿态
            
            # 每个音符编码为: [音素ID, MIDI_Token, 姿态Token]
            merged_tokens.extend([ph_id, midi_token, post_token])
            
        if len(merged_tokens) < 3: return None
        
        stem = audio_path.stem
        torchaudio.save(self.out_dir / "audio" / f"{stem}.wav", wav.unsqueeze(0), self.sr)
        np.savez_compressed(
            self.out_dir / "labels" / f"{stem}.npz",
            tokens=np.array(merged_tokens, dtype=np.int32)
        )
        return {"id": stem, "length": len(merged_tokens)}

    def run(self):
        audio_paths = sorted(Path(self.cfg["raw_audio_dir"]).glob("*.wav"))
        self.build_codebook(audio_paths)
        metadata = []
        print("\n🎵 Processing samples...")
        for p in tqdm(audio_paths):
            meta = self.process_sample(p)
            if meta: metadata.append(meta)
        with open(self.out_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✅ Done! {len(metadata)} valid samples.")

if __name__ == "__main__":
    NativeMVSPreprocessor().run()
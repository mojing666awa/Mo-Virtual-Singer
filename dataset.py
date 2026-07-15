import json, numpy as np, torch, torchaudio, random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class NativeMVSDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, labels_dir, segment_size=8192, hop_length=256):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.audio_dir = Path(audio_dir)
        self.labels_dir = Path(labels_dir)
        self.segment_size = segment_size
        self.hop_length = hop_length
        
    def __len__(self): return len(self.metadata)
    
    def __getitem__(self, idx):
        meta = self.metadata[idx]
        wav, _ = torchaudio.load(self.audio_dir / f"{meta['id']}.wav")
        lbl = np.load(self.labels_dir / f"{meta['id']}.npz")
        wav = wav.squeeze(0)
        tokens = torch.from_numpy(lbl["tokens"]).long()
        
        # 音频随机裁剪
        max_start = wav.size(0) - self.segment_size
        if max_start > 0:
            start = random.randint(0, max_start)
            wav_seg = wav[start:start+self.segment_size]
        else:
            wav_seg = torch.nn.functional.pad(wav, (0, self.segment_size - wav.size(0)))
            
        # 注意: 原生VITS不对text做裁剪, MAS会在forward中对齐
        return tokens, wav_seg

class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, max_tokens_per_batch=4096):
        self.dataset = dataset
        self.max_tokens = max_tokens_per_batch
        self.batches = self._create_batches()
    def _create_batches(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batches, cur_batch, cur_tokens = [], [], 0
        for idx in indices:
            t_len = self.dataset.metadata[idx]["length"]
            if cur_tokens + t_len > self.max_tokens and cur_batch:
                batches.append(cur_batch)
                cur_batch, cur_tokens = [], 0
            cur_batch.append(idx)
            cur_tokens += t_len
        if cur_batch: batches.append(cur_batch)
        return batches
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)

def collate_fn(batch):
    """原生VITS标准collate: pad text, stack audio"""
    texts, audios = zip(*batch)
    text_lengths = torch.LongTensor([t.size(0) for t in texts])
    text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    audios = torch.stack(audios)
    return text_padded, text_lengths, audios

def get_native_dataloader(config):
    ds = NativeMVSDataset(
        f"{config['processed_dir']}/metadata.json",
        f"{config['processed_dir']}/audio",
        f"{config['processed_dir']}/labels",
        config["segment_size"], config["hop_length"]
    )
    sampler = DynamicBatchSampler(ds, config["max_batch_tokens"])
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
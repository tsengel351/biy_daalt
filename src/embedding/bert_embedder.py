"""
Embedding гаргах модуль (fine-tune хийхгүй, зөвхөн CLS/mean pooling).
Colab/Kaggle дээр ажиллуулж .npy болгож хадгална.
"""
import numpy as np
from typing import List, Optional
from tqdm import tqdm

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    print("⚠️ PyTorch/Transformers суулгана уу (embedding гаргах үед).")

class BertEmbedder:
    def __init__(self, model_name: str, max_length: int = 256, pooling: str = "cls", device: Optional[str] = None):
        if not AVAILABLE:
            raise ImportError("PyTorch/Transformers байхгүй байна.")
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {self.model_name}"):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=ids, attention_mask=mask)
                if self.pooling == "cls":
                    emb = out.last_hidden_state[:, 0, :]
                else:  # mean
                    m = mask.unsqueeze(-1).expand(out.last_hidden_state.size())
                    emb = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)
                outs.append(emb.cpu().numpy())
                torch.cuda.empty_cache()
        return np.vstack(outs)
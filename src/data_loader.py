import os
import numpy as np
from pathlib import Path

def load_embeddings(path="embeddings"):
    data = {}
    for f in Path(path).glob("*_train_embeddings.npy"):
        name = f.stem.replace("_train_embeddings", "")
        base = f"{path}/{name}"
        needed = [f"{base}_train_labels.npy", f"{base}_test_embeddings.npy", f"{base}_test_labels.npy"]
        if all(os.path.exists(p) for p in needed):
            data[name] = {
                "X_train": np.load(f"{base}_train_embeddings.npy"),
                "y_train": np.load(f"{base}_train_labels.npy"),
                "X_test":  np.load(f"{base}_test_embeddings.npy"),
                "y_test":  np.load(f"{base}_test_labels.npy"),
            }
            print(f"✅ {name}: {data[name]['X_train'].shape}")
    return data
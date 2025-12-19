#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_embeddings
from src.models import get_classifiers
from src.evaluation import Evaluator
import os

def save(df, path="results"):
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/results.csv", index=False)
    print(f"\n Saved: {path}/results.csv")

def main():
    print("="*60)
    print(" BERT Embedding + Classification (20-run CV)")
    print("="*60)

    emb = load_embeddings("embeddings")
    if not emb:
        print(" embeddings/ хоосон байна! *_train_embeddings.npy зэргийг байрлуулна уу.")
        return

    ev = Evaluator(n_splits=2, n_repeats=1, seed=42)
    df = ev.run(emb, get_classifiers(42))
    save(df)

    print("\n ҮР ДҮН (F1-Score эрэмбэ):")
    print(df[["Embedding","Classifier","Accuracy","F1","AUC"]].to_string(index=False))

    print("\n TOP 5:")
    for _, r in df.head(5).iterrows():
        print(f"  {r['Embedding']} + {r['Classifier']}: F1={r['F1']}")

if __name__ == "__main__":
    main()
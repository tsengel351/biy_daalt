import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

class Evaluator:
    """RepeatedStratifiedKFold (2×1=2 runs)"""
    def __init__(self, n_splits=2, n_repeats=1, seed=42):
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        self.runs = n_splits * n_repeats

    def _eval(self, cls, params, X, y, name):
        m = {k: [] for k in ["acc","prec","rec","f1","auc"]}
        for tr, va in tqdm(self.cv.split(X, y), total=self.runs, desc=f"  {name}", leave=False):
            model = cls(**params).fit(X[tr], y[tr])
            pred = model.predict(X[va])
            prob = model.predict_proba(X[va])[:,1]
            m["acc"].append(accuracy_score(y[va], pred))
            m["prec"].append(precision_score(y[va], pred))
            m["rec"].append(recall_score(y[va], pred))
            m["f1"].append(f1_score(y[va], pred))
            m["auc"].append(roc_auc_score(y[va], prob))
            if hasattr(model, "clear"): model.clear()
        return {k: (np.mean(v), np.std(v)) for k,v in m.items()}

    def run(self, embeddings, classifiers):
        rows = []
        for emb, data in embeddings.items():
            X, y = data["X_train"], data["y_train"]
            print(f"\n📊 {emb} ({X.shape[0]:,} × {X.shape[1]})")
            for clf, info in classifiers.items():
                s = self._eval(info["class"], info["params"], X, y, clf)
                rows.append({
                    "Embedding": emb, "Classifier": clf,
                    "Accuracy": f"{s['acc'][0]:.4f}±{s['acc'][1]:.4f}",
                    "Precision": f"{s['prec'][0]:.4f}±{s['prec'][1]:.4f}",
                    "Recall": f"{s['rec'][0]:.4f}±{s['rec'][1]:.4f}",
                    "F1": f"{s['f1'][0]:.4f}±{s['f1'][1]:.4f}",
                    "AUC": f"{s['auc'][0]:.4f}±{s['auc'][1]:.4f}",
                    "_f1": s['f1'][0],
                    "_auc": s['auc'][0],
                })
                print(f"  {clf}: F1={s['f1'][0]:.4f}, AUC={s['auc'][0]:.4f}")
        return pd.DataFrame(rows).sort_values("_f1", ascending=False)
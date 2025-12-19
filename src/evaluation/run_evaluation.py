#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_embeddings
from src.models import LogisticRegressionModel, AdaBoostModel, RandomForestModel, LSTMModel
from src.evaluation import ModelEvaluator
from src.utils import set_seed, save_results, print_summary

def main():
    set_seed(42)
    embeddings = load_embeddings("embeddings")
    if not embeddings:
        print("embeddings/ хоосон байна! *_train_embeddings.npy байрлуулна уу.")
        return

    classifiers = {
        "Logistic Regression": {"class": LogisticRegressionModel, "params": {"C":1.0, "max_iter":1000, "random_state":42}},
        "AdaBoost": {"class": AdaBoostModel, "params": {"n_estimators":100, "learning_rate":0.1, "random_state":42}},
        "Random Forest": {"class": RandomForestModel, "params": {"n_estimators":200, "max_depth":20, "random_state":42}},
        "LSTM": {"class": LSTMModel, "params": {"lstm_units":128, "dropout":0.3, "epochs":10, "random_state":42}},
    }

    ev = ModelEvaluator(n_splits=5, n_repeats=4, random_state=42)
    df = ev.evaluate_all(embeddings, classifiers)
    save_results(df, "results", "comparison_results")

    print("\n БҮХ ҮР ДҮН:")
    print(df[["Embedding","Classifier","Accuracy","F1","AUC"]].to_string(index=False))

    print("\n TOP 5 (F1):")
    for _, r in df.head(5).iterrows():
        print(f"  {r['Embedding']} + {r['Classifier']}: F1={r['F1']} AUC={r['AUC']}")

    print_summary(df)
    print("\n Дууслаа.")

if __name__ == "__main__":
    main()
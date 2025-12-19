import os

def save(df, path="results"):
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/results.csv", index=False)
    print(f"\n💾 Saved: {path}/results.csv")

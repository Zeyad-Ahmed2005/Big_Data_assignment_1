import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
raw_files = sorted((BASE / "data" / "raw_data").glob("Youtube*.csv"))

frames = []
for f in raw_files:
    df = pd.read_csv(f)
    df["VIDEO"] = f.stem.split("-")[1]  # Psy, KatyPerry, LMFAO, Eminem, Shakira
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data.to_csv(BASE / "data" / "dataset.csv", index=False)

print(data.shape)
print(data["CLASS"].value_counts())
print(data.groupby("VIDEO").size())

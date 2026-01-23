import pandas as pd

PREFIX = "clock_kaggle/"
IN_CSV = "clock_kaggle/rotations.csv"
OUT_CSV = "rotation_stripped.csv"

df = pd.read_csv(IN_CSV)

df["file"] = df["file"].astype(str).str.strip().apply(
    lambda s: s[len(PREFIX):] if s.startswith(PREFIX) else s
)

df.to_csv(OUT_CSV, index=False)
print("saved:", OUT_CSV)
print("example:", df["file"].iloc[0])

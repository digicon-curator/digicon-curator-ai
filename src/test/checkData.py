import pandas as pd

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

'''print(df.shape)

print("\n===== source 분포 =====")
print(df["source"].value_counts())

print("\n===== 결측치 =====")
print(df.isnull().sum())'''

print(df.columns.tolist())

print(df[["description", "description.1"]].head())

print(df[["items", "items.1"]].head())
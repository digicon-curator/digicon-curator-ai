import pandas as pd

df = pd.read_csv(
    "data/raw/문화재청_문화재정보(20260607).csv",
    encoding="utf-8-sig",
    low_memory=False
)

df["items"] = (
    df["EXTENT"].fillna("").astype(str)
    + " "
    + df["MEDIUM"].fillna("").astype(str)
)

columns = [
    "TITLE",
    "SUBJECTKEYWORD",
    "DESCRIPTION",
    "SPATIAL",
    "items"
]

df = df[columns]

df = df.rename(
    columns={
        "TITLE": "name",
        "SUBJECTKEYWORD": "category",
        "DESCRIPTION": "description",
        "SPATIAL": "address"
    }
)

df.to_csv(
    "data/processed/heritage.csv",
    index=False,
    encoding="utf-8-sig"
)

print("heritage.csv 저장 완료")
print(df.shape)
print(df.head())
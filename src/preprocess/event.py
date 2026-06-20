import pandas as pd

df = pd.read_csv(
    "data/raw/문화체육관광부_기관 (문화)행사정보(20260602).csv",
    encoding="utf-8-sig",
    low_memory=False
)

df["description"] = (
    df["DESCRIPTION"].fillna("").astype(str)
    + " "
    + df["ABSTRACT"].fillna("").astype(str)
)

columns = [
    "TITLE",
    "description",
    "PERIOD",
    "TYPE",
    "SPATIALCOVERAGE"
]

df = df[columns]

df = df.rename(
    columns={
        "TITLE": "name",
        "PERIOD": "period",
        "TYPE": "category",
        "SPATIALCOVERAGE": "address"
    }
)

df.to_csv(
    "data/processed/event.csv",
    index=False,
    encoding="utf-8-sig"
)

print("event.csv 저장 완료")
print(df.shape)
print(df.head())
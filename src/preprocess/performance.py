import pandas as pd

df = pd.read_csv(
    "data/raw/국립중앙극장 외_기관 공연정보(20260602).csv",
    encoding="utf-8-sig",
    low_memory=False
)

df["description"] = (
    df["DESCRIPTION"].fillna("").astype(str)
    + " "
    + df["ABSTRACT"].fillna("").astype(str)
)

df["items"] = (
    df["AUDIENCE"].fillna("").astype(str)
    + " "
    + df["TEMPORALCOVERAGE"].fillna("").astype(str)
)

columns = [
    "TITLE",
    "description",
    "PERIOD",
    "items",
    "SPATIALCOVERAGE",
    "TYPE"
]

df = df[columns]

df = df.rename(
    columns={
        "TITLE": "name",
        "PERIOD": "period",
        "SPATIALCOVERAGE": "address",
        "TYPE": "category"
    }
)

df.to_csv(
    "data/processed/performance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("performance.csv 저장 완료")
print(df.shape)
print(df.head())
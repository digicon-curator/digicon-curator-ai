import pandas as pd

df = pd.read_csv(
    "data/raw/heritage.csv",
    encoding="utf-8-sig"
)

columns = [
    "title",
    "description",
    "spatial",
    "subjectKeyword",
    "extent",
]

df = df[columns]

df = df.rename(
    columns={
        "title": "name",
        "description": "description",
        "spatial": "address",
        "subjectKeyword": "category",
        "extent": "period",
    }
)

df.to_csv(
    "data/processed/heritageProcessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("heritageProcessed.csv 저장 완료")
print(df.shape)
print(df.head())
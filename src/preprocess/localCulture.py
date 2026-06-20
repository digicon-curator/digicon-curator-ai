import pandas as pd

df = pd.read_csv(
    "data/raw/한국학중앙연구원 외_한국의 향토문화(20260522).csv",
    encoding="utf-8-sig",
    low_memory=False
)

columns = [
    "TITLE",
    "DESCRIPTION",
]

df = df[columns]

df = df.rename(
    columns={
        "TITLE": "name",
        "DESCRIPTION": "description"
    }
)

df.to_csv(
    "data/processed/localCulture.csv",
    index=False,
    encoding="utf-8-sig"
)

print("localCulture.csv 저장 완료")
print(df.shape)
print(df.head())
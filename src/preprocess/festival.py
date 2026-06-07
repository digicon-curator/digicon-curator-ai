import pandas as pd

df = pd.read_csv(
    "data/raw/festival.csv",
    encoding="utf-8-sig"
)

columns = [
    "title",
    "description",
    "spatialCoverage",
    "subjectKeyword",
    "eventPeriod"
]

df = df[columns]

df = df.rename(
    columns={
        "title": "name",
        "description": "description",
        "spatialCoverage": "address",
        "eventPeriod": "period",
        "subjectKeyword": "category"
    }
)

df.to_csv(
    "data/processed/festivalProcessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("festivalProcessed.csv 저장 완료")
print(df.shape)
print(df.head())
import pandas as pd

df = pd.read_csv(
    "data/raw/문화체육관광부_지역축제정보(20260606).csv",
    encoding="utf-8-sig",
    low_memory=False
)

columns = [
    "TITLE",
    "DESCRIPTION",
    "SPATIALCOVERAGE",
    "EVENTPERIOD",
    "SUBJECTCATEGORY"
]

df = df[columns]

df = df.rename(
    columns={
        "TITLE": "name",
        "DESCRIPTION": "description",
        "SPATIALCOVERAGE": "address",
        "EVENTPERIOD": "period",
        "SUBJECTCATEGORY": "category"
    }
)

df.to_csv(
    "data/processed/festival.csv",
    index=False,
    encoding="utf-8-sig"
)

print("festival.csv 저장 완료")
print(df.shape)
print(df.head())
import pandas as pd

df = pd.read_csv(
    "data/raw/전국지역특화거리표준데이터.csv",
    encoding="cp949"
)

columns = [
    "거리명",
    "거리소개",
    "소재지도로명"
]

df = df[columns]

df = df.rename(
    columns={
        "거리명": "name",
        "거리소개": "description",
        "소재지도로명": "address"
    }
)

df.to_csv(
    "data/processed/street.csv",
    index=False,
    encoding="utf-8-sig"
)

print("street.csv 저장 완료")
print(df.shape)
print(df.head())
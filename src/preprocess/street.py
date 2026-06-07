import pandas as pd

df = pd.read_csv(
    "data/raw/street.csv",
    encoding="cp949"
)

columns = [
    "거리명",
    "거리소개",
    "소재지도로명",
    "소재지지번주소",
    "위도",
    "경도",
    "총길이",
    "점포수"
]

df = df[columns]

df = df.rename(
    columns={
        "거리명": "name",
        "거리소개": "description",
        "소재지도로명": "roadAddress",
        "소재지지번주소": "address",
        "위도": "latitude",
        "경도": "longitude",
        "총길이": "length",
        "점포수": "count"
    }
)

df.to_csv(
    "data/processed/streetProcessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("streetProcessed.csv 저장 완료")
print(df.shape)
print(df.head())
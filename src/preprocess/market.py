import pandas as pd

df = pd.read_csv(
    "data/raw/market.csv",
    encoding="cp949"
)

columns = [
    "시장명",
    "시장유형",
    "소재지도로명주소",
    "소재지지번주소",
    "위도",
    "경도",
    "시장개설주기",
    "점포수",
    "취급품목"
]

df = df[columns]

df = df.rename(
    columns={
        "시장명": "name",
        "시장유형": "category",
        "소재지도로명주소": "roadAddress",
        "소재지지번주소": "address",
        "위도": "latitude",
        "경도": "longitude",
        "시장개설주기": "period",
        "점포수": "count",
        "취급품목": "items"
    }
)

df.to_csv(
    "data/processed/marketProcessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("marketProcessed.csv 저장 완료")
print(df.shape)
print(df.head())
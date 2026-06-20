import pandas as pd

df = pd.read_csv(
    "data/raw/전국전통시장표준데이터.csv",
    encoding="cp949"
)

columns = [
    "시장명",
    "시장유형",
    "소재지도로명주소",
    "시장개설주기",
    "취급품목"
]

df = df[columns]

df = df.rename(
    columns={
        "시장명": "name",
        "시장유형": "category",
        "소재지도로명주소": "address",
        "시장개설주기": "period",
        "취급품목": "items"
    }
)

df.to_csv(
    "data/processed/market.csv",
    index=False,
    encoding="utf-8-sig"
)

print("market.csv 저장 완료")
print(df.shape)
print(df.head())
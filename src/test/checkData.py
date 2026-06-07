import pandas as pd

festival = pd.read_csv(
    "data/processed/festivalProcessed.csv",
    encoding="utf-8-sig"
)

heritage = pd.read_csv(
    "data/processed/heritageProcessed.csv",
    encoding="utf-8-sig"
)

print("===== Festival 컬럼 =====")
print(festival.columns.tolist())

print("\n===== Heritage 컬럼 =====")
print(heritage.columns.tolist())

print("\n===== Festival 주소 결측치 =====")
print(festival["address"].isna().sum())
print("전체:", len(festival))

print("\n===== Heritage 주소 결측치 =====")
print(heritage["address"].isna().sum())
print("전체:", len(heritage))

print(
    festival[
        [
            "name",
            "address"
        ]
    ].head(20)
)

festival_raw = pd.read_csv(
    "data/raw/festival.csv",
    encoding="utf-8-sig"
)

print(festival_raw.columns.tolist())
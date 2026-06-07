import pandas as pd

festival = pd.read_csv(
    "data/processed/festivalProcessed.csv"
)

heritage = pd.read_csv(
    "data/processed/heritageProcessed.csv"
)

market = pd.read_csv(
    "data/processed/marketProcessed.csv"
)

street = pd.read_csv(
    "data/processed/streetProcessed.csv"
)

# 데이터 유형 지정
festival["source"] = "축제"
heritage["source"] = "문화재"
market["source"] = "전통시장"
street["source"] = "특화거리"

# 전체 컬럼 수집
all_columns = set()

for df in [
    festival,
    heritage,
    market,
    street
]:
    all_columns.update(df.columns)

all_columns = list(all_columns)

# 없는 컬럼 자동 생성
festival = festival.reindex(columns=all_columns)
heritage = heritage.reindex(columns=all_columns)
market = market.reindex(columns=all_columns)
street = street.reindex(columns=all_columns)

# 데이터 병합
merged = pd.concat(
    [
        festival,
        heritage,
        market,
        street
    ],
    ignore_index=True
)

# 컬럼 순서 통일
column_order = [
    "source",
    "name",
    "category",
    "address",
    "period",
    "description",
    "items"
]

existing_columns = [
    col for col in column_order
    if col in merged.columns
]

other_columns = [
    col for col in merged.columns
    if col not in existing_columns
]

merged = merged[
    existing_columns + other_columns
]

merged.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Data.csv 저장 완료")
print(merged.shape)
print(merged.head())
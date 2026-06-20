import pandas as pd

festival = pd.read_csv(
    "data/processed/festival.csv"
)

heritage = pd.read_csv(
    "data/processed/heritage.csv"
)

localCulture = pd.read_csv(
    "data/processed/localCulture.csv"
)

event = pd.read_csv(
    "data/processed/event.csv"
)

performance = pd.read_csv(
    "data/processed/performance.csv"
)

market = pd.read_csv(
    "data/processed/market.csv"
)

street = pd.read_csv(
    "data/processed/street.csv"
)

# ==========================================
# source 지정
# ==========================================

festival["source"] = "축제"

heritage["source"] = "문화재"

localCulture["source"] = "향토문화"

event["source"] = "행사"

performance["source"] = "공연"

market["source"] = "전통시장"

street["source"] = "특화거리"

# ==========================================
# 전체 컬럼 수집
# ==========================================

all_columns = set()

for df in [
    festival,
    heritage,
    localCulture,
    event,
    performance,
    market,
    street
]:
    all_columns.update(df.columns)

all_columns = list(all_columns)

# ==========================================
# 컬럼 통일
# ==========================================

festival = festival.reindex(
    columns=all_columns
)

heritage = heritage.reindex(
    columns=all_columns
)

localCulture = localCulture.reindex(
    columns=all_columns
)

event = event.reindex(
    columns=all_columns
)

performance = performance.reindex(
    columns=all_columns
)

market = market.reindex(
    columns=all_columns
)

street = street.reindex(
    columns=all_columns
)

# ==========================================
# 병합
# ==========================================

merged = pd.concat(
    [
        festival,
        heritage,
        localCulture,
        event,
        performance,
        market,
        street
    ],
    ignore_index=True
)

# ==========================================
# 중복 제거
# ==========================================

merged = merged.drop_duplicates(
    subset=["name"]
)

# ==========================================
# 컬럼 순서 정리
# ==========================================

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
    col
    for col in column_order
    if col in merged.columns
]

other_columns = [
    col
    for col in merged.columns
    if col not in existing_columns
]

merged = merged[
    existing_columns + other_columns
]

# ==========================================
# 저장
# ==========================================

merged.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Data.csv 저장 완료")
print("총 데이터 수 :", len(merged))
print("컬럼 수 :", len(merged.columns))
print()

print(
    merged["source"]
    .value_counts()
)
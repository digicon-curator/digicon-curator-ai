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

festival["source"] = "festival"
heritage["source"] = "heritage"
market["source"] = "market"
street["source"] = "street"

all_columns = set()

for df in [
    festival,
    heritage,
    market,
    street
]:
    all_columns.update(df.columns)

all_columns = list(all_columns)

festival = festival.reindex(columns=all_columns)
heritage = heritage.reindex(columns=all_columns)
market = market.reindex(columns=all_columns)
street = street.reindex(columns=all_columns)

merged = pd.concat(
    [
        festival,
        heritage,
        market,
        street
    ],
    ignore_index=True
)

merged.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Data.csv 저장 완료")
print(merged.shape)
print(merged.head())
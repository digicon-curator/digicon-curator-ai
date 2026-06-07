import pandas as pd

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

# 결측치 처리
df = df.fillna("")

df["content"] = (
    "데이터유형: " + df["source"].astype(str)
    + "\n이름: " + df["name"].astype(str)
    + "\n위치: " + df["address"].astype(str)
    + "\n시기: " + df["period"].astype(str)
    + "\n설명: " + df["description"].astype(str)
    + "\n분류: " + df["category"].astype(str)
    + "\n관련 품목: " + df["items"].astype(str)
)

df.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("content 컬럼 생성 완료")
print(df[["name", "content"]].head())
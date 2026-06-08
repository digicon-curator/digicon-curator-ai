import pandas as pd

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

# 컬럼별 결측치 처리
if "source" in df.columns:
    df["source"] = df["source"].fillna("기타")

if "name" in df.columns:
    df["name"] = df["name"].fillna("이름 없음")

if "category" in df.columns:
    df["category"] = df["category"].fillna("기타")

if "address" in df.columns:
    df["address"] = df["address"].fillna("주소 정보 없음")

if "period" in df.columns:
    df["period"] = df["period"].fillna("정보 없음")

if "description" in df.columns:
    df["description"] = df["description"].fillna("설명 정보 없음")

if "items" in df.columns:
    df["items"] = df["items"].fillna("없음")

# RAG용 content 생성
df["content"] = (
    "유형: " + df["source"].astype(str)
    + "\n이름: " + df["name"].astype(str)
    + "\n분류: " + df["category"].astype(str)
    + "\n주소: " + df["address"].astype(str)
    + "\n기간: " + df["period"].astype(str)
    + "\n설명: " + df["description"].astype(str)
    + "\n관련 품목: " + df["items"].astype(str)
)

df.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("content 컬럼 생성 완료")
print(
    df[
        [
            "name",
            "content"
        ]
    ].head()
)
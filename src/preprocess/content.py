import re
import pandas as pd

# ==========================================
# Data 로드
# ==========================================

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 텍스트 정제 함수
# ==========================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # 줄바꿈 제거
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # 특수문자 제거
    text = re.sub(
        r"[○●▶■◆※]",
        " ",
        text
    )

    # 느낌표 정리
    text = re.sub(
        r"!+",
        " ",
        text
    )

    # 연속 공백 제거
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ==========================================
# Content 생성
# ==========================================

contents = []

for _, row in df.iterrows():

    source = clean_text(
        row.get("source", "")
    )

    name = clean_text(
        row.get("name", "")
    )

    category = clean_text(
        row.get("category", "")
    )

    address = clean_text(
        row.get("address", "")
    )

    period = clean_text(
        row.get("period", "")
    )

    description = clean_text(
        row.get("description", "")
    )

    items = clean_text(
        row.get("items", "")
    )

    parts = []

    if source:
        parts.append(
            f"유형: {source}"
        )

    if name:
        parts.append(
            f"이름: {name}"
        )

    if category:
        parts.append(
            f"분류: {category}"
        )

    if address:
        parts.append(
            f"주소: {address}"
        )

    if period:
        parts.append(
            f"기간: {period}"
        )

    if description:
        parts.append(
            f"설명: {description}"
        )

    if items:
        parts.append(
            f"부가정보: {items}"
        )

    content = "\n".join(parts)

    contents.append(
        content
    )

# ==========================================
# 저장
# ==========================================

df["content"] = contents

df.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("content 생성 완료")
print(df.shape)

print("\n===== Content 예시 =====\n")
print(df["content"].iloc[0])
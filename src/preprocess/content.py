import re
import pandas as pd

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # 줄바꿈 제거
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # 특수기호 제거
    text = re.sub(r"[○●▶■◆※]", " ", text)

    # 느낌표 여러 개 제거
    text = re.sub(r"!+", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text)

    return text.strip()

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

    content = f"""
[유형]
{source}

[이름]
{name}

[분류]
{category}

[지역]
{address}

[기간]
{period}

[설명]
{description}

[관련정보]
{items}
"""

    contents.append(
        content.strip()
    )

df["content"] = contents

df.to_csv(
    "data/processed/Data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("content 생성 완료")
print(df.shape)
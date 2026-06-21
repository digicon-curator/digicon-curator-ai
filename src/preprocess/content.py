import html
import re

import pandas as pd

try:
    from src.rag.paths import get_data_path
except ModuleNotFoundError:
    import os

    def get_data_path():
        curated_path = "data/processed/Data_curated.csv"
        original_path = "data/processed/Data.csv"
        return os.getenv(
            "CURATOR_DATA_PATH",
            curated_path if os.path.exists(curated_path) else original_path,
        )

MEANINGLESS_VALUES = {
    "",
    "-",
    "--",
    "nan",
    "none",
    "null",
    "없음",
    "해당없음",
    "해당 없음",
    "상세정보 없음",
    "설명 없음",
    "정보 없음",
    "미상",
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = text.replace("<", " ").replace(">", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in MEANINGLESS_VALUES:
        return ""

    return text


def build_region(address):
    address = clean_text(address)
    if not address:
        return ""

    parts = address.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]


def add_part(parts, label, value):
    value = clean_text(value)
    if value:
        parts.append(f"{label}: {value}")


data_path = get_data_path()
df = pd.read_csv(data_path, encoding="utf-8-sig")

contents = []

for _, row in df.iterrows():
    parts = []

    add_part(parts, "이름", row.get("name", ""))
    add_part(parts, "유형", row.get("source", ""))
    add_part(parts, "분류", row.get("category", ""))
    add_part(parts, "지역", build_region(row.get("address", "")))
    add_part(parts, "기간", row.get("period", ""))
    add_part(parts, "설명", row.get("description", ""))
    add_part(parts, "부가 정보", row.get("items", ""))

    contents.append("\n".join(parts))

df["content"] = contents

df.to_csv(data_path, index=False, encoding="utf-8-sig")

print("content 생성 완료")
print(df.shape)
print("\n===== Content 예시 =====\n")
print(df["content"].iloc[0])

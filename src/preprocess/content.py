import html
import re

import pandas as pd

try:
    from src.rag.paths import getDataPath
except ModuleNotFoundError:
    import os

    def getDataPath():
        curatedPath = "data/processed/dataCurated.csv"
        originalPath = "data/processed/Data.csv"
        return os.getenv(
            "CURATOR_DATA_PATH",
            curatedPath if os.path.exists(curatedPath) else originalPath,
        )

meaninglessValues = {
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


def cleanText(text):
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

    if text.lower() in meaninglessValues:
        return ""

    return text


def buildRegion(address):
    address = cleanText(address)
    if not address:
        return ""

    parts = address.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]


def addPart(parts, label, value):
    value = cleanText(value)
    if value:
        parts.append(f"{label}: {value}")


dataPath = getDataPath()
df = pd.read_csv(dataPath, encoding="utf-8-sig")

contents = []

for _, row in df.iterrows():
    parts = []

    addPart(parts, "이름", row.get("name", ""))
    addPart(parts, "유형", row.get("source", ""))
    addPart(parts, "분류", row.get("category", ""))
    addPart(parts, "지역", buildRegion(row.get("address", "")))
    addPart(parts, "기간", row.get("period", ""))
    addPart(parts, "설명", row.get("description", ""))
    addPart(parts, "부가 정보", row.get("items", ""))

    contents.append("\n".join(parts))

df["content"] = contents

df.to_csv(dataPath, index=False, encoding="utf-8-sig")

print("content 생성 완료")
print(df.shape)
print("\n===== Content 예시 =====\n")
print(df["content"].iloc[0])

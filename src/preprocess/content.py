import html
import re
import sys
from pathlib import Path

import pandas as pd

projectRoot = Path(__file__).resolve().parents[2]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

try:
    from src.rag.paths import getDataPath
    from src.rag.utils import normalizeLocalRegion
except ModuleNotFoundError:
    ragPath = projectRoot / "src" / "rag"
    if str(ragPath) not in sys.path:
        sys.path.insert(0, str(ragPath))

    from paths import getDataPath
    from utils import normalizeLocalRegion


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

noisePatterns = [
    r"https?://\S+",
    r"www\.\S+",
    r"\S+@\S+",
    r"\b\d{2,4}-\d{3,4}-\d{4}\b",
    r"\b\d{3}-\d{3,4}-\d{4}\b",
    r"\(?\s*(문의|전화|홈페이지|사이트|참고|출처|자료제공|담당부서|담당자)\s*[:：][^)]+",
    r"\[[^\]]*(문의|전화|홈페이지|사이트|출처|담당)[^\]]*\]",
    r"\([^\)]*(문의|전화|홈페이지|사이트|출처|담당)[^\)]*\)",
]

genericDescriptions = {
    "자세한 내용은 홈페이지 참고",
    "자세한 사항은 홈페이지를 참고하세요",
    "홈페이지 참조",
    "홈페이지 참고",
    "상세 내용 참고",
    "상세정보 없음",
}


def cleanText(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)

    for pattern in noisePatterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = text.replace("<", " ").replace(">", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[○●◎◇◆■□▲△▶▷※★☆]+", " ", text)
    text = re.sub(r"[!！?？]{2,}", " ", text)
    text = re.sub(r"[-_=]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in meaninglessValues:
        return ""

    if text in genericDescriptions:
        return ""

    return text


def cleanDescription(value):
    text = cleanText(value)
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", text)
    uniqueSentences = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        uniqueSentences.append(sentence)

    text = " ".join(uniqueSentences)

    if len(text) <= 20:
        return ""

    return text


def addPart(parts, label, value):
    value = cleanText(value)
    if value:
        parts.append(f"{label}: {value}")


dataPath = getDataPath()
df = pd.read_csv(dataPath, encoding="utf-8-sig")

for column in ["name", "source", "category", "address", "period", "items"]:
    if column not in df.columns:
        df[column] = ""
    df[column] = df[column].apply(cleanText)

if "description" not in df.columns:
    df["description"] = ""
df["description"] = df["description"].apply(cleanDescription)
df["localRegion"] = df["address"].apply(normalizeLocalRegion)

df = df[df["name"].astype(str).str.len() > 0]
df = df[df["description"].astype(str).str.len() > 20]

contents = []

for _, row in df.iterrows():
    parts = []

    addPart(parts, "이름", row.get("name", ""))
    addPart(parts, "유형", row.get("source", ""))
    addPart(parts, "분류", row.get("category", ""))
    addPart(parts, "지역", row.get("localRegion", ""))
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

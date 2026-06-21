import html
import os
import re

import pandas as pd


inputPath = os.getenv("CURATOR_SOURCE_DATA_PATH", "data/processed/Data.csv")
outputPath = os.getenv("CURATOR_CURATED_DATA_PATH", "data/processed/dataCurated.csv")
targetTotal = int(os.getenv("CURATOR_TARGET_TOTAL", "22000"))
minDescriptionLen = int(os.getenv("CURATOR_MIN_DESCRIPTION_LEN", "20"))

sourceLimits = {
    "문화재": 5000,
    "향토문화": 5000,
    "행사": 5000,
    "축제": 4000,
    "전통시장": 1500,
    "공연": 1000,
    "특화거리": 500,
}

badTexts = {
    "",
    "-",
    "--",
    "nan",
    "none",
    "null",
    "없음",
    "해당없음",
    "해당 없음",
    "설명 없음",
    "정보 없음",
    "상세정보 없음",
    "미상",
}


def cleanText(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"</?[a-zA-Z][^>]*>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"</?[a-zA-Z][^>]*>", " ", value)
    value = re.sub(r"&[a-zA-Z0-9#]+;", " ", value)
    value = value.replace("<", " ").replace(">", " ")
    value = re.sub(r"\s+", " ", value).strip()

    return "" if value.lower() in badTexts else value


def normalizeRegion(address):
    address = cleanText(address)
    if not address:
        return "지역 미상"

    parts = address.split()
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return parts[0]


def addQualityColumns(df):
    for column in ["name", "source", "category", "address", "period", "description", "items"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].apply(cleanText)

    df["region"] = df["address"].apply(normalizeRegion)
    df["descriptionLen"] = df["description"].str.len()
    df["qualityScore"] = (
        df["descriptionLen"]
        + df["category"].str.len().clip(upper=30)
        + df["items"].str.len().clip(upper=50)
        + df["period"].str.len().clip(upper=20)
    )

    return df


def balancedSample(sourceDf, limit):
    if len(sourceDf) <= limit:
        return sourceDf

    perRegion = max(1, limit // max(1, sourceDf["region"].nunique()))

    sampled = (
        sourceDf.sort_values("qualityScore", ascending=False)
        .groupby("region", group_keys=False)
        .head(perRegion)
    )

    if len(sampled) < limit:
        remain = sourceDf.drop(sampled.index, errors="ignore")
        sampled = pd.concat(
            [
                sampled,
                remain.sort_values("qualityScore", ascending=False).head(limit - len(sampled)),
            ]
        )

    return sampled.head(limit)


def buildContent(row):
    parts = []
    fields = [
        ("이름", row.get("name", "")),
        ("유형", row.get("source", "")),
        ("분류", row.get("category", "")),
        ("지역", row.get("region", "")),
        ("기간", row.get("period", "")),
        ("설명", row.get("description", "")),
        ("부가 정보", row.get("items", "")),
    ]

    for label, value in fields:
        value = cleanText(value)
        if value:
            parts.append(f"{label}: {value}")

    return "\n".join(parts)


df = pd.read_csv(inputPath, encoding="utf-8-sig")
df["originalIndex"] = df.index
df = addQualityColumns(df)

df = df[df["descriptionLen"] > minDescriptionLen]
df = df.drop_duplicates(subset=["source", "name", "address"], keep="first")
df = df.drop_duplicates(subset=["name", "address"], keep="first")

samples = []
for source, limit in sourceLimits.items():
    sourceDf = df[df["source"] == source]
    samples.append(balancedSample(sourceDf, limit))

curated = pd.concat(samples, ignore_index=True)

if len(curated) < targetTotal:
    selectedOriginalIndexes = set(curated["originalIndex"])
    remain = df[~df["originalIndex"].isin(selectedOriginalIndexes)]
    curated = pd.concat(
        [
            curated,
            remain.sort_values("qualityScore", ascending=False).head(targetTotal - len(curated)),
        ],
        ignore_index=True,
    )

curated = curated.sort_values(["source", "region", "qualityScore"], ascending=[True, True, False])
curated = curated.head(targetTotal).reset_index(drop=True)
curated["content"] = curated.apply(buildContent, axis=1)

curated.to_csv(outputPath, index=False, encoding="utf-8-sig")

print("curated 데이터 생성 완료")
print("입력 데이터 수:", len(df))
print("출력 데이터 수:", len(curated))
print("\n===== source 분포 =====")
print(curated["source"].value_counts())
print("\n===== 지역 상위 20 =====")
print(curated["region"].value_counts().head(20))

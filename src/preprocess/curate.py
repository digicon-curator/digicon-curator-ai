import html
import os
import re

import pandas as pd


INPUT_PATH = os.getenv("CURATOR_SOURCE_DATA_PATH", "data/processed/Data.csv")
OUTPUT_PATH = os.getenv("CURATOR_CURATED_DATA_PATH", "data/processed/Data_curated.csv")
TARGET_TOTAL = int(os.getenv("CURATOR_TARGET_TOTAL", "22000"))
MIN_DESCRIPTION_LEN = int(os.getenv("CURATOR_MIN_DESCRIPTION_LEN", "20"))

SOURCE_LIMITS = {
    "문화재": 5000,
    "향토문화": 5000,
    "행사": 5000,
    "축제": 4000,
    "전통시장": 1500,
    "공연": 1000,
    "특화거리": 500,
}

BAD_TEXTS = {
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


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"</?[a-zA-Z][^>]*>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"</?[a-zA-Z][^>]*>", " ", value)
    value = re.sub(r"&[a-zA-Z0-9#]+;", " ", value)
    value = value.replace("<", " ").replace(">", " ")
    value = re.sub(r"\s+", " ", value).strip()

    return "" if value.lower() in BAD_TEXTS else value


def normalize_region(address):
    address = clean_text(address)
    if not address:
        return "지역 미상"

    parts = address.split()
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return parts[0]


def add_quality_columns(df):
    for column in ["name", "source", "category", "address", "period", "description", "items"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].apply(clean_text)

    df["region"] = df["address"].apply(normalize_region)
    df["description_len"] = df["description"].str.len()
    df["quality_score"] = (
        df["description_len"]
        + df["category"].str.len().clip(upper=30)
        + df["items"].str.len().clip(upper=50)
        + df["period"].str.len().clip(upper=20)
    )

    return df


def balanced_sample(source_df, limit):
    if len(source_df) <= limit:
        return source_df

    per_region = max(1, limit // max(1, source_df["region"].nunique()))

    sampled = (
        source_df.sort_values("quality_score", ascending=False)
        .groupby("region", group_keys=False)
        .head(per_region)
    )

    if len(sampled) < limit:
        remain = source_df.drop(sampled.index, errors="ignore")
        sampled = pd.concat(
            [
                sampled,
                remain.sort_values("quality_score", ascending=False).head(limit - len(sampled)),
            ]
        )

    return sampled.head(limit)


def build_content(row):
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
        value = clean_text(value)
        if value:
            parts.append(f"{label}: {value}")

    return "\n".join(parts)


df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
df["original_index"] = df.index
df = add_quality_columns(df)

df = df[df["description_len"] > MIN_DESCRIPTION_LEN]
df = df.drop_duplicates(subset=["source", "name", "address"], keep="first")
df = df.drop_duplicates(subset=["name", "address"], keep="first")

samples = []
for source, limit in SOURCE_LIMITS.items():
    source_df = df[df["source"] == source]
    samples.append(balanced_sample(source_df, limit))

curated = pd.concat(samples, ignore_index=True)

if len(curated) < TARGET_TOTAL:
    selected_original_indexes = set(curated["original_index"])
    remain = df[~df["original_index"].isin(selected_original_indexes)]
    curated = pd.concat(
        [
            curated,
            remain.sort_values("quality_score", ascending=False).head(TARGET_TOTAL - len(curated)),
        ],
        ignore_index=True,
    )

curated = curated.sort_values(["source", "region", "quality_score"], ascending=[True, True, False])
curated = curated.head(TARGET_TOTAL).reset_index(drop=True)
curated["content"] = curated.apply(build_content, axis=1)

curated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("curated 데이터 생성 완료")
print("입력 데이터 수:", len(df))
print("출력 데이터 수:", len(curated))
print("\n===== source 분포 =====")
print(curated["source"].value_counts())
print("\n===== 지역 상위 20 =====")
print(curated["region"].value_counts().head(20))

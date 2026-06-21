import re

import faiss
import numpy as np
import pandas as pd


regionAliases = {
    "서울": "서울",
    "서울시": "서울",
    "서울특별시": "서울",
    "부산": "부산",
    "부산시": "부산",
    "부산광역시": "부산",
    "대구": "대구",
    "대구시": "대구",
    "대구광역시": "대구",
    "인천": "인천",
    "인천시": "인천",
    "인천광역시": "인천",
    "광주": "광주",
    "광주시": "광주",
    "광주광역시": "광주",
    "대전": "대전",
    "대전시": "대전",
    "대전광역시": "대전",
    "울산": "울산",
    "울산시": "울산",
    "울산광역시": "울산",
    "세종": "세종",
    "세종시": "세종",
    "세종특별자치시": "세종",
    "경기": "경기",
    "경기도": "경기",
    "강원": "강원",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "충북": "충북",
    "충청북도": "충북",
    "충남": "충남",
    "충청남도": "충남",
    "전북": "전북",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전남": "전남",
    "전라남도": "전남",
    "경북": "경북",
    "경상북도": "경북",
    "경남": "경남",
    "경상남도": "경남",
    "제주": "제주",
    "제주도": "제주",
    "제주특별자치도": "제주",
}

sourceOrder = ["문화재", "향토문화", "축제", "행사", "공연", "전통시장", "특화거리"]
badTexts = {"", "-", "--", "nan", "none", "null", "없음", "해당없음", "해당 없음", "설명 없음", "정보 없음"}


def cleanValue(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return "" if value.lower() in badTexts else value


def normalizeRegion(value):
    value = cleanValue(value)
    if not value:
        return ""

    token = value.split()[0]
    return regionAliases.get(token, regionAliases.get(value, token))


def detectRegion(text):
    text = cleanValue(text)
    if not text:
        return ""

    aliases = sorted(regionAliases.keys(), key=len, reverse=True)
    for alias in aliases:
        if alias in text:
            return regionAliases[alias]

    return ""


def detectRegionFromData(text, df):
    text = cleanValue(text)
    if not text or "address" not in df.columns:
        return ""

    candidates = set()
    for address in df["address"].dropna().astype(str):
        parts = address.split()
        candidates.update(parts[:3])

    for candidate in sorted(candidates, key=len, reverse=True):
        if len(candidate) < 2:
            continue
        shortName = re.sub(r"(특별시|광역시|특별자치시|특별자치도|시|군|구|도)$", "", candidate)
        if candidate in text or (len(shortName) >= 2 and shortName in text):
            return candidate

    return ""


def filterByRegion(df, region):
    if not region or "address" not in df.columns:
        return df

    pattern = "|".join(
        re.escape(alias)
        for alias, normalized in regionAliases.items()
        if normalized == region
    )
    if not pattern:
        pattern = re.escape(region)

    return df[df["address"].fillna("").astype(str).str.contains(pattern, regex=True, na=False)]


def applyQualityFilter(df, minDescriptionLen=20):
    if "description" not in df.columns:
        return df

    descriptions = df["description"].fillna("").astype(str).str.strip()
    return df[descriptions.str.len() > minDescriptionLen]


def dedupeRows(rows):
    results = []
    seen = set()

    for _, row in rows.iterrows():
        name = cleanValue(row.get("name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        results.append(row)

    return results


def faissSearchRows(df, embeddings, embeddingModel, query, k=40):
    if df.empty:
        return []

    if "originalIndex" in df.columns and df["originalIndex"].astype(int).max() < len(embeddings):
        embeddingIndices = df["originalIndex"].astype(int).to_numpy()
    else:
        embeddingIndices = df.index.to_numpy()

    validMask = (embeddingIndices >= 0) & (embeddingIndices < len(embeddings))
    if not validMask.any():
        return []

    sourceDf = df.loc[validMask]
    embeddingIndices = embeddingIndices[validMask]
    subsetEmbeddings = embeddings[embeddingIndices].astype("float32")
    queryEmbedding = embeddingModel.encode([query], convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(subsetEmbeddings.shape[1])
    index.add(subsetEmbeddings)

    _, indices = index.search(queryEmbedding, min(k, len(sourceDf)))
    matchedIndices = sourceDf.index.to_numpy()[indices[0]]

    return dedupeRows(sourceDf.loc[matchedIndices])


def balancedBySource(rows, limit=15):
    buckets = {source: [] for source in sourceOrder}
    extraBuckets = {}

    for row in rows:
        source = cleanValue(row.get("source", ""))
        if source in buckets:
            buckets[source].append(row)
        else:
            extraBuckets.setdefault(source, []).append(row)

    orderedSources = sourceOrder + [source for source in extraBuckets if source]
    results = []

    while len(results) < limit:
        added = False

        for source in orderedSources:
            bucket = buckets.get(source, extraBuckets.get(source, []))
            if bucket:
                results.append(bucket.pop(0))
                added = True

                if len(results) >= limit:
                    break

        if not added:
            break

    return results


def buildContext(rows):
    blocks = []

    for row in rows:
        blocks.append(
            f"""
이름: {row.get('name', '')}
유형: {row.get('source', '')}
분류: {row.get('category', '')}
주소: {row.get('address', '')}
기간: {row.get('period', '')}
설명: {row.get('description', '')}
부가 정보: {row.get('items', '')}
""".strip()
        )

    return "\n\n----------------------------------------\n\n".join(blocks)

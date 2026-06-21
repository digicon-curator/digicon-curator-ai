import re

import faiss
import numpy as np
import pandas as pd


REGION_ALIASES = {
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

SOURCE_ORDER = ["문화재", "향토문화", "축제", "행사", "공연", "전통시장", "특화거리"]
BAD_TEXTS = {"", "-", "--", "nan", "none", "null", "없음", "해당없음", "해당 없음", "설명 없음", "정보 없음"}


def clean_value(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return "" if value.lower() in BAD_TEXTS else value


def normalize_region(value):
    value = clean_value(value)
    if not value:
        return ""

    token = value.split()[0]
    return REGION_ALIASES.get(token, REGION_ALIASES.get(value, token))


def detect_region(text):
    text = clean_value(text)
    if not text:
        return ""

    aliases = sorted(REGION_ALIASES.keys(), key=len, reverse=True)
    for alias in aliases:
        if alias in text:
            return REGION_ALIASES[alias]

    return ""


def detect_region_from_data(text, df):
    text = clean_value(text)
    if not text or "address" not in df.columns:
        return ""

    candidates = set()
    for address in df["address"].dropna().astype(str):
        parts = address.split()
        candidates.update(parts[:3])

    for candidate in sorted(candidates, key=len, reverse=True):
        if len(candidate) < 2:
            continue
        short_name = re.sub(r"(특별시|광역시|특별자치시|특별자치도|시|군|구|도)$", "", candidate)
        if candidate in text or (len(short_name) >= 2 and short_name in text):
            return candidate

    return ""


def filter_by_region(df, region):
    if not region or "address" not in df.columns:
        return df

    pattern = "|".join(
        re.escape(alias)
        for alias, normalized in REGION_ALIASES.items()
        if normalized == region
    )
    if not pattern:
        pattern = re.escape(region)

    return df[df["address"].fillna("").astype(str).str.contains(pattern, regex=True, na=False)]


def apply_quality_filter(df, min_description_len=20):
    if "description" not in df.columns:
        return df

    descriptions = df["description"].fillna("").astype(str).str.strip()
    return df[descriptions.str.len() > min_description_len]


def dedupe_rows(rows):
    results = []
    seen = set()

    for _, row in rows.iterrows():
        name = clean_value(row.get("name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        results.append(row)

    return results


def faiss_search_rows(df, embeddings, embedding_model, query, k=40):
    if df.empty:
        return []

    if len(embeddings) == len(df):
        embedding_indices = df.index.to_numpy()
    elif "original_index" in df.columns:
        embedding_indices = df["original_index"].astype(int).to_numpy()
    else:
        embedding_indices = df.index.to_numpy()

    valid_mask = embedding_indices < len(embeddings)
    if not valid_mask.any():
        return []

    source_df = df.loc[valid_mask]
    embedding_indices = embedding_indices[valid_mask]
    subset_embeddings = embeddings[embedding_indices].astype("float32")
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(subset_embeddings.shape[1])
    index.add(subset_embeddings)

    _, indices = index.search(query_embedding, min(k, len(row_indices)))
    matched_indices = source_df.index.to_numpy()[indices[0]]

    return dedupe_rows(source_df.loc[matched_indices])


def balanced_by_source(rows, limit=15):
    buckets = {source: [] for source in SOURCE_ORDER}
    extra_buckets = {}

    for row in rows:
        source = clean_value(row.get("source", ""))
        if source in buckets:
            buckets[source].append(row)
        else:
            extra_buckets.setdefault(source, []).append(row)

    ordered_sources = SOURCE_ORDER + [source for source in extra_buckets if source]
    results = []

    while len(results) < limit:
        added = False

        for source in ordered_sources:
            bucket = buckets.get(source, extra_buckets.get(source, []))
            if bucket:
                results.append(bucket.pop(0))
                added = True

                if len(results) >= limit:
                    break

        if not added:
            break

    return results


def build_context(rows):
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

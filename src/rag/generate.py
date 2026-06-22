import json
import os
import re
from collections import Counter
from functools import lru_cache

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.rag.paths import getDataPath, getEmbeddingPath, originalEmbeddingPath
from src.rag.utils import (
    applyQualityFilter,
    detectLocalRegionFromData,
    detectRegion,
    detectRegionFromData,
    faissSearchRows,
    filterByRegion,
    filterByLocalRegion,
    normalizeLocalRegion,
    normalizeRegion,
    sourceBalancedFaissSearch,
)


CATEGORY_HERITAGE = "\ubb38\ud654\uc7ac"
CATEGORY_LOCAL = "\ud5a5\ud1a0\ubb38\ud654"
CATEGORY_FESTIVAL = "\ucd95\uc81c"
CATEGORY_EVENT = "\ud589\uc0ac"
CATEGORY_PERFORMANCE = "\uacf5\uc5f0"
CATEGORY_MARKET = "\uc804\ud1b5\uc2dc\uc7a5"
CATEGORY_STREET = "\ud2b9\ud654\uac70\ub9ac"
UNKNOWN_REGION = "\uc9c0\uc5ed \ubbf8\uc0c1"
UNKNOWN_TIME = "\uc2dc\uae30 \ubbf8\uc0c1"

ALLOWED_CATEGORIES = [
    CATEGORY_HERITAGE,
    CATEGORY_LOCAL,
    CATEGORY_FESTIVAL,
    CATEGORY_EVENT,
    CATEGORY_PERFORMANCE,
    CATEGORY_MARKET,
    CATEGORY_STREET,
]


def clean_value(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_category(value):
    value = clean_value(value)
    return value if value in ALLOWED_CATEGORIES else CATEGORY_LOCAL


def extract_json_object(text):
    text = clean_value(text)
    if not text:
        raise ValueError("empty model response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("model response does not contain a JSON object")


def row_to_context(row):
    return "\n".join(
        [
            f"name: {clean_value(row.get('name'))}",
            f"source: {clean_value(row.get('source'))}",
            f"category: {clean_value(row.get('category'))}",
            f"address: {clean_value(row.get('address'))}",
            f"period: {clean_value(row.get('period'))}",
            f"description: {clean_value(row.get('description'))}",
            f"items: {clean_value(row.get('items'))}",
        ]
    )


def build_context(rows):
    return "\n\n---\n\n".join(row_to_context(row) for row in rows)


def fallback_story(rows, selected_region):
    first = rows[0] if rows else {}
    title = clean_value(first.get("name")) or f"{selected_region or UNKNOWN_REGION} story"
    description = clean_value(first.get("description")) or "Generated local culture story."
    period = clean_value(first.get("period")) or UNKNOWN_TIME

    return {
        "title": title,
        "description": description[:240],
        "category": CATEGORY_LOCAL,
        "region": selected_region or normalizeLocalRegion(first.get("address", "")) or UNKNOWN_REGION,
        "timeline": [
            {
                "year": period,
                "title": title,
                "text": description[:240],
            }
        ],
    }


def validate_story(data, rows, selected_region):
    fallback = fallback_story(rows, selected_region)

    title = clean_value(data.get("title")) or fallback["title"]
    description = clean_value(data.get("description")) or fallback["description"]
    category = normalize_category(data.get("category") or fallback["category"])
    region = clean_value(data.get("region")) or fallback["region"]

    timeline = data.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        timeline = fallback["timeline"]

    normalized_timeline = []
    for item in timeline[:5]:
        if not isinstance(item, dict):
            continue

        year = clean_value(item.get("year")) or UNKNOWN_TIME
        item_title = clean_value(item.get("title")) or title
        text = clean_value(item.get("text")) or description
        normalized_timeline.append({"year": year, "title": item_title, "text": text})

    if not normalized_timeline:
        normalized_timeline = fallback["timeline"]

    return {
        "title": title,
        "description": description,
        "category": category,
        "region": region,
        "timeline": normalized_timeline,
    }


def row_to_place(row):
    return {
        "name": clean_value(row.get("name")),
        "source": clean_value(row.get("source")),
        "category": clean_value(row.get("category")),
        "address": clean_value(row.get("address")),
        "period": clean_value(row.get("period")),
        "description": clean_value(row.get("description")),
        "items": clean_value(row.get("items")),
        "region": normalizeLocalRegion(row.get("address", "")) or normalizeRegion(row.get("address", "")),
    }


def rows_to_places(rows, limit=10):
    return [row_to_place(row) for row in rows[:limit]]


def split_keywords(value):
    value = clean_value(value)
    if not value:
        return []

    tokens = re.split(r"[,/|:;\[\]\(\)\s]+", value)
    return [token.strip() for token in tokens if len(token.strip()) >= 2 and not token.isdigit()]


def extract_culture_keywords(df):
    counter = Counter()
    columns = [col for col in ["category", "name", "description", "items"] if col in df.columns]

    for col in columns:
        for value in df[col].dropna().astype(str):
            for token in split_keywords(value):
                counter[token] += 1

    return dict(counter.most_common(30))


class CuratorAIService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

        self.embedding_path = getEmbeddingPath()
        self.df = pd.read_csv(getDataPath(), encoding="utf-8-sig")
        self.embeddings = np.load(self.embedding_path)
        self.use_original_index = (
            self.embedding_path == originalEmbeddingPath and "originalIndex" in self.df.columns
        )
        self.quality_df = applyQualityFilter(self.df, minDescriptionLen=20)

    def generate_json(self, prompt):
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return extract_json_object(response.text)

    def search_rows(self, query, limit=15):
        region = detectRegionFromData(query, self.df) or detectRegion(query)
        candidate_df = self.df

        if region:
            region_df = filterByRegion(self.df, region)
            if not region_df.empty:
                candidate_df = region_df

        candidate_df = applyQualityFilter(candidate_df, minDescriptionLen=20)
        rows = sourceBalancedFaissSearch(
            candidate_df,
            self.embeddings,
            self.embedding_model,
            query,
            kPerSource=12,
            limit=limit,
            useOriginalIndex=self.use_original_index,
        )
        return region, rows

    def select_region(self, search_query):
        initial_results = faissSearchRows(
            self.quality_df,
            self.embeddings,
            self.embedding_model,
            search_query,
            k=120,
            useOriginalIndex=self.use_original_index,
        )

        region_counts = {}
        for row in initial_results:
            local_region = normalizeLocalRegion(row.get("address", ""))
            if local_region:
                region_counts[local_region] = region_counts.get(local_region, 0) + 1

        top_regions = sorted(region_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        if not top_regions:
            return "", initial_results

        region_prompt = f"""
Select exactly one Korean city/district from the candidates that best matches
the user's travel preferences. Print only the region name.

User preferences:
{search_query}

Candidate regions:
{top_regions}
"""
        response = self.model.generate_content(region_prompt)
        selected_text = clean_value(response.text)
        selected_region = detectLocalRegionFromData(selected_text, self.quality_df)

        if not selected_region:
            broad_region = detectRegion(selected_text) or normalizeRegion(selected_text)
            selected_region = broad_region if broad_region else top_regions[0][0]

        return selected_region, initial_results

    def generate(self, age, mood, purpose, interest):
        search_query = "\n".join(
            [
                f"age group: {age}",
                f"mood: {mood}",
                f"purpose: {purpose}",
                f"interest: {interest}",
            ]
        )

        selected_region, _ = self.select_region(search_query)
        region_df = filterByLocalRegion(self.quality_df, selected_region)
        if region_df.empty:
            region_df = self.quality_df

        rows = sourceBalancedFaissSearch(
            region_df,
            self.embeddings,
            self.embedding_model,
            search_query,
            kPerSource=10,
            limit=15,
            useOriginalIndex=self.use_original_index,
        )
        context = build_context(rows)

        allowed = ", ".join(ALLOWED_CATEGORIES)
        prompt = f"""
You are a Korean local culture story curator.

User input:
- age group: {age}
- mood: {mood}
- purpose: {purpose}
- interest: {interest}

Recommended region:
{selected_region or UNKNOWN_REGION}

Retrieved culture data:
{context}

Rules:
1. Use only facts from the retrieved culture data.
2. Do not invent places, events, dates, or historical facts.
3. Reflect the user's age group, mood, purpose, and interest naturally.
4. Write all JSON string values in Korean.
5. category must be exactly one of these values: {allowed}
6. Return only a raw JSON object. Do not include markdown or explanation.

JSON schema:
{{
  "title": "story title",
  "description": "story description",
  "category": "one allowed category",
  "region": "city or district",
  "timeline": [
    {{"year": "period", "title": "event title", "text": "event description"}},
    {{"year": "period", "title": "event title", "text": "event description"}}
  ]
}}
"""

        data = self.generate_json(prompt)
        return validate_story(data, rows, selected_region)

    def recommend(self, query):
        region, rows = self.search_rows(query)
        context = build_context(rows)
        places = rows_to_places(rows, limit=10)

        prompt = f"""
You are a Korean local culture recommender.

User query:
{query}

Detected region:
{region or UNKNOWN_REGION}

Retrieved culture data:
{context}

Rules:
1. Use only facts from the retrieved data.
2. Do not invent places, events, dates, or historical facts.
3. Write all JSON string values in Korean.
4. Return only a raw JSON object.

JSON schema:
{{
  "region": "detected region or null",
  "summary": "short recommendation summary",
  "recommendations": [
    {{
      "name": "culture content name",
      "category": "category",
      "reason": "why it is recommended",
      "visitPoint": "specific visitor point"
    }}
  ]
}}
"""
        data = self.generate_json(prompt)
        recommendations = data.get("recommendations")
        if not isinstance(recommendations, list):
            recommendations = []

        return {
            "region": clean_value(data.get("region")) or region or "",
            "summary": clean_value(data.get("summary")),
            "recommendations": recommendations,
            "items": places,
        }

    def travel(self, age, mood, purpose, interest):
        search_query = "\n".join(
            [
                f"age group: {age}",
                f"mood: {mood}",
                f"purpose: {purpose}",
                f"interest: {interest}",
            ]
        )
        selected_region, _ = self.select_region(search_query)
        region_df = filterByLocalRegion(self.quality_df, selected_region)
        if region_df.empty:
            region_df = self.quality_df

        rows = sourceBalancedFaissSearch(
            region_df,
            self.embeddings,
            self.embedding_model,
            search_query,
            kPerSource=10,
            limit=15,
            useOriginalIndex=self.use_original_index,
        )
        context = build_context(rows)
        places = rows_to_places(rows, limit=10)

        prompt = f"""
You are a Korean local culture travel curator.

User input:
- age group: {age}
- mood: {mood}
- purpose: {purpose}
- interest: {interest}

Recommended region:
{selected_region or UNKNOWN_REGION}

Retrieved culture data:
{context}

Rules:
1. Build a one-day local culture travel course using only retrieved data.
2. Do not invent places, events, dates, or historical facts.
3. Write all JSON string values in Korean.
4. Return only a raw JSON object.

JSON schema:
{{
  "region": "city or district",
  "course": {{
    "morning": {{"title": "stop title", "place": "place name", "text": "description"}},
    "lunch": {{"title": "stop title", "place": "place name", "text": "description"}},
    "afternoon": {{"title": "stop title", "place": "place name", "text": "description"}},
    "evening": {{"title": "stop title", "place": "place name", "text": "description"}}
  }},
  "reason": "why this course fits",
  "preferenceReflection": "how the user preferences are reflected"
}}
"""
        data = self.generate_json(prompt)
        return {
            "region": clean_value(data.get("region")) or selected_region or "",
            "course": data.get("course") if isinstance(data.get("course"), dict) else {},
            "reason": clean_value(data.get("reason")),
            "preferenceReflection": clean_value(data.get("preferenceReflection")),
            "items": places,
        }

    def trend(self):
        source_count = self.df["source"].fillna("unknown").astype(str).value_counts().to_dict()
        category_count = {}
        if "category" in self.df.columns:
            category_count = (
                self.df["category"].fillna("unknown").astype(str).value_counts().head(15).to_dict()
            )

        region_counter = Counter()
        if "address" in self.df.columns:
            for address in self.df["address"].fillna("").astype(str):
                region = normalizeRegion(address)
                if region:
                    region_counter[region] += 1

        region_count = dict(region_counter.most_common(15))
        keyword_count = extract_culture_keywords(self.df)

        prompt = f"""
You are a Korean culture data analyst.

Actual data statistics:
sourceCount: {source_count}
categoryCount: {category_count}
regionCount: {region_count}
cultureKeywordCount: {keyword_count}

Rules:
1. Analyze only the provided statistics.
2. Do not invent numbers.
3. Write all JSON string values in Korean.
4. Return only a raw JSON object.

JSON schema:
{{
  "characteristics": "data characteristics",
  "regionInsights": "main regional insights",
  "keywordInsights": "keyword insights",
  "trendInterpretation": "culture trend interpretation",
  "usageIdeas": ["idea 1", "idea 2", "idea 3"]
}}
"""
        analysis = self.generate_json(prompt)
        return {
            "sourceCount": source_count,
            "categoryCount": category_count,
            "regionCount": region_count,
            "cultureKeywordCount": keyword_count,
            "analysis": analysis,
        }

    def discover(self, region):
        region = clean_value(region)
        region_df = self.df[
            self.df["address"].fillna("").astype(str).str.contains(region, regex=False, na=False)
        ]
        if region_df.empty:
            return {
                "region": region,
                "totalCount": 0,
                "assets": [],
                "analysis": {},
            }

        quality_df = applyQualityFilter(region_df, minDescriptionLen=20)
        rows = rows_to_places(
            [row for _, row in quality_df.head(50).iterrows()],
            limit=20,
        )
        context = build_context([row for _, row in quality_df.head(30).iterrows()])

        prompt = f"""
You are a Korean local culture researcher.

Region:
{region}

Retrieved culture data:
{context}

Rules:
1. Use only facts from the retrieved data.
2. Do not invent places, events, dates, or historical facts.
3. Write all JSON string values in Korean.
4. Return only a raw JSON object.

JSON schema:
{{
  "representativeAssets": ["asset 1", "asset 2", "asset 3"],
  "culturalIdentity": "regional cultural identity",
  "differentiators": "what makes this region different",
  "tourismPotential": "tourism potential",
  "contentIdeas": ["idea 1", "idea 2", "idea 3"]
}}
"""
        analysis = self.generate_json(prompt)
        return {
            "region": region,
            "totalCount": int(len(region_df)),
            "assets": rows,
            "analysis": analysis,
        }


@lru_cache(maxsize=1)
def get_ai_service():
    return CuratorAIService()


def get_story_generator():
    return get_ai_service()

import os

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import get_data_path, get_embedding_path
    from src.rag.utils import (
        apply_quality_filter,
        balanced_by_source,
        build_context,
        detect_region,
        detect_region_from_data,
        faiss_search_rows,
        filter_by_region,
    )
except ModuleNotFoundError:
    from paths import get_data_path, get_embedding_path
    from utils import (
        apply_quality_filter,
        balanced_by_source,
        build_context,
        detect_region,
        detect_region_from_data,
        faiss_search_rows,
        filter_by_region,
    )


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

df = pd.read_csv(get_data_path(), encoding="utf-8-sig")
embeddings = np.load(get_embedding_path())

print("\n===== AI 지역 문화 추천 =====")

query = input("\n질문을 입력하세요: ").strip()
region = detect_region_from_data(query, df) or detect_region(query)

candidate_df = df
if region:
    region_df = filter_by_region(df, region)
    if not region_df.empty:
        candidate_df = region_df

candidate_df = apply_quality_filter(candidate_df, min_description_len=20)

search_results = faiss_search_rows(
    candidate_df,
    embeddings,
    embedding_model,
    query,
    k=60,
)
balanced_results = balanced_by_source(search_results, limit=15)

print("\n===== 추천 검색 결과 =====\n")
if region:
    print(f"감지된 지역: {region}\n")

for row in balanced_results:
    print(f"[{row.get('source', '')}] {row.get('name', '')}")

print("\n=========================\n")

context = build_context(balanced_results)

prompt = f"""
당신은 한국 지역 문화 전문 큐레이터입니다.

사용자 질문:
{query}

감지된 지역:
{region or "없음"}

검색된 문화 데이터:
{context}

목표

사용자가 궁금해하는 지역 또는 문화적 관심사에 맞춰 방문 가치가 있는 문화 콘텐츠를 추천하세요.

규칙

1. 반드시 검색 결과에 있는 데이터만 근거로 작성하세요.
2. 검색 결과에 없는 장소, 행사, 역사적 사실은 만들지 마세요.
3. 지역명이 감지된 경우 해당 지역의 문화적 특징을 먼저 설명하세요.
4. 문화재, 향토문화, 행사, 공연, 축제, 전통시장, 특화거리를 균형 있게 엮어 설명하세요.
5. 단순 목록이 아니라 왜 이 콘텐츠들이 함께 추천되는지 문화적 맥락을 설명하세요.
6. 관광객 관점에서 방문 가치와 체험 포인트를 구체적으로 제안하세요.
7. 설명이 빈약한 항목보다 문화적 맥락이 분명한 항목을 우선 소개하세요.

출력 형식

[문화 특징]

[추천 문화 콘텐츠]

[추천 이유]

[방문 포인트]
"""

print("\n답변 생성 중...\n")

response = model.generate_content(prompt)
print(response.text)

print("\n===== 추천 Context =====\n")
print(context)

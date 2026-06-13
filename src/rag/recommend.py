# AI 지역 문화 추천

import os
import faiss
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ==========================================
# 환경변수
# ==========================================

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================================
# 데이터 로드
# ==========================================

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

# ==========================================
# FAISS
# ==========================================

index = faiss.read_index(
    "data/processed/data.index"
)

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

# ==========================================
# 사용자 입력
# ==========================================

print("\n===== AI 지역 문화 추천 =====")

query = input(
    "\n질문을 입력하세요 : "
)

# ==========================================
# 임베딩
# ==========================================

query_embedding = embedding_model.encode(
    [query],
    convert_to_numpy=True
)

# ==========================================
# 검색
# ==========================================

distances, indices = index.search(
    query_embedding,
    k=20
)

# ==========================================
# 중복 제거
# ==========================================

search_results = []

seen_names = set()

for idx in indices[0]:

    row = df.iloc[idx]

    name = str(
        row.get("name", "")
    )

    if name in seen_names:
        continue

    seen_names.add(name)

    search_results.append(row)

# ==========================================
# Source 균형화
# ==========================================

heritage = []
festival = []
market = []
street = []

for row in search_results:

    source = row.get(
        "source",
        ""
    )

    if (
        source == "문화재"
        and len(heritage) < 3
    ):
        heritage.append(row)

    elif (
        source == "축제"
        and len(festival) < 3
    ):
        festival.append(row)

    elif (
        source == "전통시장"
        and len(market) < 2
    ):
        market.append(row)

    elif (
        source == "특화거리"
        and len(street) < 2
    ):
        street.append(row)

balanced_results = (
    heritage
    + festival
    + market
    + street
)

# ==========================================
# 부족 시 추가
# ==========================================

if len(balanced_results) < 10:

    existing_names = {
        row.get("name", "")
        for row in balanced_results
    }

    for row in search_results:

        if row.get(
            "name",
            ""
        ) in existing_names:
            continue

        balanced_results.append(row)

        if len(balanced_results) >= 10:
            break

# ==========================================
# Context 생성
# ==========================================

context = ""

for row in balanced_results:

    context += f"""
이름: {row.get('name', '')}
유형: {row.get('source', '')}
분류: {row.get('category', '')}
주소: {row.get('address', '')}
설명: {row.get('description', '')}

----------------------------------------
"""

# ==========================================
# Prompt
# ==========================================

prompt = f"""
당신은 한국 문화 전문 큐레이터입니다.

사용자 질문:
{query}

검색된 문화 데이터:
{context}

목표

사용자가 궁금해하는 지역 또는 문화의 특징을 설명하고,
대표 문화 콘텐츠를 추천하세요.

규칙

1. 단순 장소 나열 금지
2. 문화적 특징을 먼저 설명
3. 문화재, 축제, 전통시장, 특화거리를 종합적으로 고려
4. 검색 결과 기반으로 작성
5. 검색 결과에 없는 내용은 생성하지 마세요
6. 관광객 관점에서 설명하세요

출력 형식

[문화 특징]

[대표 문화 콘텐츠]

문화재:
축제:
전통시장:
특화거리:

[추천 이유]

[방문 포인트]
"""

# ==========================================
# Gemini
# ==========================================

print("\n답변 생성 중...\n")

response = model.generate_content(
    prompt
)

print(
    response.text
)

print("\n===== 추천 Context =====\n")
print(context)
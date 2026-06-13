# AI 지역 스토리 생성

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

print("\n===== AI 지역 스토리 생성 =====")

query = input(
    "\n스토리를 생성할 지역 또는 문화 자산을 입력하세요 : "
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
# 부족하면 추가
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
당신은 지역 문화 전문 스토리텔러입니다.

사용자 요청:
{query}

검색된 문화 데이터:
{context}

목표

검색된 문화 자산을 활용하여
흥미로운 지역 문화 스토리를 작성하세요.

규칙

1. 단순 정보 나열 금지
2. 이야기 형식으로 작성
3. 역사적 의미 포함
4. 지역의 문화적 가치 포함
5. 현대적 의미도 함께 설명
6. 검색 결과에 없는 사실은 생성하지 마세요
7. 관광객이 방문하고 싶어지도록 작성하세요
8. 자연스럽고 몰입감 있게 작성하세요
9. 800~1200자 내외로 작성하세요

출력 형식

[문화 스토리]

(스토리 작성)

[문화적 의미]

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
# AI 맞춤형 로컬 여행 추천

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

print("\n===== AI 맞춤형 로컬 여행 추천 =====")

age = input(
    "\n연령대를 입력하세요 : "
)

interest = input(
    "관심사를 입력하세요 : "
)

mood = input(
    "선호 분위기를 입력하세요 : "
)

purpose = input(
    "여행 목적을 입력하세요 : "
)

search_query = (
    f"{interest} {purpose}"
)

# ==========================================
# 임베딩
# ==========================================

query_embedding = embedding_model.encode(
    [search_query],
    convert_to_numpy=True
)

# ==========================================
# 1차 검색
# ==========================================

distances, indices = index.search(
    query_embedding,
    k=30
)

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
# 지역 집계
# ==========================================

region_count = {}

for row in search_results:

    address = str(
        row.get("address", "")
    )

    if (
        address == ""
        or address == "nan"
    ):
        continue

    region = address.split()[0]

    region_count[region] = (
        region_count.get(region, 0)
        + 1
    )

# ==========================================
# 추천 지역 선정
# ==========================================

selected_region = None

if region_count:

    selected_region = max(
        region_count,
        key=region_count.get
    )

print(
    f"\n추천 지역 : {selected_region}"
)

# ==========================================
# 해당 지역 데이터만 추출
# ==========================================

region_df = df[
    df["address"]
    .fillna("")
    .str.contains(
        selected_region,
        na=False
    )
]

# ==========================================
# 유형별 분리
# ==========================================

heritage = region_df[
    region_df["source"] == "문화재"
].head(3)

festival = region_df[
    region_df["source"] == "축제"
].head(2)

market = region_df[
    region_df["source"] == "전통시장"
].head(2)

street = region_df[
    region_df["source"] == "특화거리"
].head(2)

travel_data = pd.concat(
    [
        heritage,
        festival,
        market,
        street
    ]
)

# ==========================================
# Context 생성
# ==========================================

context = ""

for _, row in travel_data.iterrows():

    context += f"""
이름: {row.get('name', '')}
유형: {row.get('source', '')}
분류: {row.get('category', '')}
주소: {row.get('address', '')}
기간: {row.get('period', '')}
설명: {row.get('description', '')}

----------------------------------------
"""

# ==========================================
# Prompt
# ==========================================

prompt = f"""
당신은 지역 문화 전문 여행 큐레이터입니다.

사용자 정보

연령대: {age}
관심사: {interest}
선호 분위기: {mood}
여행 목적: {purpose}

추천 지역:
{selected_region}

지역 문화 데이터:
{context}

목표

사용자의 취향에 맞는
하루 여행 코스를 설계하세요.

규칙

1. 추천 지역 내 데이터만 사용하세요.
2. 검색 결과에 없는 장소는 생성하지 마세요.
3. 문화재, 축제, 전통시장, 특화거리를 조합하세요.
4. 동일 장소를 반복 사용하지 마세요.
5. 실제 여행 일정처럼 작성하세요.
6. 가족 여행 여부를 고려하세요.
7. 조용한 분위기 여부를 고려하세요.

출력 형식

[추천 지역]

[추천 여행 코스]

오전:
점심:
오후:
저녁:

[추천 이유]

[사용자 취향 반영 내용]
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

print("\n===== 여행 추천 Context =====\n")
print(context)
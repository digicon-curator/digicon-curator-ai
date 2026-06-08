# RAG 검색 + Gemini 생성

import os
import faiss
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 환경변수 로드
load_dotenv()

# Gemini 설정
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# 데이터 로드
df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

# FAISS 인덱스 로드
index = faiss.read_index(
    "data/processed/data.index"
)

# 임베딩 모델
embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

print("\n===== Curator AI =====")
print("1. 문화 추천")
print("2. 문화 스토리 생성")
print("3. AI 맞춤형 로컬 여행 추천")

mode = input("\n메뉴를 선택하세요 : ")

# ==========================================
# 메뉴별 입력
# ==========================================

if mode == "3":

    age = input("\n연령대를 입력하세요 : ")

    interest = input(
        "관심사를 입력하세요 : "
    )

    mood = input(
        "선호 분위기를 입력하세요 : "
    )

    purpose = input(
        "여행 목적을 입력하세요 : "
    )

    # 검색용 쿼리
    search_query = (
        f"{interest} {mood} {purpose}"
    )

    # Gemini용 사용자 정보
    query = f"""
연령대: {age}
관심사: {interest}
선호 분위기: {mood}
여행 목적: {purpose}
"""

else:

    query = input(
        "\n질문을 입력하세요 : "
    )

    search_query = query

# ==========================================
# 질문 임베딩
# ==========================================

query_embedding = embedding_model.encode(
    [search_query],
    convert_to_numpy=True
)

# ==========================================
# 유사 문서 검색
# ==========================================

distances, indices = index.search(
    query_embedding,
    k=5
)

# ==========================================
# Context 생성
# ==========================================

context = ""

for idx in indices[0]:

    row = df.iloc[idx]

    context += f"""
이름: {row.get('name', '')}
유형: {row.get('source', '')}
분류: {row.get('category', '')}
주소: {row.get('address', '')}
기간: {row.get('period', '')}
설명: {row.get('description', '')}
관련 품목: {row.get('items', '')}

----------------------------------------
"""

# ==========================================
# 문화 추천
# ==========================================

if mode == "1":

    prompt = f"""
당신은 한국 문화관광 전문 큐레이터입니다.

사용자 질문:
{query}

검색된 문화 데이터:
{context}

규칙

1. 사용자에게 가장 적합한 문화 콘텐츠를 추천하세요.
2. 추천 이유를 설명하세요.
3. 검색 결과에 없는 내용은 생성하지 마세요.

출력 형식

[추천 장소]

[추천 이유]

[추천 포인트]
"""

# ==========================================
# 문화 스토리 생성
# ==========================================

elif mode == "2":

    prompt = f"""
당신은 문화 해설사입니다.

사용자 질문:
{query}

검색된 문화 데이터:
{context}

규칙

1. 검색된 정보를 기반으로 문화 스토리를 작성하세요.
2. 단순 정보 나열이 아닌 이야기 형식으로 작성하세요.
3. 문화적 의미와 역사적 배경을 포함하세요.
4. 사용자가 흥미를 느낄 수 있도록 작성하세요.
5. 검색 결과에 없는 내용은 과도하게 추측하지 마세요.
"""

# ==========================================
# AI 맞춤형 로컬 여행 추천
# ==========================================

elif mode == "3":

    prompt = f"""
당신은 지역 문화 전문 여행 큐레이터입니다.

사용자 정보

{query}

검색된 문화 데이터:
{context}

규칙

1. 사용자의 관심사를 최우선으로 고려하세요.
2. 선호 분위기를 고려하세요.
3. 여행 목적을 고려하세요.
4. 문화재, 축제, 전통시장, 특화거리를 적절히 조합하세요.
5. 검색 결과에 기반하여 추천하세요.
6. 검색 결과에 없는 장소는 생성하지 마세요.

출력 형식

[추천 여행 코스]

오전:
점심:
오후:
저녁:

[추천 이유]

[사용자 취향 반영 내용]
"""

else:

    print("잘못된 메뉴입니다.")
    exit()

# ==========================================
# Gemini 호출
# ==========================================

print("\n답변 생성 중...\n")

response = model.generate_content(
    prompt
)

print(response.text)
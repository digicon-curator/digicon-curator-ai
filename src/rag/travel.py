import os

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import getDataPath, getEmbeddingPath
    from src.rag.utils import (
        applyQualityFilter,
        balancedBySource,
        buildContext,
        detectRegion,
        faissSearchRows,
        filterByRegion,
        normalizeRegion,
    )
except ModuleNotFoundError:
    from paths import getDataPath, getEmbeddingPath
    from utils import (
        applyQualityFilter,
        balancedBySource,
        buildContext,
        detectRegion,
        faissSearchRows,
        filterByRegion,
        normalizeRegion,
    )


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
embeddingModel = SentenceTransformer("intfloat/multilingual-e5-base")

df = pd.read_csv(getDataPath(), encoding="utf-8-sig")
embeddings = np.load(getEmbeddingPath())

print("\n===== AI 맞춤형 로컬 여행 추천 =====")

age = input("\n연령대를 입력하세요: ").strip()
interest = input("관심사를 입력하세요: ").strip()
mood = input("선호 분위기를 입력하세요: ").strip()
purpose = input("여행 목적을 입력하세요: ").strip()

searchQuery = f"""
관심사:{interest}
분위기:{mood}
목적:{purpose}
"""

qualityDf = applyQualityFilter(df, minDescriptionLen=20)
initialResults = faissSearchRows(
    qualityDf,
    embeddings,
    embeddingModel,
    searchQuery,
    k=80,
)

regionCount = {}
for row in initialResults:
    region = normalizeRegion(row.get("address", ""))
    if not region:
        continue
    regionCount[region] = regionCount.get(region, 0) + 1

topRegions = sorted(regionCount.items(), key=lambda item: item[1], reverse=True)[:5]

regionPrompt = f"""
사용자 정보

연령대: {age}
관심사: {interest}
선호 분위기: {mood}
여행 목적: {purpose}

추천 후보 지역:
{topRegions}

위 후보 중 사용자에게 가장 적합한 지역 1곳만 고르세요.
반드시 지역명만 출력하세요.
"""

selectedRegionText = model.generate_content(regionPrompt).text.strip()
selectedRegion = detectRegion(selectedRegionText) or normalizeRegion(selectedRegionText)

print(f"\n추천 지역: {selectedRegion}")

regionDf = filterByRegion(qualityDf, selectedRegion)
if regionDf.empty:
    regionDf = qualityDf

regionResults = faissSearchRows(
    regionDf,
    embeddings,
    embeddingModel,
    searchQuery,
    k=60,
)
travelData = balancedBySource(regionResults, limit=15)

print("\n===== 여행 추천 검색 결과 =====\n")
for row in travelData:
    print(f"[{row.get('source', '')}] {row.get('name', '')}")

print("\n=========================\n")

context = buildContext(travelData)

prompt = f"""
당신은 지역 문화 전문 여행 큐레이터입니다.

사용자 정보

연령대: {age}
관심사: {interest}
선호 분위기: {mood}
여행 목적: {purpose}

추천 지역:
{selectedRegion}

지역 문화 데이터:
{context}

목표

사용자의 취향에 맞는 하루 로컬 문화 여행 코스를 설계하세요.

규칙

1. 추천 지역의 데이터만 중심으로 사용하세요.
2. 검색 결과에 없는 장소나 행사는 만들지 마세요.
3. 문화재, 향토문화, 행사, 공연, 축제, 전통시장, 특화거리를 가능한 한 균형 있게 활용하세요.
4. 같은 장소를 반복하지 마세요.
5. 실제 여행 일정처럼 오전, 점심, 오후, 저녁 흐름으로 작성하세요.
6. 연령대, 관심사, 분위기, 목적을 모두 반영하세요.
7. 최소 3개 이상의 문화 자산을 사용하세요.
8. 각 코스마다 왜 그 시간대에 적합한지 간단히 설명하세요.

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

print("\n답변 생성 중...\n")

response = model.generate_content(prompt)
print(response.text)

print("\n===== 여행 추천 Context =====\n")
print(context)

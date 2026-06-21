import os

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import getDataPath, getEmbeddingPath, originalEmbeddingPath
    from src.rag.utils import (
        applyQualityFilter,
        buildContext,
        detectRegion,
        detectRegionFromData,
        filterByRegion,
        sourceBalancedFaissSearch,
    )
except ModuleNotFoundError:
    from paths import getDataPath, getEmbeddingPath, originalEmbeddingPath
    from utils import (
        applyQualityFilter,
        buildContext,
        detectRegion,
        detectRegionFromData,
        filterByRegion,
        sourceBalancedFaissSearch,
    )


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
embeddingModel = SentenceTransformer("intfloat/multilingual-e5-base")

embeddingPath = getEmbeddingPath()
df = pd.read_csv(getDataPath(), encoding="utf-8-sig")
embeddings = np.load(embeddingPath)
useOriginalIndex = embeddingPath == originalEmbeddingPath and "originalIndex" in df.columns

print("\n===== AI 지역 문화 스토리 생성 =====")

query = input("\n스토리를 생성할 지역 또는 문화 자산을 입력하세요: ").strip()
region = detectRegionFromData(query, df) or detectRegion(query)

candidateDf = df
if region:
    regionDf = filterByRegion(df, region)
    if not regionDf.empty:
        candidateDf = regionDf

candidateDf = applyQualityFilter(candidateDf, minDescriptionLen=20)

storyResults = sourceBalancedFaissSearch(
    candidateDf,
    embeddings,
    embeddingModel,
    query,
    kPerSource=12,
    limit=15,
    useOriginalIndex=useOriginalIndex,
)

print("\n===== 스토리 검색 결과 =====\n")
if region:
    print(f"감지된 지역: {region}\n")

for row in storyResults:
    print(f"[{row.get('source', '')}] {row.get('name', '')}")

print("\n=========================\n")

context = buildContext(storyResults)

prompt = f"""
당신은 지역 문화 전문 스토리텔러입니다.

사용자 요청:
{query}

감지된 지역:
{region or "없음"}

검색된 문화 데이터:
{context}

목표

검색된 문화 자산만 사용해 지역의 역사, 생활문화, 향토문화, 축제와 공연, 시장과 거리의 분위기가 자연스럽게 이어지는 문화 스토리를 작성하세요.

강조할 관점

1. 문화재는 지역의 역사와 기억을 보여주는 축으로 다루세요.
2. 향토문화는 지역민의 생활 방식, 전승, 공동체성을 보여주는 요소로 다루세요.
3. 행사, 공연, 축제는 현재 지역 문화가 살아 움직이는 장면으로 연결하세요.
4. 전통시장과 특화거리는 관광객이 지역성을 직접 체감하는 공간으로 설명하세요.
5. 감지된 지역이 있으면 해당 지역 데이터만 중심으로 서사를 구성하세요.

규칙

1. 검색 결과에 없는 문화 자산, 지명, 역사적 사실은 만들지 마세요.
2. 설명이 20자 이하인 빈약한 데이터는 이미 제외되었으므로, 제공된 데이터의 구체 설명을 적극 활용하세요.
3. 단순 정보 나열을 피하고 방문자가 현장에서 이동하며 느끼는 흐름으로 작성하세요.
4. 문화적 의미와 관광 포인트를 함께 설명하세요.
5. 1000~1500자 내외로 작성하세요.

출력 형식

[문화 스토리]

[지역성과 문화적 의미]

[방문 포인트]
"""

print("\n답변 생성 중...\n")

response = model.generate_content(prompt)
print(response.text)

print("\n===== 스토리 Context =====\n")
print(context)

# AI 지역 문화 자산 발굴

import os
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv

try:
    from src.rag.paths import getDataPath
except ModuleNotFoundError:
    from paths import getDataPath

# ==========================================
# 환경변수 로드
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
    getDataPath(),
    encoding="utf-8-sig"
)

# ==========================================
# 지역 입력
# ==========================================

region = input(
    "\n분석할 지역을 입력하세요 : "
)

# ==========================================
# 지역 데이터 조회
# ==========================================

regionDf = df[
    df["address"]
    .fillna("")
    .str.contains(region, na=False)
]

if len(regionDf) == 0:

    print(
        "\n해당 지역의 데이터가 없습니다."
    )

    exit()

print(
    f"\n검색된 데이터 수 : {len(regionDf)}"
)

# ==========================================
# Source 균형 추출
# ==========================================

maxPerSource = 10

heritageDf = regionDf[
    regionDf["source"] == "문화재"
].head(maxPerSource)

festivalDf = regionDf[
    regionDf["source"] == "축제"
].head(maxPerSource)

marketDf = regionDf[
    regionDf["source"] == "전통시장"
].head(maxPerSource)

streetDf = regionDf[
    regionDf["source"] == "특화거리"
].head(maxPerSource)

eventDf = regionDf[
    regionDf["source"] == "행사"
].head(maxPerSource)

localCultureDf = regionDf[
    regionDf["source"] == "향토문화"
].head(maxPerSource)

performanceDf = regionDf[
    regionDf["source"] == "공연"
].head(maxPerSource)

balancedDf = pd.concat(
    [
        heritageDf,
        festivalDf,
        marketDf,
        streetDf,
        eventDf,
        localCultureDf,
        performanceDf
    ],
    ignore_index=True
)

print(
    f"분석 대상 데이터 수 : {len(balancedDf)}"
)

# ==========================================
# Context 생성
# ==========================================

context = ""

for _, row in balancedDf.iterrows():

    context += f"""
유형: {row.get('source', '')}
이름: {row.get('name', '')}
분류: {row.get('category', '')}
주소: {row.get('address', '')}
기간: {row.get('period', '')}
설명: {row.get('description', '')}
부가정보: {row.get('items', '')}

----------------------------------------
"""

# ==========================================
# Gemini Prompt
# ==========================================

prompt = f"""
당신은 지역 문화 연구원입니다.

지역:
{region}

문화 데이터:
{context}

목표

단순히 문화재나 축제 목록을 나열하지 말고,
지역을 대표하는 문화 정체성과 문화 자산을 분석하세요.

분석 항목

1. 대표 문화 자산
2. 지역 문화 정체성
3. 다른 지역과 차별화되는 특징
4. 관광 활용 가능성
5. 발전 가능성이 높은 문화 콘텐츠

규칙

- 제공된 데이터만 기반으로 분석하세요.
- 없는 사실은 추측하지 마세요.
- 문화재, 축제, 전통시장, 특화거리를 종합적으로 고려하세요.
- 단순 나열보다 의미를 분석하세요.
- 보고서 형식으로 작성하세요.

출력 형식

[대표 문화 자산]

[지역 문화 정체성]

[차별화 요소]

[관광 활용 가능성]

[발전 가능성이 높은 콘텐츠]
"""

# ==========================================
# Gemini 호출
# ==========================================

print(
    "\n문화 자산 분석 중...\n"
)

response = model.generate_content(
    prompt
)

print(
    response.text
)

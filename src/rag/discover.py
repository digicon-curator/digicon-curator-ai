# AI 지역 문화 자산 발굴

import os
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv

try:
    from src.rag.paths import get_data_path
except ModuleNotFoundError:
    from paths import get_data_path

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
    get_data_path(),
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

region_df = df[
    df["address"]
    .fillna("")
    .str.contains(region, na=False)
]

if len(region_df) == 0:

    print(
        "\n해당 지역의 데이터가 없습니다."
    )

    exit()

print(
    f"\n검색된 데이터 수 : {len(region_df)}"
)

# ==========================================
# Source 균형 추출
# ==========================================

MAX_PER_SOURCE = 10

heritage_df = region_df[
    region_df["source"] == "문화재"
].head(MAX_PER_SOURCE)

festival_df = region_df[
    region_df["source"] == "축제"
].head(MAX_PER_SOURCE)

market_df = region_df[
    region_df["source"] == "전통시장"
].head(MAX_PER_SOURCE)

street_df = region_df[
    region_df["source"] == "특화거리"
].head(MAX_PER_SOURCE)

event_df = region_df[
    region_df["source"] == "행사"
].head(MAX_PER_SOURCE)

localCulture_df = region_df[
    region_df["source"] == "향토문화"
].head(MAX_PER_SOURCE)

performance_df = region_df[
    region_df["source"] == "공연"
].head(MAX_PER_SOURCE)

balanced_df = pd.concat(
    [
        heritage_df,
        festival_df,
        market_df,
        street_df,
        event_df,
        localCulture_df,
        performance_df
    ],
    ignore_index=True
)

print(
    f"분석 대상 데이터 수 : {len(balanced_df)}"
)

# ==========================================
# Context 생성
# ==========================================

context = ""

for _, row in balanced_df.iterrows():

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

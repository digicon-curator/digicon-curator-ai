# AI 지역 문화 자산 발굴

import os
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv

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
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 지역 입력
# ==========================================

region = input(
    "\n분석할 지역을 입력하세요 : "
)

# ==========================================
# 지역 데이터 추출
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
# Context 생성
# ==========================================

context = ""

for _, row in region_df.head(50).iterrows():

    context += f"""
이름: {row.get('name', '')}
유형: {row.get('source', '')}
분류: {row.get('category', '')}
주소: {row.get('address', '')}
설명: {row.get('description', '')}

----------------------------------------
"""

# ==========================================
# Gemini Prompt
# ==========================================

prompt = f"""
당신은 지역 문화 연구원입니다.

다음은 {region} 지역의 문화 데이터입니다.

{context}

아래 내용을 분석하세요.

1. 이 지역을 대표하는 문화 자산은 무엇인가?
2. 이 지역의 문화적 특징은 무엇인가?
3. 관광 자원으로 활용 가능한 핵심 콘텐츠는 무엇인가?
4. 다른 지역과 차별화되는 문화적 강점은 무엇인가?
5. 향후 발전 가능성이 높은 문화 콘텐츠는 무엇인가?

규칙

- 제공된 데이터만 기반으로 분석하세요.
- 없는 사실은 추측하지 마세요.
- 보고서 형식으로 작성하세요.

출력 형식

[지역 문화 자산 발굴 결과]

[대표 문화 자산]

[문화적 특징]

[관광 활용 가능성]

[잠재력 있는 문화 콘텐츠]
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
import os
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv

# 환경변수
load_dotenv()

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

# ====================================
# 데이터 분석
# ====================================

source_count = (
    df["source"]
    .value_counts()
    .to_dict()
)

category_count = {}

if "category" in df.columns:

    category_count = (
        df["category"]
        .fillna("기타")
        .value_counts()
        .head(10)
        .to_dict()
    )

# 주소가 있는 데이터만 사용
address_data = (
    df["address"]
    .fillna("")
)

region_count = {}

for address in address_data:

    if address == "":
        continue

    region = address.split()[0]

    region_count[region] = (
        region_count.get(region, 0) + 1
    )

region_count = dict(
    sorted(
        region_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
)

# ====================================
# Gemini Prompt
# ====================================

prompt = f"""
당신은 문화 데이터 분석가입니다.

다음 통계를 분석하세요.

[문화 유형 개수]
{source_count}

[상위 카테고리]
{category_count}

[상위 지역]
{region_count}

다음 형식으로 작성하세요.

1. 문화 데이터 특징
2. 주요 지역 특징
3. 문화 트렌드 분석
4. 활용 방안

전문가 보고서 형식으로 작성하세요.
"""

response = model.generate_content(
    prompt
)

print("\n===== 문화 트렌드 분석 =====\n")

print(response.text)
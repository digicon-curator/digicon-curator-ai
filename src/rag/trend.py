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
당신은 문화 데이터 분석 전문가입니다.

다음은 실제 문화 데이터 통계입니다.

[문화 유형 개수]
{source_count}

[상위 카테고리]
{category_count}

[상위 지역]
{region_count}

분석 목표

1. 단순 수치 나열이 아니라 데이터의 의미를 분석하세요.
2. 제공된 통계만 활용하여 분석하세요.
3. 없는 수치나 사실은 생성하지 마세요.
4. 작성일, 작성자, 기관명 등 제공되지 않은 정보는 생성하지 마세요.
5. 객관적인 데이터 분석 보고서 형식으로 작성하세요.

출력 형식

## 문화 데이터 분석 보고서

### 1. 문화 데이터 특징

- 문화 유형 분포 특징
- 카테고리 분포 특징
- 데이터 품질 관점 특징

### 2. 주요 지역 특징

- 지역별 문화 자산 분포
- 문화 자원이 집중된 지역 특징

### 3. 문화 트렌드 분석

- 현재 문화 데이터가 보여주는 문화 트렌드
- 향후 발전 가능성

### 4. 활용 방안

- 관광 활용 방안
- 지역 활성화 방안
- 데이터 활용 전략

규칙

- 제공된 데이터 범위 안에서만 분석하세요.
- 추측성 표현은 최소화하세요.
- 문화재, 축제, 전통시장, 특화거리를 함께 고려하세요.
- 보고서 형식으로 작성하세요.
"""

response = model.generate_content(
    prompt
)

print("\n===== 문화 트렌드 분석 =====\n")

print(response.text)
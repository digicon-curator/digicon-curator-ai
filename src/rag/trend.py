import os
import re
from collections import Counter

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

try:
    from src.rag.paths import getDataPath
    from src.rag.utils import cleanValue, normalizeRegion
except ModuleNotFoundError:
    from paths import getDataPath
    from utils import cleanValue, normalizeRegion


stopwords = {
    "문화",
    "지역",
    "행사",
    "축제",
    "공연",
    "시장",
    "전통",
    "관광",
    "체험",
    "운영",
    "개최",
    "관련",
    "제공",
    "위한",
    "통해",
    "있는",
    "없는",
    "및",
    "등",
}


def splitKeywords(value):
    value = cleanValue(value)
    if not value:
        return []

    tokens = re.split(r"[,/|·ㆍ;:\[\]\(\)\s]+", value)
    return [token.strip() for token in tokens if len(token.strip()) >= 2]


def extractCultureKeywords(df):
    counter = Counter()

    columns = [col for col in ["category", "name", "description", "items"] if col in df.columns]
    for col in columns:
        for value in df[col].dropna().astype(str):
            for token in splitKeywords(value):
                if token in stopwords:
                    continue
                if token.isdigit():
                    continue
                counter[token] += 1

    return dict(counter.most_common(30))


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

df = pd.read_csv(getDataPath(), encoding="utf-8-sig")

sourceCount = df["source"].fillna("기타").value_counts().to_dict()

categoryCount = {}
if "category" in df.columns:
    categoryCount = (
        df["category"]
        .fillna("기타")
        .astype(str)
        .value_counts()
        .head(15)
        .to_dict()
    )

regionCounter = Counter()
if "address" in df.columns:
    for address in df["address"].fillna("").astype(str):
        region = normalizeRegion(address)
        if region:
            regionCounter[region] += 1

regionCount = dict(regionCounter.most_common(15))
cultureKeywordCount = extractCultureKeywords(df)

prompt = f"""
당신은 문화 데이터 분석 전문가입니다.

다음은 실제 문화 데이터 통계입니다.

[문화 유형 개수]
{sourceCount}

[상위 분류]
{categoryCount}

[상위 지역]
{regionCount}

[문화 키워드]
{cultureKeywordCount}

분석 목표

1. 단순 수치 나열이 아니라 데이터가 보여주는 문화적 특징과 편중을 분석하세요.
2. 주소는 정규화된 광역 지역 기준으로 집계되었습니다.
3. 문화 키워드를 중심으로 현재 데이터가 어떤 문화 주제를 많이 담는지 설명하세요.
4. 제공된 통계에 없는 수치나 사실은 만들지 마세요.
5. 문화재, 향토문화, 행사, 공연, 축제, 전통시장, 특화거리를 함께 고려하세요.

출력 형식

## 문화 데이터 분석 보고서

### 1. 문화 데이터의 특징

### 2. 주요 지역 특징

### 3. 문화 키워드 분석

### 4. 문화 트렌드 해석

### 5. 활용 방안
"""

print("\n===== 문화 트렌드 분석 =====\n")

response = model.generate_content(prompt)
print(response.text)

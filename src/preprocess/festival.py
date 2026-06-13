import pandas as pd

df = pd.read_csv(
    "data/raw/festival.csv",
    encoding="utf-8-sig"
)

columns = [
    "title",
    "description",
    "spatialCoverage",
    "subjectKeyword",
    "eventPeriod"
]

df = df[columns]

df = df.rename(
    columns={
        "title": "name",
        "description": "description",
        "spatialCoverage": "address",
        "eventPeriod": "period",
        "subjectKeyword": "category"
    }
)

# ==========================================
# 주소 자동 보정
# ==========================================

region_map = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",

    "수원": "경기도 수원시",
    "성남": "경기도 성남시",
    "용인": "경기도 용인시",
    "고양": "경기도 고양시",
    "부천": "경기도 부천시",
    "안산": "경기도 안산시",
    "안양": "경기도 안양시",
    "평택": "경기도 평택시",

    "강릉": "강원특별자치도 강릉시",
    "속초": "강원특별자치도 속초시",
    "춘천": "강원특별자치도 춘천시",
    "원주": "강원특별자치도 원주시",

    "청주": "충청북도 청주시",
    "충주": "충청북도 충주시",

    "천안": "충청남도 천안시",
    "아산": "충청남도 아산시",
    "서산": "충청남도 서산시",

    "전주": "전북특별자치도 전주시",
    "군산": "전북특별자치도 군산시",
    "익산": "전북특별자치도 익산시",
    "남원": "전북특별자치도 남원시",

    "목포": "전라남도 목포시",
    "순천": "전라남도 순천시",
    "여수": "전라남도 여수시",
    "광양": "전라남도 광양시",

    "경주": "경상북도 경주시",
    "포항": "경상북도 포항시",
    "안동": "경상북도 안동시",
    "구미": "경상북도 구미시",

    "창원": "경상남도 창원시",
    "진주": "경상남도 진주시",
    "통영": "경상남도 통영시",
    "거제": "경상남도 거제시",

    "제주": "제주특별자치도 제주시"
}

filled_count = 0

for idx, row in df.iterrows():

    address = row["address"]

    if pd.notna(address):
        continue

    name = str(row["name"])

    for keyword, region in region_map.items():

        if keyword in name:

            df.at[idx, "address"] = region
            filled_count += 1
            break

print(f"제목 기반 주소 보정: {filled_count}건")

# ==========================================
# 저장
# ==========================================

df.to_csv(
    "data/processed/festivalProcessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("festivalProcessed.csv 저장 완료")
print(df.shape)
print(df.head())
import os
import math
import requests
import pandas as pd
import xmltodict

from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("HERITAGE_KEY")

API_URL = "https://api.kcisa.kr/openapi/service/rest/meta/CHAheri"

num_rows = 1000

# 첫 페이지 호출
response = requests.get(
    API_URL,
    params={
        "serviceKey": SERVICE_KEY,
        "pageNo": 1,
        "numOfRows": num_rows
    }
)

xml_data = xmltodict.parse(response.text)

total_count = int(
    xml_data["response"]["body"]["totalCount"]
)

total_pages = math.ceil(
    total_count / num_rows
)

print(f"전체 데이터 수 : {total_count}")
print(f"전체 페이지 수 : {total_pages}")

all_items = []

for page in range(1, total_pages + 1):

    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": page,
        "numOfRows": num_rows
    }

    response = requests.get(
        API_URL,
        params=params
    )

    xml_data = xmltodict.parse(response.text)

    items = xml_data["response"]["body"]["items"]["item"]

    if isinstance(items, list):
        all_items.extend(items)
    else:
        all_items.append(items)

    print(f"{page}/{total_pages} 수집 완료")

df = pd.DataFrame(all_items)

df.to_csv(
    "data/raw/heritage.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"최종 저장 완료 : {len(df)}건")
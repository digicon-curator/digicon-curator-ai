import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

index = faiss.read_index(
    "data/processed/data.index"
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

query = "역사적인 여행지"

query_embedding = model.encode(
    [query]
)

distances, indices = index.search(
    query_embedding,
    k=5
)

print("\n검색 결과\n")

for idx in indices[0]:

    row = df.iloc[idx]

    print("이름:", row["name"])

    print(
        "주소:",
        row["address"]
        if pd.notna(row["address"])
        else "주소 정보 없음"
    )

    print(
        "설명:",
        str(row["description"])[:150]
    )

    print("-" * 50)
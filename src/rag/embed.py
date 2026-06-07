import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

df = pd.read_csv(
    "data/processed/Data.csv",
    encoding="utf-8-sig"
)

contents = df["content"].fillna("").tolist()

print(f"데이터 수 : {len(contents)}")

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

print("모델 로드 완료")

all_embeddings = []

for i in range(0, len(contents), 1000):

    batch = contents[i:i + 1000]

    embeddings = model.encode(
        batch,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    all_embeddings.append(
        embeddings
    )

    print(
        f"{min(i + 1000, len(contents))} / {len(contents)} 완료"
    )

embeddings = np.vstack(
    all_embeddings
)

print("최종 Shape :", embeddings.shape)

np.save(
    "data/processed/embeddings.npy",
    embeddings
)

print("embeddings.npy 저장 완료")
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import CURATED_DATA_PATH, CURATED_EMBEDDING_PATH, ORIGINAL_EMBEDDING_PATH, get_data_path
except ModuleNotFoundError:
    from paths import CURATED_DATA_PATH, CURATED_EMBEDDING_PATH, ORIGINAL_EMBEDDING_PATH, get_data_path


data_path = get_data_path()
embedding_path = CURATED_EMBEDDING_PATH if data_path == CURATED_DATA_PATH else ORIGINAL_EMBEDDING_PATH

df = pd.read_csv(data_path, encoding="utf-8-sig")
contents = df["content"].fillna("").tolist()

print(f"데이터 경로: {data_path}")
print(f"데이터 수: {len(contents)}")

model = SentenceTransformer("intfloat/multilingual-e5-base")
print("모델 로드 완료")

all_embeddings = []

for i in range(0, len(contents), 1000):
    batch = contents[i : i + 1000]

    embeddings = model.encode(
        batch,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    all_embeddings.append(embeddings)
    print(f"{min(i + 1000, len(contents))} / {len(contents)} 완료")

embeddings = np.vstack(all_embeddings)

print("최종 Shape:", embeddings.shape)

np.save(embedding_path, embeddings)

print(f"{embedding_path} 저장 완료")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import curatedDataPath, curatedEmbeddingPath, originalEmbeddingPath, getDataPath
except ModuleNotFoundError:
    from paths import curatedDataPath, curatedEmbeddingPath, originalEmbeddingPath, getDataPath


dataPath = getDataPath()
embeddingPath = curatedEmbeddingPath if dataPath == curatedDataPath else originalEmbeddingPath

df = pd.read_csv(dataPath, encoding="utf-8-sig")
contents = df["content"].fillna("").tolist()

print(f"데이터 경로: {dataPath}")
print(f"데이터 수: {len(contents)}")

model = SentenceTransformer("intfloat/multilingual-e5-base")
print("모델 로드 완료")

allEmbeddings = []

for i in range(0, len(contents), 1000):
    batch = contents[i : i + 1000]

    embeddings = model.encode(
        batch,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    allEmbeddings.append(embeddings)
    print(f"{min(i + 1000, len(contents))} / {len(contents)} 완료")

embeddings = np.vstack(allEmbeddings)

print("최종 Shape:", embeddings.shape)

np.save(embeddingPath, embeddings)

print(f"{embeddingPath} 저장 완료")

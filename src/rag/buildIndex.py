import faiss
import numpy as np

embeddings = np.load(
    "data/processed/embeddings.npy"
)

print("임베딩 로드 완료")
print("Shape :", embeddings.shape)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "data/processed/data.index"
)

print("FAISS 인덱스 저장 완료")
print("총 데이터 수 :", index.ntotal)
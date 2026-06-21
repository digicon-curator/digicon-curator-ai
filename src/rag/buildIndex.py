import faiss
import numpy as np

try:
    from src.rag.paths import CURATED_EMBEDDING_PATH, CURATED_INDEX_PATH, ORIGINAL_INDEX_PATH, get_embedding_path
except ModuleNotFoundError:
    from paths import CURATED_EMBEDDING_PATH, CURATED_INDEX_PATH, ORIGINAL_INDEX_PATH, get_embedding_path


embedding_path = get_embedding_path()
index_path = CURATED_INDEX_PATH if embedding_path == CURATED_EMBEDDING_PATH else ORIGINAL_INDEX_PATH

embeddings = np.load(embedding_path).astype("float32")

print(f"임베딩 로드 완료: {embedding_path}")
print("Shape:", embeddings.shape)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, index_path)

print(f"FAISS 인덱스 저장 완료: {index_path}")
print("총 데이터 수:", index.ntotal)

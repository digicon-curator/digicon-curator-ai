import faiss
import numpy as np

try:
    from src.rag.paths import curatedEmbeddingPath, curatedIndexPath, originalIndexPath, getEmbeddingPath
except ModuleNotFoundError:
    from paths import curatedEmbeddingPath, curatedIndexPath, originalIndexPath, getEmbeddingPath


embeddingPath = getEmbeddingPath()
indexPath = curatedIndexPath if embeddingPath == curatedEmbeddingPath else originalIndexPath

embeddings = np.load(embeddingPath).astype("float32")

print(f"임베딩 로드 완료: {embeddingPath}")
print("Shape:", embeddings.shape)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, indexPath)

print(f"FAISS 인덱스 저장 완료: {indexPath}")
print("총 데이터 수:", index.ntotal)

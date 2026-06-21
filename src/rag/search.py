import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import getDataPath, getEmbeddingPath, originalEmbeddingPath
    from src.rag.utils import applyQualityFilter, faissSearchRows
except ModuleNotFoundError:
    from paths import getDataPath, getEmbeddingPath, originalEmbeddingPath
    from utils import applyQualityFilter, faissSearchRows


embeddingPath = getEmbeddingPath()
df = pd.read_csv(getDataPath(), encoding="utf-8-sig")
embeddings = np.load(embeddingPath)
useOriginalIndex = embeddingPath == originalEmbeddingPath and "originalIndex" in df.columns
model = SentenceTransformer("intfloat/multilingual-e5-base")

query = "역사적인 여행지"
candidateDf = applyQualityFilter(df, minDescriptionLen=20)
results = faissSearchRows(
    candidateDf,
    embeddings,
    model,
    query,
    k=5,
    useOriginalIndex=useOriginalIndex,
)

print("\n검색 결과\n")

for row in results:
    print("이름:", row.get("name", ""))
    print("주소:", row.get("address", "주소 정보 없음"))
    print("설명:", str(row.get("description", ""))[:150])
    print("-" * 50)

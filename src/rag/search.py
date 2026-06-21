import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from src.rag.paths import get_data_path, get_embedding_path
    from src.rag.utils import apply_quality_filter, faiss_search_rows
except ModuleNotFoundError:
    from paths import get_data_path, get_embedding_path
    from utils import apply_quality_filter, faiss_search_rows


df = pd.read_csv(get_data_path(), encoding="utf-8-sig")
embeddings = np.load(get_embedding_path())
model = SentenceTransformer("intfloat/multilingual-e5-base")

query = "역사적인 여행지"
candidate_df = apply_quality_filter(df, min_description_len=20)
results = faiss_search_rows(candidate_df, embeddings, model, query, k=5)

print("\n검색 결과\n")

for row in results:
    print("이름:", row.get("name", ""))
    print("주소:", row.get("address", "주소 정보 없음"))
    print("설명:", str(row.get("description", ""))[:150])
    print("-" * 50)

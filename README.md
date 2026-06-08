# 가상환경 
- 생성
    python -m venv venv
- 활성화
   venv\Scripts\activate 
- 패키지 설치
    pip install -r requirements.txt
- 가상환경 비활성화
    deactivate

## 프로젝트 구조

```text
src
├─ api
│  ├─ festival.py
│  ├─ heritage.py
│  ├─ market.py
│  ├─ Street.py
│  └─ localCulture.py

├─ preprocess
│  ├─ merge.py
│  └─ content.py

├─ rag
│  ├─ embed.py
│  ├─ build_index.py
│  ├─ search.py
│  ├─ generate.py
│  ├─ discover.py
│  └─ trend.py
```
# 파일 설명
- API 수집
    - festival.py	지역 축제 데이터 수집
    - heritage.py	문화재 데이터 수집
    - market.py	전통시장 데이터 수집
    - specialStreet.py	특화거리 데이터 수집
    - localCulture.py	향토문화 데이터 수집
- 전처리
    - merge.py	수집 데이터 병합
    - content.py	RAG 학습용 content 생성
- RAG
    - embed.py	임베딩 생성
    - build_index.py	FAISS 인덱스 생성
    - search.py	벡터 검색 테스트
    - generate.py	문화 추천, 스토리 생성, 맞춤형 여행 추천
    - discover.py	지역 문화 자산 발굴
    - trend.py	문화 트렌드 분석

# 실행 순서
1. 데이터 병합
    python src/preprocess/merge.py
2. Content 생성
    python src/preprocess/content.py
3. 임베딩 생성
    python src/rag/embed.py
4. FAISS 인덱스 생성
    python src/rag/build_index.py
5. AI 기능 실행
    python src/rag/generate.py
    python src/rag/discover.py
    python src/rag/trend.py

# 데이터 
- 문화체육관광부_지역축제정보: API
- 한국학중앙연구원 외_한국의 향토 문화: API (미사용)
- 문화재청_문화재정보: API
- 전국전통시장표준데이터: CSV
- 전국지역특화거리표준데이터: CSV

# 사용 기술
- Python 
- Pandas
- FAISS
- Sentence Transformers
- Gemini 2.5 Flash
- RAG (Retrieval-Augmented Generation)

# Curator

지역 문화 데이터를 기반으로 문화 콘텐츠 추천, 지역 스토리 생성, 맞춤형 여행 코스 추천, 문화 트렌드 분석을 수행하는 RAG 기반 큐레이션 프로젝트입니다.

## 주요 기능

- 지역 문화 추천: 사용자 질문에서 지역명을 감지하고, 해당 지역 데이터를 우선 검색해 문화 콘텐츠를 추천합니다.
- 지역 문화 스토리: 문화재, 향토문화, 행사, 축제, 공연, 전통시장, 특화거리를 연결해 지역성 중심의 스토리를 생성합니다.
- 맞춤형 여행 추천: 관심사, 분위기, 목적을 바탕으로 Gemini가 시군구 단위 지역을 선정하고 하루 여행 코스를 제안합니다.
- 문화 트렌드 분석: source, category, 지역, 문화 키워드 분포를 분석합니다.
- 데이터 큐레이션: 원본 대용량 데이터를 품질 기준과 source/지역 균형 기준으로 축소합니다.

## 기술 스택

- Python
- Pandas
- NumPy
- FAISS
- Sentence Transformers
- Google Gemini 2.5 Flash
- RAG, Retrieval-Augmented Generation

## 프로젝트 구조

```text
Curator
|-- data
|   `-- processed
|       |-- Data.csv
|       |-- dataCurated.csv
|       |-- embeddings.npy
|       |-- embeddingsCurated.npy
|       |-- data.index
|       `-- dataCurated.index
|-- src
|   |-- preprocess
|   |   |-- festival.py
|   |   |-- heritage.py
|   |   |-- localCulture.py
|   |   |-- event.py
|   |   |-- performance.py
|   |   |-- market.py
|   |   |-- street.py
|   |   |-- merge.py
|   |   |-- curate.py
|   |   `-- content.py
|   |-- rag
|   |   |-- paths.py
|   |   |-- utils.py
|   |   |-- embed.py
|   |   |-- buildIndex.py
|   |   |-- search.py
|   |   |-- recommend.py
|   |   |-- story.py
|   |   |-- travel.py
|   |   |-- discover.py
|   |   `-- trend.py
|   `-- test
|       |-- checkData.py
|       `-- geminiTest.py
|-- requirements.txt
`-- README.md
```

## 환경 설정

가상환경을 생성하고 패키지를 설치합니다.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일을 만들고 Gemini API 키를 설정합니다.

```text
GEMINI_API_KEY=your_api_key_here
```

## 실행 순서

### 1. 원본 데이터 병합

각 전처리 파일로 생성된 source별 CSV를 하나의 `Data.csv`로 병합합니다.

```powershell
python src/preprocess/merge.py
```

생성 파일:

```text
data/processed/Data.csv
```

### 2. 데이터 축소 및 품질 정리

공모전 시연과 RAG 검색 품질을 위해 원본 데이터를 약 2만 개 규모로 축소합니다.

```powershell
python src/preprocess/curate.py
```

생성 파일:

```text
data/processed/dataCurated.csv
```

기본 기준:

- description 20자 이하 제거
- HTML 태그와 HTML 엔티티 제거
- 의미 없는 값 제거
- 축제, 행사, 공연은 2021년 이후 데이터 우선 유지
- `source + name + address` 기준 중복 제거
- source별 상한 적용
- source 내부에서 지역 균형 샘플링

기본 목표 개수는 22,000개이고, 이벤트성 데이터의 최소 연도는 2021년입니다. 필요하면 환경 변수로 바꿀 수 있습니다.

```powershell
$env:CURATOR_TARGET_TOTAL="30000"
$env:CURATOR_MIN_EVENT_YEAR="2021"
python src/preprocess/curate.py
```

### 3. RAG content 생성

검색 임베딩에 사용할 `content` 컬럼을 생성합니다.

```powershell
python src/preprocess/content.py
```

`dataCurated.csv`가 있으면 자동으로 축소 데이터를 우선 사용합니다. 없으면 `Data.csv`를 사용합니다.

`content.py`는 검색 품질을 높이기 위해 URL, 이메일, 전화번호, 홈페이지/문의 안내, 반복 문장부호, HTML 태그와 엔티티, 너무 짧은 설명을 제거합니다. 지역 정보는 광역 단위가 아니라 가능한 시군구 단위로 `content`에 포함합니다.

### 4. 임베딩 생성

```powershell
python src/rag/embed.py
```

생성 파일:

```text
data/processed/embeddingsCurated.npy
```

`dataCurated.csv`가 없으면 기존 원본 데이터 기준으로 `embeddings.npy`를 생성합니다.

### 5. FAISS 인덱스 생성

```powershell
python src/rag/buildIndex.py
```

생성 파일:

```text
data/processed/dataCurated.index
```

## RAG 기능 실행

지역 문화 추천:

```powershell
python src/rag/recommend.py
```

지역 문화 스토리 생성:

```powershell
python src/rag/story.py
```

맞춤형 로컬 여행 추천:

```powershell
python src/rag/travel.py
```

지역 문화 자산 발견:

```powershell
python src/rag/discover.py
```

문화 트렌드 분석:

```powershell
python src/rag/trend.py
```

검색 테스트:

```powershell
python src/rag/search.py
```

## 데이터 경로 정책

[src/rag/paths.py](src/rag/paths.py)가 데이터 경로를 자동 선택합니다.

우선순위:

1. `dataCurated.csv`가 있으면 축소 데이터를 사용합니다.
2. 없으면 `Data.csv`를 사용합니다.
3. `embeddingsCurated.npy`가 있으면 축소 임베딩을 사용합니다.
4. 없으면 `embeddings.npy`를 사용합니다.

환경 변수로 직접 경로를 지정할 수도 있습니다.

```powershell
$env:CURATOR_DATA_PATH="data/processed/Data.csv"
$env:CURATOR_EMBEDDING_PATH="data/processed/embeddings.npy"
$env:CURATOR_INDEX_PATH="data/processed/data.index"
```

## 주요 파일 설명

- `src/preprocess/merge.py`: source별 전처리 데이터를 `Data.csv`로 병합합니다.
- `src/preprocess/curate.py`: 대용량 원본 데이터를 품질, 최신성, 균형 기준으로 축소합니다.
- `src/preprocess/content.py`: RAG 검색용 `content` 컬럼을 생성하고 노이즈 텍스트를 제거합니다.
- `src/rag/paths.py`: 축소 데이터와 원본 데이터 경로를 자동 선택합니다.
- `src/rag/utils.py`: 지역 감지, 시군구 정규화, 품질 필터, 지역 필터, FAISS 검색, source 균형화를 담당합니다.
- `src/rag/embed.py`: SentenceTransformer 임베딩을 생성합니다.
- `src/rag/buildIndex.py`: FAISS 인덱스를 생성합니다.
- `src/rag/recommend.py`: 지역 문화 추천을 실행합니다.
- `src/rag/story.py`: 지역 문화 스토리를 생성합니다.
- `src/rag/travel.py`: 사용자 취향 기반 시군구 단위 여행 코스를 추천합니다.
- `src/rag/discover.py`: 특정 지역의 문화 자산을 요약하고 발견합니다.
- `src/rag/trend.py`: 문화 데이터 분포와 키워드를 분석합니다.

## 권장 실행 흐름

공모전 제출 또는 시연 기준으로는 아래 순서를 권장합니다.

```powershell
python src/preprocess/merge.py
python src/preprocess/curate.py
python src/preprocess/content.py
python src/rag/embed.py
python src/rag/buildIndex.py
python src/rag/recommend.py
```

이미 `Data.csv`가 있고 축소 데이터만 다시 만들고 싶다면 아래부터 실행하면 됩니다.

```powershell
python src/preprocess/curate.py
python src/preprocess/content.py
python src/rag/embed.py
python src/rag/buildIndex.py
```

## 주의사항

- `dataCurated.csv`를 새로 만들면 임베딩과 FAISS 인덱스도 다시 생성하는 것을 권장합니다.
- 축제, 행사, 공연은 시간성이 강하므로 기본적으로 2021년 이전 데이터가 추천용 데이터에서 제외됩니다.
- 원본 데이터와 축소 데이터의 행 수가 다르면, `originalIndex` 컬럼을 통해 기존 원본 임베딩을 임시로 참조할 수 있습니다.
- 최종 시연 전에는 `embeddingsCurated.npy`와 `dataCurated.index`를 새로 생성해 데이터와 인덱스를 맞추는 것이 가장 안정적입니다.
- Gemini 기능을 실행하려면 `.env`의 `GEMINI_API_KEY`가 반드시 필요합니다.

## AI Server API

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

All endpoints return JSON.

- `POST /generate`: backend-compatible story generation
- `POST /stories`: story generation
- `POST /recommend`: culture recommendation
- `POST /travel`: travel course recommendation
- `GET /trend`: culture trend analysis
- `POST /discover`: regional culture discovery

`/api/...` routes belong to the backend server contract. The AI server does not need the `/api` prefix unless the backend team decides to call it with that prefix.

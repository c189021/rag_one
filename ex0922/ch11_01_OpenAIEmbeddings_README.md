# ch11_01_OpenAIEmbeddings.ipynb

OpenAI의 임베딩 모델을 사용해 텍스트를 벡터로 변환하고, 벡터 간 유사도를 비교하며, 임베딩 결과를 캐싱하는 방법을 다루는 노트북입니다.

## 주요 개념

### 1. 임베딩(Embeddings)이란
텍스트를 고정된 길이의 실수 벡터로 변환하는 것입니다. 의미가 비슷한 문장일수록 벡터 공간에서 서로 가까운 위치에 놓이며, 이를 통해 문장 간 의미적 유사도를 계산할 수 있습니다.

### 2. OpenAIEmbeddings 기본 사용법
- `langchain_openai.OpenAIEmbeddings`로 임베딩 모델(`text-embedding-3-small`)을 생성합니다.
- `embed_query(text)`: 단일 텍스트(쿼리)를 임베딩 벡터로 변환합니다.
- `embed_documents([text, ...])`: 여러 문서를 한 번에 임베딩하여 벡터 리스트를 반환합니다.
- `text-embedding-3-small` 모델의 기본 임베딩 차원은 **1536**입니다.

### 3. 임베딩 차원 축소 (dimensions 옵션)
`OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)`처럼 `dimensions` 파라미터를 지정하면 임베딩 벡터의 차원 수를 줄일 수 있습니다(예: 1536 → 1024). 저장 공간과 연산량을 줄이면서도 어느 정도 의미 정보를 유지하는 트레이드오프입니다.

### 4. 코사인 유사도(Cosine Similarity)를 이용한 문장 비교
- `sklearn.metrics.pairwise.cosine_similarity`를 사용해 두 임베딩 벡터 간 유사도를 계산합니다.
- 실습 결과 요약:
  - 표현만 다른 유사 문장(구두점 차이 등)은 유사도가 매우 높음 (0.96 수준)
  - 같은 언어로 의미가 비슷하지만 표현이 다른 문장은 중간 수준 유사도 (0.82~0.84)
  - 언어가 다른 문장(한국어 vs 영어) 간에는 유사도가 낮아짐 (0.48~0.52)
  - 의미적으로 전혀 관련 없는 문장은 유사도가 가장 낮음 (0.13~0.23)
- 즉, 임베딩 벡터의 코사인 유사도는 문장 간 의미적 유사성을 잘 반영합니다.

### 5. 임베딩 캐싱 (CacheBackedEmbeddings)
매번 API를 호출해 임베딩을 새로 계산하면 비용과 시간이 낭비되므로, 계산된 임베딩 결과를 캐시에 저장해 재사용합니다.

- **`LocalFileStore`**: 임베딩 결과를 로컬 디스크(`./cache/` 폴더)에 영구적으로 저장합니다. 프로그램을 재시작해도 캐시가 유지됩니다.
- **`InMemoryByteStore`**: 임베딩 결과를 메모리에만 저장합니다(비영구적). 프로세스가 종료되면 캐시가 사라집니다.
- `CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, document_embedding_cache, namespace)`로 캐시 지원 임베딩 객체를 생성합니다.
  - `namespace`는 임베딩 모델별로 캐시를 구분하는 키 역할을 합니다.
- 캐시 효과 실증: 문서를 로드해 `FAISS.from_documents()`로 벡터 스토어를 처음 만들 때보다, 동일한 임베딩을 캐시에서 재사용해 두 번째로 만들 때가 훨씬 빠릅니다 (예제에서 42.2ms → 12.4ms).

### 6. 문서 로드 및 분할과의 연계
- `TextLoader`로 텍스트 파일(`./data/appendix-keywords.txt`)을 로드합니다.
- `CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)`로 문서를 청크 단위로 분할합니다.
- 분할된 문서들을 캐시 지원 임베딩과 함께 `FAISS` 벡터 스토어에 저장하여, 이후 유사도 검색(RAG의 핵심 구성 요소)에 활용할 수 있습니다.

## 전체 흐름 요약
1. 텍스트를 OpenAI 임베딩 모델로 벡터화
2. 벡터 차원 확인 및 축소 옵션 실습
3. 코사인 유사도로 문장 간 의미적 유사성 비교
4. 임베딩 결과를 로컬 파일 또는 메모리에 캐싱하여 재계산 비용 절감
5. 캐시된 임베딩을 FAISS 벡터 스토어와 결합하여 문서 검색 준비

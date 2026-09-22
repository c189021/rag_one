# ch12_02_PineconeVectorStore.ipynb

Pinecone 벡터 데이터베이스를 이용해 임베딩 벡터를 저장(upsert)하고, 질문과 의미적으로 유사한 문서를 검색(query)하는 전체 과정을 실습하는 노트북입니다.

## 주요 개념

### 1. Pinecone이란
- **Pinecone**은 대규모 벡터 데이터를 저장하고 빠르게 유사도 검색을 수행할 수 있는 완전관리형(managed) 벡터 데이터베이스 서비스입니다.
- RAG(Retrieval-Augmented Generation) 시스템에서 문서 임베딩을 저장해두고, 사용자 질문과 유사한 문서를 빠르게 찾아내는 검색 계층으로 널리 사용됩니다.

### 2. Pinecone 클라이언트 초기화 및 인덱스 생성
```python
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index(
    name="quickstart-index",
    dimension=1536,       # OpenAI text-embedding-3-small 기준
    metric="cosine",      # 유사도 측정 방식
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```
- **인덱스(Index)**는 Pinecone에서 벡터들을 저장하는 단위(테이블과 유사한 개념)입니다.
- `dimension`은 저장할 임베딩 벡터의 차원 수와 반드시 일치해야 합니다(여기서는 OpenAI `text-embedding-3-small`의 1536차원).
- `metric="cosine"`은 벡터 간 유사도를 코사인 유사도 방식으로 계산하겠다는 설정입니다.
- `ServerlessSpec`은 서버리스 방식(사용한 만큼 과금, 인프라 관리 불필요)으로 인덱스를 클라우드(AWS us-east-1)에 배포하는 설정입니다.
- 이미 동일한 이름의 인덱스가 존재하면 재생성하지 않도록 `pc.list_indexes()`로 중복 체크를 합니다.

### 3. 문서 임베딩 후 Upsert(저장)
- OpenAI API(`openai_client.embeddings.create`)로 문서 텍스트를 임베딩 벡터로 변환합니다.
- 변환된 벡터를 Pinecone이 요구하는 형식(`id`, `values`, `metadata`)으로 구성합니다.
  - `id`: 벡터를 식별하는 고유 키
  - `values`: 실제 임베딩 벡터 값
  - `metadata`: 원본 텍스트 등 부가 정보(검색 결과 확인용)
- `index.upsert(vectors=vectors)`로 Pinecone 인덱스에 벡터를 저장합니다.
  - **Upsert = Update + Insert**: 동일한 `id`로 다시 upsert하면 기존 데이터를 덮어쓰며, 별도의 오류 없이 갱신됩니다.
- `time.sleep(2)`는 Pinecone 서버에 데이터가 색인(indexing)될 시간을 확보하기 위한 대기입니다.

### 4. 질문 임베딩 및 유사도 검색 (Query)
- 사용자 질문("임베딩 데이터를 저장하고 유사한 정보를 찾는 서비스는 무엇인가요?")도 동일한 임베딩 모델로 벡터화합니다.
- `index.query(vector=question_vector, top_k=2, include_metadata=True)`로 질문 벡터와 가장 유사한 상위 2개 문서를 검색합니다.
  - `top_k`: 반환할 결과 개수
  - `include_metadata=True`: 검색 결과에 저장해둔 메타데이터(원본 텍스트)를 포함
- 검색 결과는 `match.id`, `match.metadata`, `match.score`(유사도 점수)로 구성됩니다.

### 5. 실습 결과
질문과 의미가 더 가까운 문서(`doc2`: "임베딩 벡터를 저장하고 유사한 정보를 검색하는 서비스")가 유사도 0.6642로 1위, 상대적으로 포괄적인 설명(`doc1`: "벡터 데이터베이스입니다")이 0.4797로 2위를 차지했습니다. 질문의 표현과 더 유사한 문서가 실제로 더 높은 점수를 받는 것을 확인할 수 있습니다.

## 전체 흐름 요약
1. Pinecone API 키로 클라이언트 초기화
2. 임베딩 차원(1536)에 맞는 서버리스 인덱스 생성
3. OpenAI 임베딩으로 문서를 벡터화한 후 Pinecone에 upsert(저장)
4. 사용자 질문을 동일한 방식으로 임베딩
5. `index.query()`로 질문과 가장 유사한 문서를 검색하고 유사도 점수 확인

## 참고 사항
- 이 노트북은 `ch11` 시리즈의 임베딩 개념(텍스트 → 벡터 변환, 유사도 비교)을 실제 벡터 데이터베이스(Pinecone)와 결합해, RAG 파이프라인의 저장·검색 단계를 완성하는 실습입니다.
- 인덱스 생성/upsert 과정을 시각적으로 보여주는 참고 이미지(`Pinecone_인덱스생성.png`, `Pinecone_인덱스upsert.png`)가 원본 노트북 주석에 언급되어 있습니다.

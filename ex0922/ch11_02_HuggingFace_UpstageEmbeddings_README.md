# ch11_02_HuggingFace_UpstageEmbeddings.ipynb

HuggingFace의 임베딩 엔드포인트와 국내 스타트업 Upstage의 임베딩 모델을 사용해 텍스트를 벡터화하고, 쿼리와 문서 간 유사도를 비교하는 노트북입니다.

## 주요 개념

### 1. HuggingFaceEndpointEmbeddings
- `langchain_huggingface.embeddings.HuggingFaceEndpointEmbeddings`를 사용해 HuggingFace Hub의 Inference API를 통해 임베딩 모델(`intfloat/multilingual-e5-large-instruct`, 다국어 지원)을 호출합니다.
- `huggingfacehub_api_token`으로 인증하며, `task="feature-extraction"`으로 임베딩 추출 작업을 지정합니다.
- `os.environ["HF_HOME"]`으로 모델 다운로드/캐시 경로를 지정할 수 있습니다.
- **실행 결과**: 이 노트북에서는 HuggingFace의 월간 무료 크레딧이 소진되어 `402 Payment Required` 에러가 발생했습니다. 즉, HuggingFace Inference Providers는 무료 사용량에 한도가 있으며, 초과 시 유료 크레딧 구매나 PRO 구독이 필요합니다. (개념 이해 및 코드 구조 학습에는 문제가 없지만, 실제 API 호출은 실패한 예제입니다.)

### 2. UpstageEmbeddings
- Upstage는 LLM 및 문서 AI(Document AI) 분야에 특화된 국내 스타트업입니다.
- `langchain_upstage.UpstageEmbeddings`를 사용하며, **쿼리용 모델과 문서용 모델을 별도로 분리**하는 것이 특징입니다.
  - `solar-embedding-1-large-query`: 검색 질의(쿼리) 임베딩 전용
  - `solar-embedding-1-large-passage`: 검색 대상 문서(passage) 임베딩 전용
  - 이렇게 역할을 분리하면 검색(retrieval) 상황에서 쿼리와 문서 각각에 최적화된 벡터 표현을 얻을 수 있습니다(비대칭 임베딩 구조).
- Upstage 임베딩의 벡터 차원은 **4096**으로, OpenAI(1536)보다 훨씬 큽니다.

### 3. 쿼리-문서 유사도 계산 및 랭킹
- `embed_query()`로 질문을 임베딩하고, `embed_documents()`로 여러 문서를 임베딩합니다.
- `numpy`의 행렬 곱(`@`, 내적)을 이용해 쿼리 벡터와 문서 벡터들 간 유사도를 한 번에 계산합니다.
  - `similarity = query_vector @ document_vectors.T`
- `argsort()[::-1]`로 유사도를 내림차순 정렬하여, 질문과 가장 관련성 높은 문서 순으로 랭킹을 매깁니다.
- 실습 결과: "LangChain에 대해서 알려주세요"라는 질문에 대해 LangChain을 직접 설명하는 문장들이 가장 높은 유사도(0.43~0.48)를 보였고, 관련 없는 인사말("안녕, 만나서 반가워")이나 다른 주제(RAG 설명)는 낮은 유사도(0.15~0.18)를 보였습니다.

## 전체 흐름 요약
1. HuggingFace Inference API 기반 다국어 임베딩 모델 사용을 시도 (크레딧 소진으로 실패)
2. Upstage의 쿼리 전용/문서 전용 임베딩 모델을 각각 생성
3. 질문과 여러 후보 문서를 임베딩
4. 내적(dot product) 기반 유사도 계산 및 정렬로 질문과 가장 관련 있는 문서를 찾는 검색(retrieval) 과정을 실습

## 참고 사항
- HuggingFace Inference API는 무료 크레딧 소진 시 유료 결제가 필요하므로, 실습 환경에서는 로컬 모델 실행(`HuggingFaceEmbeddings`)이나 다른 임베딩 제공자로 대체할 수 있습니다.
- Upstage처럼 쿼리/문서용 임베딩 모델을 분리하는 방식은 비대칭 검색(asymmetric retrieval)에 특화된 설계로, RAG 시스템 성능 향상에 활용됩니다.

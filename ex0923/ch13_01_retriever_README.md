# ch13_01_retriever.ipynb

벡터스토어(FAISS) 기반 리트리버(Retriever)의 다양한 검색 방식과, LLM/임베딩을 이용한 문서 압축(Contextual Compression) 기법을 다루는 노트북입니다.

## 주요 개념

### 1. Retriever란
벡터스토어에 저장된 문서 중에서 질의(query)와 관련성이 높은 문서를 찾아 반환하는 컴포넌트입니다. `vectorstore.as_retriever()`로 벡터스토어를 리트리버 객체로 변환하여 사용합니다.

### 2. 기본 검색 (Similarity Search)
- `TextLoader`로 문서를 로드하고 `CharacterTextSplitter(chunk_size=300, chunk_overlap=0)`로 분할한 뒤, `OpenAIEmbeddings`로 임베딩하여 `FAISS` 벡터스토어를 생성합니다.
- `retriever.invoke(query)`로 기본 유사도 검색을 수행합니다.

### 3. 다양한 검색 유형(search_type)
- **`similarity`(기본값)**: 질의와 가장 유사한 문서를 단순 유사도 순으로 반환합니다.
- **`mmr` (Maximal Marginal Relevance)**: `search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6}`처럼 설정하며, 유사도가 높으면서도 서로 중복되지 않는(다양성 있는) 문서를 선택합니다. `fetch_k`개 후보 중에서 `k`개를 최종 선택하고, `lambda_mult`로 관련성과 다양성 간 균형을 조절합니다.
- **`similarity_score_threshold`**: `search_kwargs={"score_threshold": 0.8}`처럼 특정 유사도 점수 이상인 문서만 반환합니다.
- **`k` 값 조정**: `search_kwargs={"k": 1}`처럼 반환할 문서 개수를 직접 지정할 수 있습니다.

### 4. ConfigurableField로 런타임 설정 변경
- `retriever.configurable_fields(search_type=ConfigurableField(...), search_kwargs=ConfigurableField(...))`를 사용하면, 리트리버를 미리 만들어두고 **실행 시점(invoke 호출 시)에 config 매개변수로 검색 옵션(k 값 등)을 동적으로 변경**할 수 있습니다.
- 예: `retriever.invoke(query, config={"configurable": {"search_kwargs": {"k": 3}}})`

### 5. Upstage 쿼리/문서 분리 임베딩과의 연동
- 문서는 `solar-embedding-1-large-passage` 모델로, 질의는 `solar-embedding-1-large-query` 모델로 각각 임베딩한 뒤, `db.similarity_search_by_vector(query_vector, k=2)`로 이미 계산된 벡터를 직접 넘겨 검색할 수 있습니다.

### 6. 문서 압축기 (Contextual Compression Retriever)
검색된 문서 전체를 그대로 LLM에 전달하면 불필요한 내용까지 포함되어 비효율적이므로, 검색 결과를 질의에 맞게 압축/필터링하는 기법입니다.
- **`LLMChainExtractor`**: LLM(`gpt-4o-mini`)을 사용해 검색된 각 문서에서 질의와 관련된 내용만 추출(요약)합니다. 압축 전에는 문서 전체가 반환되지만, 압축 후에는 질의와 직접 관련된 핵심 문장만 남습니다.
- **`LLMChainFilter`**: 문서를 요약/수정하지 않고, LLM이 판단하여 질의와 관련 없는 문서 자체를 필터링(제거)합니다.
- **`EmbeddingsFilter`**: LLM 호출 없이 임베딩 유사도(`similarity_threshold=0.86`)만으로 관련 없는 문서를 빠르게 걸러냅니다. LLM 방식보다 비용이 저렴하고 속도가 빠릅니다.
- 이 세 가지 압축기는 모두 `ContextualCompressionRetriever(base_compressor=..., base_retriever=...)`에 결합하여 "검색 + 압축"을 하나의 파이프라인으로 사용합니다.

## 전체 흐름 요약
1. 문서를 로드·분할·임베딩하여 FAISS 벡터스토어 생성
2. `as_retriever()`로 다양한 검색 전략(similarity, MMR, score threshold, k 조정) 실습
3. `ConfigurableField`로 런타임에 검색 파라미터를 동적으로 변경하는 방법 학습
4. Upstage의 쿼리/문서 분리 임베딩을 벡터 검색에 직접 활용
5. LLM 기반(`LLMChainExtractor`, `LLMChainFilter`) 및 임베딩 기반(`EmbeddingsFilter`) 문서 압축으로 검색 결과의 품질과 효율을 개선

## 참고 사항
- `langchain-community`는 유지보수 종료(sunset) 예정이라는 경고가 있으며, 문서 압축기 관련 클래스는 `langchain_classic` 패키지로 이전되어 사용되고 있습니다(1.x 버전 호환성 이슈로 `langchain_teddynote` 대신 `langchain_classic` 사용).

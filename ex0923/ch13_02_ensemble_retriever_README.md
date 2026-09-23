# ch13_02_ensemble_retriever.ipynb

키워드 기반 검색(BM25)과 의미 기반 검색(FAISS 벡터 검색)을 결합한 앙상블 리트리버(EnsembleRetriever)를 다루는 노트북입니다.

## 주요 개념

### 1. EnsembleRetriever란
서로 다른 방식으로 동작하는 여러 리트리버의 검색 결과를 가중치를 두어 결합(앙상블)하는 리트리버입니다. 한 가지 검색 방식만으로는 놓칠 수 있는 결과를 보완하기 위해 사용합니다.

### 2. BM25Retriever (키워드 기반 검색)
- `BM25Retriever.from_texts(doc_list)`로 생성하며, TF-IDF 계열의 통계적 알고리즘(BM25)을 사용해 **키워드 매칭 기반**으로 문서를 검색합니다.
- 질의에 등장한 단어와 동일하거나 유사한 단어가 문서에 많이 포함될수록 높은 점수를 받습니다. (의미보다는 어휘적 일치에 강함)
- `bm25_retriever.k = 1`로 반환할 문서 개수를 설정합니다.

### 3. FAISS Retriever (의미 기반 검색)
- `OpenAIEmbeddings`로 문서를 벡터화하고 `FAISS.from_texts()`로 벡터스토어를 생성합니다.
- 벡터 간 유사도로 검색하므로, 질의와 문서에 동일한 단어가 없어도 **의미적으로 유사**하면 검색될 수 있습니다.

### 4. 두 리트리버의 결합
```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3],
)
```
- `weights`로 각 리트리버의 검색 결과에 부여할 가중치를 지정합니다(합이 1일 필요는 없으나 보통 비율로 사용).
- 실습 결과: `"my favorite fruit is apple"` 질의에 대해
  - BM25는 "apple"이라는 단어가 정확히 포함된 "Apple is my favorite company"를 1위로 선택
  - FAISS는 의미적으로 사과(과일)와 관련된 "I like apples"를 선택
  - 앙상블 결과는 두 방식의 결과를 모두 반영하여 두 문서를 함께 반환
- 즉, 키워드 검색과 의미 검색의 장점을 모두 살릴 수 있습니다.

### 5. 런타임 Config로 가중치 동적 변경
- `EnsembleRetriever(...).configurable_fields(weights=ConfigurableField(id="ensemble_weights", ...))`로 리트리버를 구성해두면, 검색 시점에 가중치를 바꿀 수 있습니다.
- `config = {"configurable": {"ensemble_weights": [1, 0]}}` → BM25(키워드) 결과만 우선 반영
- `config = {"configurable": {"ensemble_weights": [0, 1]}}` → FAISS(의미) 결과만 우선 반영
- 실습에서 가중치를 `[1, 0]`으로 설정했을 때는 "Apple is my favorite company"가 1순위로, `[0, 1]`로 설정했을 때는 "I like apples"가 1순위로 나와, 가중치 조정이 실제로 결과 순서에 영향을 준다는 것을 확인할 수 있습니다.

## 전체 흐름 요약
1. 동일한 문서 목록에 대해 BM25 리트리버(키워드 기반)와 FAISS 리트리버(임베딩 기반)를 각각 생성
2. `EnsembleRetriever`로 두 리트리버를 가중치 기반으로 결합
3. 각 리트리버(BM25/FAISS/앙상블)의 검색 결과를 비교하여 방식별 차이를 확인
4. `ConfigurableField`를 이용해 런타임에 가중치를 동적으로 조정하는 방법 학습

## 참고 사항
- 키워드 검색(BM25)은 정확한 용어 일치에 강하고, 벡터 검색(FAISS)은 의미적 유사성 파악에 강하므로, 두 방식을 앙상블하면 RAG 시스템의 검색 품질(재현율/정확도)을 함께 개선할 수 있습니다.

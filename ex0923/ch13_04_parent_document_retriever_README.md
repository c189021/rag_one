# ch13_04_parent_document_retriever.ipynb

문서를 작은 조각(청크)으로 나눠 검색 정확도를 높이면서도, 검색 시에는 더 큰 문맥(부모 문서)을 함께 제공하는 ParentDocumentRetriever를 다루는 노트북입니다.

## 주요 개념

### 1. 문제 상황: 청크 크기의 딜레마
문서 검색을 위해 청크로 나눌 때 두 가지 상충되는 요구사항이 있습니다.
- **작은 청크가 필요한 이유**: 청크가 너무 길면 임베딩이 여러 의미를 뭉뚱그리게 되어, 임베딩이 원래 의미를 정확히 반영하지 못할 수 있습니다.
- **큰 청크(맥락)가 필요한 이유**: 청크가 너무 작으면 문맥 정보가 끊겨서, 검색은 되어도 LLM이 답변을 생성하기에 정보가 부족할 수 있습니다.

### 2. ParentDocumentRetriever의 해결 방식
- 문서를 **작은 자식(child) 청크**로 나누어 벡터스토어에 임베딩·저장해 검색 정확도를 높입니다.
- 각 자식 청크는 그 청크가 속한 **원본(부모) 문서 또는 더 큰 청크의 식별자(ID)** 를 함께 저장합니다.
- 검색 시에는 먼저 작은 자식 청크로 유사도 검색을 수행한 뒤, 해당 청크가 속한 **부모 문서(또는 더 큰 청크) 전체를 반환**하여 충분한 맥락을 제공합니다.
- 즉, "검색 정확도는 작은 청크로, 맥락 제공은 큰 문서로" 라는 두 마리 토끼를 모두 잡는 구조입니다.

### 3. 구성 요소
- **`vectorstore` (예: Chroma)**: 자식(작은) 청크들의 임베딩 벡터를 저장하고 유사도 검색을 수행하는 벡터스토어.
- **`docstore` (예: `InMemoryStore`)**: 부모 문서(또는 더 큰 청크)의 원문을 ID와 함께 저장하는 저장소.
- **`child_splitter`**: 자식 청크를 만드는 텍스트 분할기.
- **`parent_splitter`** (선택): 부모 청크를 만드는 텍스트 분할기. 지정하지 않으면 원본 문서 전체가 "부모"가 됩니다.

### 4. 모드 1 — 전체 문서 검색 (parent_splitter 미지정)
```python
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
vectorstore = Chroma(collection_name="full_documents", embedding_function=OpenAIEmbeddings())
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
retriever.add_documents(docs, ids=None, add_to_docstore=True)
```
- `parent_splitter`를 지정하지 않으면, 자식 청크가 속한 "부모"는 **원본 문서 전체**가 됩니다.
- `vectorstore.similarity_search()`로 검색하면 작은 청크(200자)가 반환되지만, `retriever.invoke()`로 검색하면 그 청크가 포함된 **원본 문서 전체**(예제에서 길이 5733자)가 반환됩니다.
- 문서가 매우 클 경우, 부모 문서 전체를 반환하는 것이 오히려 너무 길어 비효율적일 수 있다는 한계가 드러납니다.

### 5. 모드 2 — 부모/자식 청크 크기를 함께 조절
```python
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # 부모: 큰 청크
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)    # 자식: 작은 청크

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
```
- 원본 문서를 먼저 `parent_splitter`(1000자)로 큼직하게 나눈 뒤, 각 부모 청크를 다시 `child_splitter`(200자)로 잘게 나눕니다.
- 벡터스토어에는 작은 자식 청크만 인덱싱되지만, 검색 시에는 원본 문서 전체가 아니라 **자식 청크가 속한 부모 청크(최대 1000자)** 가 반환됩니다.
- 이를 통해 "검색 정밀도(작은 청크)"와 "적당한 맥락 크기(1000자 부모 청크, 문서 전체보다 훨씬 작음)"의 균형을 맞출 수 있습니다.
- 실습 결과, `store.yield_keys()`의 개수가 7개로 늘어난 것을 확인할 수 있는데(원본 문서가 7개의 부모 청크로 분할됨), 이는 모드 1(부모=문서 전체, 키 1개)과의 차이를 보여줍니다.

## 전체 흐름 요약
1. ParentDocumentRetriever의 필요성(작은 청크의 정확도 vs 큰 청크의 맥락) 이해
2. `child_splitter`만 지정하여 "부모=원본 문서 전체" 모드로 검색 실습 (문서가 너무 커질 수 있는 한계 확인)
3. `parent_splitter`와 `child_splitter`를 함께 지정하여 "부모=중간 크기 청크" 모드로 검색 실습
4. 두 모드에서 벡터스토어의 실제 검색 단위(작은 청크)와 최종 반환 단위(부모 청크/문서)가 어떻게 다른지 비교

## 참고 사항
- `InMemoryStore`는 비영구적 저장소이므로, 실제 서비스에서는 파일 기반이나 DB 기반의 docstore로 교체할 수 있습니다.
- 이 기법은 RAG 파이프라인에서 "검색은 정밀하게, 답변 생성 컨텍스트는 충분하게" 라는 목표를 동시에 달성하기 위한 대표적인 패턴입니다.

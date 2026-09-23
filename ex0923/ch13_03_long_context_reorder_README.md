# ch13_03_long_context_reorder.ipynb

검색된 문서가 많을 때(긴 컨텍스트) LLM 성능이 저하되는 문제를 완화하기 위한 LongContextReorder(긴 문맥 재정렬) 기법을 다루는 노트북입니다.

## 주요 개념

### 1. 문제 상황: "Lost in the Middle"
- 모델의 아키텍처와 관계없이, 검색된 문서가 10개 이상처럼 많아지면 LLM 성능이 상당히 저하되는 경향이 있습니다.
- 특히 **컨텍스트의 중간 부분에 위치한 관련 정보**를 모델이 잘 활용하지 못하고 무시하는 경향이 있습니다(관련 문서가 앞이나 뒤에 있을 때보다 중간에 있을 때 답변 품질이 떨어짐).

### 2. LongContextReorder 클래스
이 문제를 완화하기 위해, 검색 후 **문서의 순서를 재배열**하여 관련성이 높은 문서를 컨텍스트의 시작과 끝 부분에 배치하고, 관련성이 낮은 문서를 중간에 배치하는 기법입니다.
```python
from langchain_community.document_transformers import LongContextReorder

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)
```
- 원래 유사도 순으로 정렬된 문서 리스트를 입력받아, 관련성이 높은 문서일수록 리스트의 양 끝(처음과 끝)에 오도록 재배치합니다.

### 3. 실습 과정
1. `Chroma` 벡터스토어에 다양한 주제(ChatGPT, 애플 제품, 비트코인, 월드컵 등)의 문장 10개를 저장합니다.
2. `retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(search_kwargs={"k": 10})`로 k=10인 리트리버를 생성해 일부러 많은 문서를 검색하도록 설정합니다.
3. "ChatGPT에 대해 무엇을 말해줄 수 있나요?"라는 질의로 10개 문서를 모두 검색한 뒤, 유사도 순으로 정렬된 결과를 확인합니다.
4. `LongContextReorder().transform_documents(docs)`로 문서를 재정렬하여, ChatGPT 관련성이 높은 문서들이 리스트의 시작과 끝에 위치하도록 조정합니다.

### 4. Context Reordering을 적용한 질의-응답 체인 구성
- `format_docs()` 함수로 문서 목록을 번호와 출처(`source`) 정보가 포함된 하나의 문자열(context)로 변환합니다.
- `reorder_documents()` 함수에서 검색된 문서를 재정렬한 뒤 문자열로 합칩니다.
- LCEL(LangChain Expression Language) 파이프라인으로 체인을 구성합니다:
  ```python
  chain = (
      {
          "context": itemgetter("question") | retriever | RunnableLambda(reorder_documents),
          "question": itemgetter("question"),
          "language": itemgetter("language"),
      }
      | prompt
      | ChatOpenAI(model="gpt-4o-mini")
      | StrOutputParser()
  )
  ```
  - 질문을 받아 리트리버로 검색 → 재정렬 → 프롬프트에 컨텍스트로 삽입 → LLM 호출 → 문자열로 파싱하는 전체 RAG 흐름을 하나의 체인으로 표현합니다.
- 최종적으로 질문("ChatGPT에 대해 무엇을 말해줄 수 있나요?")과 원하는 답변 언어("KOREAN")를 입력하면, 재정렬된 컨텍스트를 바탕으로 LLM이 한국어로 정리된 답변을 생성합니다.

## 전체 흐름 요약
1. 여러 주제가 섞인 문서를 벡터스토어에 저장하고 일부러 많은 개수(k=10)를 검색
2. 유사도 순으로 정렬된 검색 결과를 확인 (관련 문서가 중간에 위치할 수 있음)
3. `LongContextReorder`로 관련 문서를 컨텍스트 양 끝으로 재배치
4. 재정렬된 문서를 컨텍스트로 사용하는 LCEL 기반 질의응답 체인을 구성하여 실제 답변 생성까지 확인

## 참고 사항
- 이 기법은 검색 결과 자체를 바꾸는 것이 아니라 **LLM에 전달하는 순서만 조정**하여, 동일한 검색 품질에서도 LLM이 관련 정보를 더 잘 활용하도록 돕는 후처리(post-processing) 기법입니다.
- 문서가 적을 때(예: k=2~3)는 효과가 크지 않지만, 검색 결과가 많아질수록(예: k=10 이상) 유용합니다.

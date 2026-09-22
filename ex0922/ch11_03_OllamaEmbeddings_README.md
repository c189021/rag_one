# ch11_03_OllamaEmbeddings.ipynb

Ollama를 이용한 로컬 임베딩 모델 사용을 위한 노트북이지만, 현재는 초기 준비 단계만 작성되어 있는 미완성 노트북입니다.

## 주요 개념

### 1. OllamaEmbeddings란
- `Ollama`는 LLM 및 임베딩 모델을 로컬(내 컴퓨터/서버) 환경에서 직접 실행할 수 있게 해주는 도구입니다.
- OpenAI나 Upstage, HuggingFace Inference API처럼 외부 클라우드 API를 호출하는 대신, 로컬에 다운로드한 모델로 임베딩을 생성하므로 **API 비용이 들지 않고 네트워크 없이도 동작**할 수 있다는 장점이 있습니다.
- 다만 로컬 하드웨어(CPU/GPU, 메모리) 성능에 따라 속도가 좌우됩니다.

### 2. 노트북 현재 상태
- 마크다운 제목(`### OllamaEmbeddings`)과 테스트용 문장 리스트(`texts`)만 정의되어 있고, 실제로 `OllamaEmbeddings` 객체를 생성하거나 임베딩을 수행하는 코드는 아직 작성되어 있지 않습니다.
- 다른 노트북들(`ch11_01`, `ch11_02`)과 동일한 5개의 한국어/영어 예제 문장을 사용하는 것으로 보아, 앞선 노트북들과 동일한 방식(문서 임베딩 → 유사도 비교)으로 이어질 것으로 예상되는 초안 단계입니다.

## 참고: 일반적인 OllamaEmbeddings 사용법 (참고용)
실제 코드는 비어 있지만, LangChain 생태계에서 일반적으로 사용하는 방식은 다음과 같습니다.
```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="모델명")  # 예: "nomic-embed-text"
embedded_documents = embeddings.embed_documents(texts)
```
- 사용 전 로컬에 Ollama가 설치되어 있어야 하고, 원하는 임베딩 모델을 `ollama pull`로 미리 받아두어야 합니다.

## 전체 흐름 요약 (예정)
1. 로컬 Ollama 임베딩 모델 로드
2. 예제 문장들을 임베딩 벡터로 변환
3. (다른 노트북과 마찬가지로) 코사인 유사도 등을 이용한 문장 간 유사도 비교 예상

※ 이 노트북은 실행 가능한 임베딩 코드가 아직 작성되지 않은 미완성/초안 상태입니다.

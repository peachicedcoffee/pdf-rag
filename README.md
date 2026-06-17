# PDF RAG Chatbot

PDF 문서를 업로드하고 자연어로 질문할 수 있는 RAG 기반 챗봇입니다.
여러 PDF를 동시에 등록할 수 있고, Streamlit 웹 UI에서 채팅 형태로 질문할 수 있습니다.

## 기술 스택
- Python 3.11
- LangChain
- OpenAI GPT-4o-mini
- OpenAI text-embedding-3-small
- ChromaDB
- Streamlit
- PyPDF

## 기능
- 여러 PDF 문서 동시 업로드 및 인덱싱
- 채팅 형태의 질의응답 UI
- 답변에 참고한 문서명, 페이지, 내용 일부 확인 가능

## 실행 방법

### 1. 패키지 설치
pip install -r requirements.txt

### 2. 환경변수 설정
.env 파일 생성 후 OpenAI API 키 입력
OPENAI_API_KEY=your_api_key_here

### 3. 웹 앱 실행
streamlit run app.py

실행하면 브라우저가 자동으로 열립니다 (기본: localhost:8501).
왼쪽 사이드바에서 PDF를 업로드하고 "문서 처리하기"를 누르면 됩니다.

## CLI로 사용하고 싶은 경우

### PDF 임베딩
python ingest.py

data 폴더에 있는 모든 PDF를 읽어 chroma_db에 저장합니다.

### 질문하기
python query.py
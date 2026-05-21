# PDF RAG Chatbot

PDF 문서를 업로드하고 자연어로 질문할 수 있는 RAG 기반 챗봇입니다.

## 기술 스택
- Python 3.11
- LangChain
- OpenAI GPT-4o-mini
- ChromaDB
- PyPDF

## 실행 방법

### 1. 패키지 설치
pip install -r requirements.txt

### 2. 환경변수 설정
.env 파일 생성 후 OpenAI API 키 입력
OPENAI_API_KEY=your_api_key_here

### 3. data 폴더 생성 후 PDF 파일 넣기
mkdir data
data/ 폴더 안에 질문할 PDF 파일을 넣어주세요.

### 4. PDF 임베딩 (chroma_db 폴더 자동 생성됨)
python ingest.py data/파일명.pdf

### 5. 질문하기
python query.py

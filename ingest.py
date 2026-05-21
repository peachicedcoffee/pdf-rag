# pdf읽기 -> 청크로 자르기 -> 임베딩 -> 벡터db 저장
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest(pdf_path: str):
    print(f"[1/4] PDF 로딩 중: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"      → {len(documents)}페이지 로드 완료")

    print("[2/4] 텍스트 청크로 분할 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"      → {len(chunks)}개 청크 생성 완료")

    print("[3/4] 임베딩 생성 및 벡터 DB 저장 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"      → 벡터 DB 저장 완료 (chroma_db/)")
    print("[4/4] 완료!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python ingest.py <PDF파일경로>")
        print("예시:   python ingest.py data/sample.pdf")
    else:
        ingest(sys.argv[1])
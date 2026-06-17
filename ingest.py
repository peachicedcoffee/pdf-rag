# pdf읽기 -> 청크로 자르기 -> 임베딩 -> 벡터db 저장
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_pdfs(folder_path: str):
    documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f" '{folder_path}' 폴더에 pdf 파일이 없습니다.")
        return documents
    
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        print(f"    - 로딩: {filename}")
        loader = PyPDFLoader(filepath)
        documents.extend(loader.load())
        
    return documents
    
def ingest_folder(folder_path: str = "data", persist_directory: str = "chroma_db"):
    print(f"[1/4] pdf 폴더 스캔 중: {folder_path}")
    documents = load_pdfs(folder_path)
    if not documents:
        return None
    print(f"    -> 총 {len(documents)}페이지 로드 완료")
    
    print("[2/4] 텍스트 청크로 분할 중...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"    -> {len(chunks)}개 청크 생성 완료")
    
    print("[3/4] 임베딩 생성 및 벡터 db 저장 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"    -> 벡터 db 저장 완료 ({persist_directory}/)")
    print("[4/4] 완료!")
    return db


if __name__ == "__main__":
        ingest_folder()
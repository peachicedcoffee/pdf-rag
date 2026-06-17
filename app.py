import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingest import ingest_folder

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="pdf Q&A 챗봇", page_icon="📄")
st.title("📄 pdf Q&A 챗봇")

os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_pipeline():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k":5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = PromptTemplate.from_template("""
아래 컨텍스트를 참고해서 질문에 답변해주세요.
컨텍스트에 없는 내용은 모른다고 답변하세요.
                                        
컨텍스트:
{context}
                                        
질문: {question}
                                        
답변:""")
    
    return retriever, prompt, llm

with st.sidebar:
    st.header("문서 업로드")
    uploaded_files = st.file_uploader(
        "pdf 파일을 선택하세요 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True     
    )
    
    if st.button("문서 처리하기", disabled=not uploaded_files):
        for file in uploaded_files:
            save_path = os.path.join(DATA_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
                
        with st.spinner("문서를 분석하고 있어요..."):
            ingest_folder(DATA_DIR, PERSIST_DIR)
            
        load_pipeline.clear()
        st.success("처리 완료! 이제 질문해보세요.")
        st.rerun()
        
    st.divider()
    existing = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if existing:
        st.caption("현재 등록된 문서")
        for f in existing:
            st.text(f"- {f}")
            



if not os.path.exists(PERSIST_DIR):
    st.info("왼쪽에서 pdf를 업로드하고 '문서 처리하기'를 눌러주세요.")
else:
    retriever, prompt, llm = load_pipeline()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("참고한 문서 내용 보기"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {src['filename']} - {src['page']}페이지**")
                        st.text(src["snippet"])
            
    question = st.chat_input("질문을 입력하세요")
    if question:
        st.session_state.messages.append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있어요..."):
                docs = retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context":context, "question":question})
                sources = []
                for doc in docs:
                    filename = os.path.basename(doc.metadata.get("source", "알 수 없음"))
                    page = doc.metadata.get("page", -1)+1
                    sources.append({
                        "filename": filename,
                        "page": page,
                        "snippet": doc.page_content[:300]
                    })    
                                  
            st.markdown(answer)
            with st.expander("참고한 문서 내용 보기"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}. {src['filename']} - {src['page']}페이지**")
                    st.text(src["snippet"])
                
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources })
# 질문입력 -> 벡터db 검색 -> LLM 답변 생성
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query():
    print("벡터 DB 로딩 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
아래 컨텍스트를 참고해서 질문에 답변해주세요.
컨텍스트에 없는 내용은 모른다고 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("준비 완료! 질문을 입력하세요. (종료: 'q' 입력)\n")

    while True:
        question = input("질문: ").strip()
        if question.lower() == 'q':
            print("종료합니다.")
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\n답변: {answer}\n")

if __name__ == "__main__":
    query()
    print("벡터 DB 로딩 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
아래 컨텍스트를 참고해서 질문에 답변해주세요.
컨텍스트에 없는 내용은 모른다고 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("준비 완료! 질문을 입력하세요. (종료: 'q' 입력)\n")
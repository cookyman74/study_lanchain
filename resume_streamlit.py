import streamlit as st
import os
import json
import hashlib
import faiss
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from dotenv import load_dotenv
import asyncio

# 환경 변수 로드
load_dotenv()

# 환경 변수 사용
openai_api_key = os.getenv('OPENAI_API_KEY')
faiss_index_path = os.getenv('FAISS_INDEX_PATH')

# FAISS 벡터 저장소 초기화
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
dimension = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(dimension)

if os.path.exists(faiss_index_path):
    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    st.error("FAISS 인덱스를 찾을 수 없습니다.")
    st.stop()

# Streamlit 인터페이스
st.title("입사 지원서 RAG 챗봇")
st.write("입사 지원서에 대한 질문을 입력하세요.")

query = st.text_input("질문:")

if st.button("질문 제출"):
    if query:
        # FAISS에서 관련 문서 검색
        results = vector_store.similarity_search(query, k=1)

        if results:
            related_doc = results[0]
            related_text = related_doc.page_content

            # LangChain을 사용하여 응답 생성
            prompt = ChatPromptTemplate.from_template(
                """
                사용자 질문: {USER_QUERY}
                참고 정보: {RELATED_TEXT}
                
                질문에 대한 답변을 작성하세요.
                """
            )

            chain = (
                    prompt
                    | ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo"
            )
            )

            response = chain.invoke({"USER_QUERY": query, "RELATED_TEXT": related_text})
            response_text = response.content  # AIMessage 객체의 content 속성을 사용

            st.write("**답변:**")
            st.write(response_text)
        else:
            st.write("관련 정보를 찾을 수 없습니다.")
    else:
        st.error("질문을 입력하세요.")

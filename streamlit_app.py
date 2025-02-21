import streamlit as st
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
import asyncio
from utils.tool_calling_event import invoke_our_graph
from utils.agent import create_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Optional, Tuple
from langchain.tools.retriever import create_retriever_tool
# import chromadb

# chromadb 캐시 클리어
# chromadb.api.client.SharedSystemClient.clear_system_cache()

# 환경 변수 로드
load_dotenv()


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [AIMessage(content="무엇을 도와드릴까요?")]
    if "graph" not in st.session_state:
        st.session_state.graph = create_agent()


# Streamlit 앱 시작
st.title("Sourcing Chatbot")
initialize_session_state()

# OpenAI API 키 확인
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 없습니다. .env 파일에 API키를 저장해주세요.")
    st.stop()


# 채팅 인터페이스 표시
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

# 사용자 입력 처리
prompt = st.chat_input("메시지를 입력하세요")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.container()
        try:
            response = asyncio.run(
                invoke_our_graph(
                    {"messages": st.session_state.messages},
                    placeholder,
                    st.session_state.graph,  # 현재 세션의 그래프 전달
                )
            )
            st.session_state.messages.append(AIMessage(content=response))
        except RecursionError as e:
            error_message = f"⚠️ 너무 많은 재귀 호출이 발생했습니다: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))

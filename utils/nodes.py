from utils.tools import get_tools
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from datetime import datetime
import os

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


def get_system_prompt(docs_info):
    system_prompt = f"""
    Today is {datetime.now().strftime("%Y-%m-%d")}
    You are a helpful AI Assistant that can assist E-commerce related questions.
    If user asks unrelated questions, you can ask them to ask again in E-commerce related questions.
    """

    system_prompt += "\nYou should always answer in same language as user's ask."
    return system_prompt


def create_chatbot(docs_info=None, retriever_tool=None):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(get_system_prompt(docs_info)),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # retriever_tool이 있을 경우 tools에 추가
    tools = get_tools(retriever_tool)
    llm_with_tools = llm.bind_tools(tools)
    chain = prompt | llm_with_tools

    def chatbot(state: MessagesState):
        response = chain.invoke(state["messages"])
        return {"messages": response}

    return chatbot

from langchain_core.tools import tool
from openai import OpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import requests
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import json


class GenderAge(BaseModel):
    gender: str = Field(title="gender", description="성별을 입력해주세요")
    age: int = Field(title="age", description="연령대를 입력해주세요")


class KeywordAnalysis(BaseModel):
    productCount: int = Field(title="productCount", description="상품 수")
    isBrandKeyword: bool = Field(title="isBrandKeyword", description="브랜드 키워드 여부")
    smartStoreRelatedKeyWords: list = Field(title="smartStoreRelatedKeyWords", description="스마트스토어 연관 키워드")
    QcCnt: int = Field(title="QcCnt", description="검색량")
    monthlyAvePcClkCnt: float = Field(title="monthlyAvePcClkCnt", description="월평균 PC 클릭수")
    monthlyAveMobileClkCnt: float = Field(title="monthlyAveMobileClkCnt", description="월평균 모바일 클릭수")
    monthlyAvePcCtr: float = Field(title="monthlyAvePcCtr", description="월평균 PC CTR")
    monthlyAveMobileCtr: float = Field(title="monthlyAveMobileCtr", description="월평균 모바일 CTR")
    menuRank: int = Field(title="menuRank", description="메뉴 랭킹")
    bodyRank: int = Field(title="bodyRank", description="본문 랭킹")
    PcBid: int = Field(title="PcBid", description="PC 광고입찰가")
    MobileBid: int = Field(title="MobileBid", description="모바일 광고입찰가")


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
domeme_api_key = os.getenv("DOMEME_API_KEY")


def get_keyword_age(human_message):

    prompt = PromptTemplate(
        template=f"""
                아래 입력된 문장에서 연령대, 성별을 추출하여 주어진 결과 형태로 반환하라.
                {human_message}

                60대 이상인 경우, age는 50으로 반환하라.
                성별을 알수 없는 경우 아래 성별 중 임의로 선택하라.
                연령을 알수 없는 경우 ALL을 선택하라.
                성별,연령을 알수 없는 경우 ALL을 선택하라.
                gender : MEN, WOMEN
                age : 10, 20, 30, 40, 50, ALL
                """,
        input_variables=["human_message"],
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    output_parser = JsonOutputParser(pydantic_object=GenderAge)

    chain = prompt | llm | output_parser

    response = chain.invoke({"query": human_message})

    if response["gender"] == "ALL":
        result = "ALL"
    else:
        result = f"{response['gender']}_{response['age']}"

    return result


@tool
def get_popular_keywords(human_message, age_type="ALL"):
    """
    Get a list of popular keywords
    """
    age_type = get_keyword_age(human_message)

    url = "http://3.37.249.133/getDailyTrendKeyword"

    payload = {"marketName": "smartstore", "ageType": age_type}
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers).json()
    return response["data"]


@tool
def analyze_keyword(keyword):
    """
    Analyze keyword
    """

    url = "http://3.37.249.133/keywordAnalysis/getKeywordAnalysis"

    payload = {"searchKeyWord": keyword}
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers).json()["data"][0]
    result = {
        "productCount": response["productCount"],
        "isBrandKeyword": response["isBrandKeyword"],
        "smartStoreRelatedKeyWords": response["smartStoreRelatedKeyWords"],
        "searchCnt": response["QcCnt"],  # 검색량
        "monthlyAvePcClkCnt": response["monthlyAvePcClkCnt"],
        "monthlyAveMobileClkCnt": response["monthlyAveMobileClkCnt"],
        "monthlyAvePcCtr": response["monthlyAvePcCtr"],
        "monthlyAveMobileCtr": response["monthlyAveMobileCtr"],
        "menuRank": response["menuRank"],
        "bodyRank": response["bodyRank"],
        "PcBid": response["PcBid"],
        "MobileBid": response["MobileBid"],
    }

    # result = KeywordAnalysis(**result)
    return result


@tool(parse_docstring=True)
def domeme_keyword_product(keyword, endpoint="domeme", sortedBy="popular"):
    """
    endpoint : {"도매매":"domeme", "도매꾹":"domeggook", "오너클랜":"ownerclan"}
    """

    url = "http://3.37.249.133/itemRecommendation/getWholesaleItems"

    payload = {
        "wholesaleMall": endpoint,
        "keyword": keyword,
        "sortedBy": sortedBy,
    }
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers).json()

    result_list = [
        {
            "title": item["title"],
            "price": int(item["price"]),
            "item_key": item["item_key"],
            "url": item["url"],
            "image_url": item["image_url"],
            "delivery_fee": int(item["delivery_fee"]),
            "unit_quantity": int(item["unit_quantity"]),
        }
        for item in response["data"][0]["item_list"]
    ]

    return result_list


def get_tools(retriever_tool=None):
    base_tools = [get_popular_keywords, analyze_keyword, domeme_keyword_product]
    if retriever_tool:
        base_tools.append(retriever_tool)
    return base_tools

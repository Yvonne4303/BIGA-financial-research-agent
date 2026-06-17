from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from loguru import logger

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            temperature=0.3,
        )
    return _llm


HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的A股财务分析师，请根据以下问题，生成一段可能出现在财报中的假设性答案段落。要求：使用财报语言风格，包含具体数字（可虚构），约100字。"),
    ("human", "{query}"),
])


def generate_hypothetical_doc(query: str) -> str:
    """HyDE：生成假设性文档，用于提升向量检索的召回率"""
    try:
        chain = HYDE_PROMPT | get_llm()
        result = chain.invoke({"query": query})
        hypothetical = result.content
        logger.debug(f"HyDE 生成: {hypothetical[:60]}...")
        # 将原始问题与假设文档合并，保留语义
        return f"{query}\n\n{hypothetical}"
    except Exception as e:
        logger.warning(f"HyDE 生成失败，回退到原始 query: {e}")
        return query

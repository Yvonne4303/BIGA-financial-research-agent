from tavily import TavilyClient
from langchain_core.tools import tool
from config.settings import TAVILY_API_KEY
from loguru import logger

_client = None

def get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


@tool
def search_financial_news(query: str, max_results: int = 5) -> str:
    """搜索最新财经新闻。query 示例：'茅台 2024 业绩 分红'"""
    try:
        client = get_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=["eastmoney.com", "10jqka.com.cn", "sina.com.cn",
                             "caixin.com", "yicai.com", "cls.cn"],
        )
        results = response.get("results", [])
        if not results:
            return "未找到相关新闻"
        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"[{i}] {r['title']}\n"
                f"    来源: {r.get('url','')}\n"
                f"    摘要: {r.get('content','')[:200]}"
            )
        return "\n\n".join(output)
    except Exception as e:
        logger.error(f"search_financial_news error: {e}")
        return f"搜索失败: {str(e)}"


@tool
def search_company_news(company_name: str, days: int = 7) -> str:
    """搜索某公司最近 N 天的重要公告和新闻。company_name 示例：'贵州茅台'"""
    query = f"{company_name} 最新公告 重大事项 {days}天内"
    return search_financial_news.invoke({"query": query, "max_results": 8})

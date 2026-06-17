from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from tools.stock_data import get_stock_basic_info, get_stock_price, get_financial_indicators, get_price_dataframe
from tools.web_search import search_financial_news, search_company_news
from tools.python_sandbox import run_python_code, KLINE_CODE_TEMPLATE
from tools.feishu_webhook import send_research_report
from memory.long_term import recall_user_context, recall_stock_history, remember_stock_judgment
from loguru import logger
import re


# ── State Schema ─────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    ts_code: str                    # 股票代码，如 600519.SH
    company_name: str               # 公司名称
    user_id: str                    # 用户 ID（用于长期记忆）
    user_memory: str                # 用户历史记忆
    stock_history: str              # 该股历史判断

    # Agent 1: 信息收集
    basic_info: str
    price_info: str
    financial_info: str
    news_info: str

    # Agent 2: RAG 分析（财报）
    rag_analysis: str

    # Agent 3: 量化分析
    chart_result: str
    quant_analysis: str

    # Agent 4: 汇总审查
    draft_report: str
    self_correction_needed: bool
    correction_reason: str
    final_report: str

    # 迭代计数（防止无限纠错）
    correction_count: int

    messages: Annotated[list, add_messages]


# ── LLM ──────────────────────────────────────────────────────────────────────

def make_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        timeout=90,
        max_retries=1,
    )


# ── Node 1: 信息收集 Agent ────────────────────────────────────────────────────

def info_collector_node(state: ResearchState) -> dict:
    logger.info(f"[Agent1] 开始收集 {state['ts_code']} 信息")
    ts_code = state["ts_code"]

    basic = get_stock_basic_info.invoke({"ts_code": ts_code})
    price = get_stock_price.invoke({"ts_code": ts_code, "days": 60})
    financial = get_financial_indicators.invoke({"ts_code": ts_code})

    # 从基础信息中提取公司名称
    name = state.get("company_name", "")
    if not name:
        match = re.search(r"公司名称: (.+)", basic)
        name = match.group(1).strip() if match else ts_code

    news = search_company_news.invoke({"company_name": name, "days": 7})

    # 检索用户记忆
    user_mem = recall_user_context(state["user_id"], f"{name} 投资分析")
    stock_hist = recall_stock_history(state["user_id"], ts_code)

    return {
        "company_name": name,
        "basic_info": basic,
        "price_info": price,
        "financial_info": financial,
        "news_info": news,
        "user_memory": user_mem,
        "stock_history": stock_hist,
        "messages": [HumanMessage(content=f"开始分析 {name}（{ts_code}）")],
    }


# ── Node 2: RAG 分析 Agent ────────────────────────────────────────────────────

RAG_SYSTEM = """你是一位专业的A股财务分析师。根据提供的信息，对公司基本面做深度分析。
重点关注：
1. 盈利能力（净利润、ROE、毛利率趋势）
2. 估值水平（PE/PB 与历史和行业对比）
3. 成长性（营收/利润增速）
4. 风险因素（负债率、商誉、政策风险）
请用专业但易懂的语言，约 300 字。"""

def rag_analyst_node(state: ResearchState) -> dict:
    logger.info(f"[Agent2] 开始 RAG 分析 {state['company_name']}")
    llm = make_llm()
    prompt = f"""
公司：{state['company_name']}（{state['ts_code']}）

基础信息：
{state['basic_info']}

行情数据：
{state['price_info']}

财务指标：
{state['financial_info']}

用户背景（参考）：
{state['user_memory']}
{state['stock_history']}

请对该公司基本面进行深度分析。
"""
    result = llm.invoke([SystemMessage(content=RAG_SYSTEM), HumanMessage(content=prompt)])
    return {
        "rag_analysis": result.content,
        "messages": [AIMessage(content=f"基本面分析完成: {result.content[:100]}...")],
    }


# ── Node 3: 量化分析 Agent ────────────────────────────────────────────────────

QUANT_SYSTEM = """你是一位量化研究员。根据K线图和技术指标，给出技术面分析。
重点：MACD 金叉/死叉信号、均线支撑/压力位、成交量配合情况、短期趋势判断。约 150 字。"""

def quant_analyst_node(state: ResearchState) -> dict:
    logger.info(f"[Agent3] 开始量化分析 {state['company_name']}")

    # 获取行情 CSV 数据并在沙盒中画图
    csv_data = get_price_dataframe.invoke({"ts_code": state["ts_code"], "days": 120})
    chart_code = KLINE_CODE_TEMPLATE.format(title=f"{state['company_name']} K线图")
    chart_result = run_python_code.invoke({"code": chart_code, "stock_csv": csv_data})

    # LLM 根据行情数据做技术分析
    llm = make_llm(temperature=0.2)
    prompt = f"""
公司：{state['company_name']}（{state['ts_code']}）

行情数据（最近60日）：
{state['price_info']}

图表生成结果：{chart_result}

请给出技术面分析。
"""
    result = llm.invoke([SystemMessage(content=QUANT_SYSTEM), HumanMessage(content=prompt)])
    return {
        "chart_result": chart_result,
        "quant_analysis": result.content,
        "messages": [AIMessage(content=f"技术分析完成")],
    }


# ── Node 4: 审查 & 汇总 Agent ─────────────────────────────────────────────────

REVIEW_SYSTEM = """你是一位资深投研总监。你的职责是：
1. 汇总各 Agent 分析结果，生成最终投研简报
2. 检查逻辑一致性：如"大跌但结论强烈推荐"这类矛盾
3. 如发现矛盾，在报告末尾标注 [CORRECTION_NEEDED: 原因]

报告结构：核心观点 | 基本面 | 技术面 | 近期催化剂 | 投资建议 | 风险提示
约 400 字。"""

def review_agent_node(state: ResearchState) -> dict:
    logger.info(f"[Agent4] 开始审查汇总 {state['company_name']}")
    llm = make_llm(temperature=0.4)
    prompt = f"""
公司：{state['company_name']}（{state['ts_code']}）

=== 基本面分析 ===
{state['rag_analysis']}

=== 技术面分析 ===
{state['quant_analysis']}

=== 最新新闻 ===
{state['news_info']}

=== 用户偏好 ===
{state['user_memory']}

请生成最终投研简报。如有逻辑矛盾，在末尾标注 [CORRECTION_NEEDED: 具体原因]。
"""
    result = llm.invoke([SystemMessage(content=REVIEW_SYSTEM), HumanMessage(content=prompt)])
    report = result.content

    # 检测是否需要自我纠错
    _NO_CORRECTION = ["未发现", "没有矛盾", "无矛盾", "逻辑一致", "不存在矛盾", "无需"]
    correction_match = re.search(r"\[CORRECTION_NEEDED:\s*(.+?)\]", report, re.IGNORECASE)
    correction_reason = correction_match.group(1).strip() if correction_match else ""
    _trivial = any(kw in correction_reason for kw in _NO_CORRECTION)
    needs_correction = bool(correction_match) and not _trivial and state.get("correction_count", 0) < 2

    return {
        "draft_report": report,
        "self_correction_needed": needs_correction,
        "correction_reason": correction_reason,
        "correction_count": state.get("correction_count", 0) + (1 if needs_correction else 0),
        "messages": [AIMessage(content=f"审查完成，需要纠错: {needs_correction}")],
    }


# ── Node 5: Self-Correction ───────────────────────────────────────────────────

CORRECTION_SYSTEM = """你是一位投研质控专家。请根据指出的矛盾，重新生成一份逻辑自洽的投研简报。
严格要求：结论必须与数据一致，不得出现自相矛盾。"""

def self_correction_node(state: ResearchState) -> dict:
    logger.info(f"[SelfCorrection] 纠错原因: {state['correction_reason']}")
    llm = make_llm(temperature=0.2)
    prompt = f"""
原始报告（存在逻辑问题）：
{state['draft_report']}

发现的矛盾：{state['correction_reason']}

原始数据参考：
- 行情: {state['price_info']}
- 财务: {state['financial_info']}
- 基本面分析: {state['rag_analysis']}
- 技术分析: {state['quant_analysis']}

请修正逻辑矛盾，重新生成完整投研简报（不要包含 [CORRECTION_NEEDED] 标注）。
"""
    result = llm.invoke([SystemMessage(content=CORRECTION_SYSTEM), HumanMessage(content=prompt)])
    return {
        "draft_report": result.content,
        "self_correction_needed": False,
        "messages": [AIMessage(content="自我纠错完成")],
    }


# ── Node 6: 最终输出 & 记忆存储 ──────────────────────────────────────────────

def finalize_node(state: ResearchState) -> dict:
    logger.info(f"[Finalize] 生成最终报告并推送")
    report = state["draft_report"]

    # 提取各部分用于飞书卡片（简单分段）
    lines = report.split("\n")
    summary = "\n".join(lines[:3]) if len(lines) >= 3 else report[:200]

    feishu_result = send_research_report.invoke({
        "company_name": state["company_name"],
        "ts_code": state["ts_code"],
        "summary": summary,
        "price_info": state["price_info"][:300],
        "financial_info": state["financial_info"][:300],
        "news_highlights": state["news_info"][:400],
        "investment_view": report[-400:] if len(report) > 400 else report,
        "risk_warning": "本报告由AI自动生成，仅供参考，不构成投资建议。市场有风险，投资需谨慎。",
    })

    # 存储本次判断到长期记忆
    remember_stock_judgment(
        state["user_id"],
        state["ts_code"],
        f"投研简报摘要: {summary[:100]}"
    )

    return {
        "final_report": report,
        "messages": [AIMessage(content=f"报告完成。{feishu_result}")],
    }


# ── 路由函数 ──────────────────────────────────────────────────────────────────

def should_correct(state: ResearchState) -> str:
    if state.get("self_correction_needed") and state.get("correction_count", 0) <= 2:
        return "self_correction"
    return "finalize"


# ── 构建 Graph ────────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("info_collector", info_collector_node)
    graph.add_node("rag_analyst", rag_analyst_node)
    graph.add_node("quant_analyst", quant_analyst_node)
    graph.add_node("review_agent", review_agent_node)
    graph.add_node("self_correction", self_correction_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("info_collector")
    graph.add_edge("info_collector", "rag_analyst")
    graph.add_edge("rag_analyst", "quant_analyst")
    graph.add_edge("quant_analyst", "review_agent")
    graph.add_conditional_edges(
        "review_agent",
        should_correct,
        {"self_correction": "self_correction", "finalize": "finalize"},
    )
    graph.add_edge("self_correction", "review_agent")
    graph.add_edge("finalize", END)

    return graph.compile()


# ── 对外接口 ──────────────────────────────────────────────────────────────────

_app = None

def get_research_app():
    global _app
    if _app is None:
        _app = build_research_graph()
    return _app


def run_research(ts_code: str, user_id: str = "default_user", company_name: str = "") -> str:
    """触发完整投研流程，返回最终报告文本"""
    app = get_research_app()
    initial_state: ResearchState = {
        "ts_code": ts_code,
        "company_name": company_name,
        "user_id": user_id,
        "user_memory": "",
        "stock_history": "",
        "basic_info": "",
        "price_info": "",
        "financial_info": "",
        "news_info": "",
        "rag_analysis": "",
        "chart_result": "",
        "quant_analysis": "",
        "draft_report": "",
        "self_correction_needed": False,
        "correction_reason": "",
        "final_report": "",
        "correction_count": 0,
        "messages": [],
    }
    result = app.invoke(initial_state)
    return result.get("final_report", "")

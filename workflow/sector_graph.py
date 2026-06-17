from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from tools.sector_data import get_all_sectors_performance, get_sector_history, search_sector_news
from tools.feishu_webhook import send_research_report
from memory.long_term import recall_user_context, remember_user_preference
from loguru import logger


# ── State ────────────────────────────────────────────────────────────────────

class SectorState(TypedDict):
    period: str                  # "日" / "周" / "月"
    user_id: str                 # 用户ID，用于读取个人记忆
    all_sectors: list            # 全部板块涨跌数据
    top_gainers: list            # 涨幅前5
    top_losers: list             # 跌幅前5
    sector_analyses: list        # 每个板块的详细分析
    user_memory: str             # 从 Mem0 读取的用户记忆
    final_report: str
    messages: Annotated[list, add_messages]


# ── LLM ─────────────────────────────────────────────────────────────────────

def make_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        timeout=90,
        max_retries=1,
    )


def _llm_invoke(llm: ChatOpenAI, messages: list, retries: int = 2) -> AIMessage:
    import time
    for i in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if i < retries - 1:
                logger.warning(f"LLM调用失败(第{i+1}次): {e}，5秒后重试")
                time.sleep(5)
            else:
                raise


# ── Node 1: 板块筛选 ──────────────────────────────────────────────────────────

def screener_node(state: SectorState) -> dict:
    period = state["period"]
    user_id = state.get("user_id", "default_user")
    logger.info(f"[板块筛选] 周期: {period}, 用户: {user_id}")

    # 读取用户记忆（持仓、偏好、历史关注）
    user_memory = recall_user_context(user_id, "持仓 关注板块 风险偏好 上周判断")
    logger.info(f"[记忆] {user_memory[:80] if user_memory else '无历史记忆'}...")

    sectors = get_all_sectors_performance(period)

    # 过滤掉涨跌幅为0的（数据缺失）
    valid = [s for s in sectors if s["change_pct"] != 0.0]
    if not valid:
        valid = sectors

    gainers = sorted(valid, key=lambda x: x["change_pct"], reverse=True)[:5]
    losers = sorted(valid, key=lambda x: x["change_pct"])[:5]

    logger.info(f"涨幅前5: {[s['name'] for s in gainers]}")
    logger.info(f"跌幅前5: {[s['name'] for s in losers]}")

    return {
        "all_sectors": sectors,
        "top_gainers": gainers,
        "user_memory": user_memory,
        "top_losers": losers,
        "messages": [HumanMessage(content=f"开始{period}度板块分析")],
    }


# ── Node 2: 板块深度分析 ──────────────────────────────────────────────────────

SECTOR_ANALYST_SYSTEM = """你是一位专业的A股行业研究员。请根据提供的板块涨跌数据和新闻，
分析该行业的近期表现驱动因素，并给出短期展望。
要求：语言简洁专业，约150字，重点说明：
1. 近期涨跌的核心驱动（政策/业绩/资金/情绪）
2. 板块内值得关注的方向或风险
3. 一句话短期展望"""

def _analyze_one_sector(sector: dict, period: str, llm: ChatOpenAI) -> dict:
    name = sector["name"]
    change = sector["change_pct"]

    # 获取新闻
    news = search_sector_news.invoke({"sector_name": name, "days": 7})
    history = get_sector_history(name, days=20)

    prompt = f"""
行业板块：{name}
近期涨跌幅（{period}）：{change:+.2f}%
历史走势：{history}
最新新闻：
{news}

请分析该板块近期表现。
"""
    result = _llm_invoke(llm, [SystemMessage(content=SECTOR_ANALYST_SYSTEM), HumanMessage(content=prompt)])
    return {
        "name": name,
        "change_pct": change,
        "analysis": result.content,
        "direction": "上涨" if change >= 0 else "下跌",
    }


def researcher_node(state: SectorState) -> dict:
    logger.info("[板块分析] 开始分析涨跌幅前后各5板块")
    llm = make_llm(temperature=0.3)
    period = state["period"]

    analyses = []
    targets = state["top_gainers"] + state["top_losers"]
    # 去重（极端情况下涨跌幅列表可能重叠）
    seen = set()
    unique_targets = []
    for s in targets:
        if s["name"] not in seen:
            seen.add(s["name"])
            unique_targets.append(s)

    for i, sector in enumerate(unique_targets):
        logger.info(f"  分析 [{i+1}/{len(unique_targets)}]: {sector['name']} {sector['change_pct']:+.2f}%")
        try:
            analysis = _analyze_one_sector(sector, period, llm)
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"  {sector['name']} 分析失败: {e}")
            analyses.append({
                "name": sector["name"],
                "change_pct": sector["change_pct"],
                "analysis": "分析失败",
                "direction": "上涨" if sector["change_pct"] >= 0 else "下跌",
            })

    return {
        "sector_analyses": analyses,
        "messages": [AIMessage(content=f"完成 {len(analyses)} 个板块分析")],
    }


# ── Node 3: 汇总报告 ──────────────────────────────────────────────────────────

REPORT_SYSTEM = """你是一位资深A股策略分析师，同时了解该用户的个人持仓和偏好。
请根据各行业板块分析，撰写一份个性化的板块轮动策略报告。结构：
1. 市场整体情绪判断（一句话）
2. 强势板块（涨幅前列）：共同驱动逻辑
3. 弱势板块（跌幅前列）：共同压制因素
4. 个性化建议：结合用户持仓和偏好，说明哪些板块值得关注或规避
5. 风险提示
约400字，专业简洁。若无用户记忆则忽略第4条个性化部分。"""

def reporter_node(state: SectorState) -> dict:
    logger.info("[汇总报告] 生成板块轮动报告")
    llm = make_llm(temperature=0.4)
    period = state["period"]
    analyses = state["sector_analyses"]
    user_memory = state.get("user_memory", "")

    # 今日分析结束后，自动记录本次报告的强弱板块，供下次参考
    user_id = state.get("user_id", "default_user")
    top3 = [a["name"] for a in analyses[:3]]
    try:
        remember_user_preference(user_id, f"{period}度分析：强势板块为{','.join(top3)}")
    except Exception:
        pass

    gainers_text = "\n".join([
        f"- {a['name']}: {a['change_pct']:+.2f}% | {a['analysis']}"
        for a in analyses if a["change_pct"] >= 0
    ])
    losers_text = "\n".join([
        f"- {a['name']}: {a['change_pct']:+.2f}% | {a['analysis']}"
        for a in analyses if a["change_pct"] < 0
    ])

    prompt = f"""
分析周期：{period}度
用户背景（个人持仓与偏好）：
{user_memory or '暂无用户记忆'}

涨幅领先板块：
{gainers_text or '无数据'}

跌幅领先板块：
{losers_text or '无数据'}

请生成板块轮动策略报告。
"""
    result = _llm_invoke(llm, [SystemMessage(content=REPORT_SYSTEM), HumanMessage(content=prompt)])
    report = result.content

    # 推送到飞书
    period_label = {"日": "今日", "周": "本周", "月": "本月"}.get(period, period)
    top3_names = " | ".join([a["name"] for a in analyses[:3]])
    summary = f"{period_label}强势板块：{top3_names}\n" + "\n".join(report.split("\n")[:3])

    from tools.feishu_webhook import _send_card
    from datetime import datetime
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📊 A股{period_label}行业板块轮动报告 {datetime.today().strftime('%m-%d')}"},
            "template": "green",
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**{period_label}强势板块**\n" + "\n".join(
                f"🔴 {a['name']} {a['change_pct']:+.2f}%" for a in analyses if a["change_pct"] >= 0
            )}},
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**{period_label}弱势板块**\n" + "\n".join(
                f"🟢 {a['name']} {a['change_pct']:+.2f}%" for a in analyses if a["change_pct"] < 0
            )}},
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**策略报告**\n{report}"}},
            {"tag": "div", "text": {"tag": "lark_md", "content": "⚠️ 本报告由AI自动生成，仅供参考，不构成投资建议。"}},
        ],
    }
    _send_card(card)

    return {
        "final_report": report,
        "messages": [AIMessage(content="板块报告已生成并推送")],
    }


# ── 构建 Graph ────────────────────────────────────────────────────────────────

def build_sector_graph() -> StateGraph:
    graph = StateGraph(SectorState)
    graph.add_node("screener", screener_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reporter", reporter_node)
    graph.set_entry_point("screener")
    graph.add_edge("screener", "researcher")
    graph.add_edge("researcher", "reporter")
    graph.add_edge("reporter", END)
    return graph.compile()


_app = None

def run_sector_analysis(period: str = "日", user_id: str = "default_user") -> str:
    global _app
    if _app is None:
        _app = build_sector_graph()
    initial: SectorState = {
        "period": period,
        "user_id": user_id,
        "all_sectors": [],
        "top_gainers": [],
        "top_losers": [],
        "sector_analyses": [],
        "user_memory": "",
        "final_report": "",
        "messages": [],
    }
    result = _app.invoke(initial)
    return result.get("final_report", "")

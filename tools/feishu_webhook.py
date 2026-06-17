import httpx
from langchain_core.tools import tool
from config.settings import FEISHU_WEBHOOK_URL
from loguru import logger


def _send_card(card: dict) -> bool:
    try:
        resp = httpx.post(
            FEISHU_WEBHOOK_URL,
            json={"msg_type": "interactive", "card": card},
            timeout=10,
        )
        data = resp.json()
        if data.get("StatusCode") == 0 or data.get("code") == 0:
            return True
        logger.warning(f"飞书返回: {data}")
        return False
    except Exception as e:
        logger.error(f"飞书推送失败: {e}")
        return False


@tool
def send_research_report(
    company_name: str,
    ts_code: str,
    summary: str,
    price_info: str,
    financial_info: str,
    news_highlights: str,
    investment_view: str,
    risk_warning: str,
) -> str:
    """将投研报告以飞书富文本卡片形式推送。"""
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📊 {company_name}（{ts_code}）投研简报"},
            "template": "blue",
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**核心观点**\n{summary}"},
            },
            {"tag": "hr"},
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**行情数据**\n{price_info}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**财务指标**\n{financial_info}"}},
                ],
            },
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**近期新闻**\n{news_highlights}"}},
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**投资观点**\n{investment_view}"}},
            {"tag": "div", "text": {"tag": "lark_md", "content": f"⚠️ **风险提示**\n{risk_warning}"}},
        ],
    }
    ok = _send_card(card)
    return "飞书推送成功" if ok else "飞书推送失败，请检查 Webhook URL"

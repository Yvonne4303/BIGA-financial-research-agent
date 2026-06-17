# BIGA — A-Share Multi-Agent Financial Research System

> Automatically analyzes A-share sector rotation every trading day at 2:30 PM and pushes a structured report to Feishu (Lark). Built with LangGraph multi-agent workflow, powered by real TongHuaSun (THS) official sector indices.

[中文文档](README_CN.md)

---

## What It Does

- **Sector Rotation Analysis**: Covers 22 major industries (semiconductor, food & beverage, banking, new energy, etc.), each mapped to multiple THS official sub-indices for accuracy
- **AI-Generated Report**: DeepSeek LLM analyzes the data and writes a ~400-word strategy report — top gainers, top losers, rotation logic, and personalized suggestions based on your holdings
- **Auto Push to Feishu**: Sends a rich-text card to your Feishu group every weekday at 14:30 CST (via GitHub Actions — no server, no Mac required)
- **Individual Stock Research**: Deep-dive report for any A-share stock (financials + news + K-line chart + self-correction review)
- **Personalized Memory**: Mem0 long-term memory remembers your portfolio and past judgments, making each report progressively more relevant to you

---

## Architecture

```
GitHub Actions (14:30 CST, Mon–Fri)
        ↓
sector_main.py
        ↓
┌─────────────────────────────────────┐
│         LangGraph Workflow          │
│                                     │
│  screener_node                      │
│  → Pull 22 sectors from THS API    │
│  → Read user memory (Mem0)         │
│                                     │
│  researcher_node                    │
│  → THS sector index history        │
│  → Tavily news search              │
│  → DeepSeek analysis per sector    │
│                                     │
│  reporter_node                      │
│  → Compile rotation report         │
│  → Save to Mem0 memory             │
│  → Push Feishu rich-text card      │
└─────────────────────────────────────┘
```

**Tech Stack**

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph + LangChain |
| LLM | DeepSeek V4 (via DMXAPI) |
| Sector Data | AKShare → TongHuaSun Official Indices |
| News Search | Tavily Search API |
| Long-term Memory | Mem0 + Qdrant (local file mode, no Docker) |
| Notification | Feishu Custom Bot Webhook |
| Automation | GitHub Actions (free, cloud-based) |
| PDF Parsing | LlamaParse (optional, for annual reports) |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yunxuanQu999/BIGA-financial-research-agent.git
cd BIGA-financial-research-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required keys:

| Key | Where to Get |
|-----|-------------|
| `DEEPSEEK_API_KEY` | [dmxapi.cn](https://dmxapi.cn) or [platform.deepseek.com](https://platform.deepseek.com) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) — free 1,000 requests/month |
| `FEISHU_WEBHOOK_URL` | Feishu group → Settings → Bots → Add Bot → Custom Bot → Copy URL |

### 3. Run

```bash
# Daily sector rotation analysis
python sector_main.py --period 日

# Weekly
python sector_main.py --period 周

# Individual stock research
python main.py --code 600519.SH --user my_user --name 贵州茅台
```

---

## GitHub Actions Auto-Schedule

The workflow runs automatically every weekday at **14:30 CST** (06:30 UTC) without any server or Mac.

To enable:
1. Fork this repo
2. Go to **Settings → Secrets and variables → Actions**
3. Add these secrets: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `TAVILY_API_KEY`, `FEISHU_WEBHOOK_URL`
4. Go to **Actions** tab → **A股板块轮动日报** → **Run workflow** to test manually

---

## Personalized Memory

Tell the system about your portfolio so it gives you relevant suggestions:

```bash
python -c "
from memory.long_term import remember_user_preference
remember_user_preference('my_user', '我持有半导体ETF，关注新能源板块，风险偏好中等')
"
```

From the next run, the report will include a section tailored to your holdings.

---

## Project Structure

```
├── sector_main.py          # Entry: sector rotation analysis
├── main.py                 # Entry: individual stock research
├── workflow/
│   ├── sector_graph.py     # LangGraph: 3-node sector workflow
│   └── graph.py            # LangGraph: 4-node stock research workflow
├── tools/
│   ├── sector_data.py      # THS sector index data
│   ├── stock_data.py       # AKShare stock price & financials
│   ├── web_search.py       # Tavily news search
│   ├── python_sandbox.py   # K-line chart generation
│   └── feishu_webhook.py   # Feishu push
├── memory/
│   └── long_term.py        # Mem0 + Qdrant memory
├── rag/
│   ├── loader.py           # LlamaParse PDF parsing
│   ├── hybrid_search.py    # Vector + BM25 hybrid retrieval
│   └── hyde.py             # HyDE query enhancement
├── .github/workflows/
│   └── daily_sector.yml    # GitHub Actions schedule
└── .env.example            # Key template
```

---

## Key Engineering Decisions

- **THS Official Indices**: Each broad sector averages multiple THS sub-indices (e.g., "Food & Beverage" = 白酒 + 饮料制造 + 食品加工制造) for more accurate representation
- **macOS Proxy Bypass**: Monkey-patches `requests.Session.send` to force direct connection to Chinese financial domains, bypassing system-level proxies
- **LLM Timeout Guard**: `timeout=90, max_retries=1` prevents the workflow from hanging for hours when the LLM API is unresponsive
- **Self-Correction**: The review agent checks for logical contradictions and triggers a retry loop (max 2 times), with keyword filtering to prevent false positives

---

## License

MIT

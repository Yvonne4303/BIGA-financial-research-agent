from dotenv import load_dotenv
import os

load_dotenv()

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Data
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# Search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# PDF
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Sandbox
E2B_API_KEY = os.getenv("E2B_API_KEY", "")

# Memory
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")

# Feishu
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")

# Vector DB
QDRANT_URL = os.getenv("QDRANT_URL", "")  # 空字符串 = 内存模式，无需 Docker
QDRANT_COLLECTION = "financial_reports"

# Embedding model (本地，无需 API Key)
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")

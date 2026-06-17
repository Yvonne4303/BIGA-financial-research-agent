import os
from mem0 import Memory
from config.settings import MEM0_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, QDRANT_URL
from loguru import logger

# Mem0 的 openai provider 通过环境变量读取自定义 endpoint
os.environ.setdefault("OPENAI_API_KEY", DEEPSEEK_API_KEY)
os.environ.setdefault("OPENAI_BASE_URL", DEEPSEEK_BASE_URL)

_memory = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        qdrant_config = {
            "collection_name": "user_memory",
            "embedding_model_dims": 512,  # BAAI/bge-small-zh-v1.5 维度
        }
        if QDRANT_URL:
            qdrant_config["url"] = QDRANT_URL
        else:
            qdrant_config["path"] = "/tmp/mem0_qdrant"

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": DEEPSEEK_MODEL,
                    "api_key": DEEPSEEK_API_KEY,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_config,
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "BAAI/bge-small-zh-v1.5"},
            },
        }
        _memory = Memory.from_config(config)
    return _memory


def remember_user_preference(user_id: str, content: str) -> str:
    """存储用户偏好（持仓、风险偏好、关注板块等）"""
    try:
        mem = _get_memory()
        result = mem.add(content, user_id=user_id)
        logger.info(f"记忆存储: user={user_id}, content={content[:50]}")
        return f"已记住: {content}"
    except Exception as e:
        logger.error(f"记忆存储失败: {e}")
        return f"记忆存储失败: {str(e)}"


def recall_user_context(user_id: str, query: str) -> str:
    """根据当前问题检索用户相关记忆"""
    try:
        mem = _get_memory()
        results = mem.search(query, user_id=user_id, limit=5)
        if not results or not results.get("results"):
            return "暂无该用户的历史记忆"
        memories = results["results"]
        lines = [f"- {m['memory']}" for m in memories]
        return "用户历史记忆:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"记忆检索失败: {e}")
        return ""


def remember_stock_judgment(user_id: str, ts_code: str, judgment: str) -> str:
    """记录对某只股票的分析判断（实体记忆）"""
    content = f"关于股票 {ts_code}：{judgment}"
    return remember_user_preference(user_id, content)


def recall_stock_history(user_id: str, ts_code: str) -> str:
    """检索对某只股票的历史判断"""
    return recall_user_context(user_id, f"股票 {ts_code} 的历史分析判断")

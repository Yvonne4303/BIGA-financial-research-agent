from llama_parse import LlamaParse
from langchain_core.documents import Document
from config.settings import LLAMA_CLOUD_API_KEY, REPORTS_DIR
from loguru import logger
import os


def parse_pdf_to_documents(pdf_path: str) -> list[Document]:
    """用 LlamaParse 将财报 PDF 解析为结构化 Document 列表（保留表格 Markdown 格式）"""
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        language="ch_sim",          # 简体中文
        parsing_instruction=(
            "这是一份A股上市公司的年报或季报，请完整保留所有财务数据表格，"
            "将表格转换为 Markdown 格式，保持数字精确。"
        ),
    )
    try:
        documents = parser.load_data(pdf_path)
        logger.info(f"解析完成: {pdf_path}，共 {len(documents)} 个文档块")
        # 转换为 LangChain Document
        return [
            Document(
                page_content=doc.text,
                metadata={"source": os.path.basename(pdf_path), "page": i},
            )
            for i, doc in enumerate(documents)
        ]
    except Exception as e:
        logger.error(f"PDF 解析失败 {pdf_path}: {e}")
        return []


def load_all_reports(company_dir: str | None = None) -> list[Document]:
    """加载指定目录下的所有 PDF 财报"""
    target_dir = company_dir or REPORTS_DIR
    all_docs = []
    for fname in os.listdir(target_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(target_dir, fname)
            docs = parse_pdf_to_documents(path)
            all_docs.extend(docs)
    logger.info(f"共加载 {len(all_docs)} 个文档块来自 {target_dir}")
    return all_docs

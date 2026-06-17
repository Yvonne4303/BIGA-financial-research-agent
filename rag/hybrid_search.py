from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config.settings import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL
from loguru import logger
import jieba


def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


class HybridSearchEngine:
    """向量检索 + BM25 稀疏检索混合，结果 RRF 融合排序"""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.qdrant_client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(location=":memory:")
        self.vector_store: QdrantVectorStore | None = None
        self.bm25: BM25Okapi | None = None
        self.all_docs: list[Document] = []

    def build_index(self, documents: list[Document]):
        """首次构建索引（Qdrant + BM25）"""
        logger.info(f"构建索引，共 {len(documents)} 个文档块")

        # Qdrant
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )

        self.vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )

        # BM25
        self.all_docs = documents
        corpus = [_tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info("索引构建完成")

    def load_existing_index(self, documents: list[Document]):
        """加载已有 Qdrant 集合 + 重建 BM25（无需重新向量化）"""
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding=self.embeddings,
        )
        self.all_docs = documents
        corpus = [_tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 6, alpha: float = 0.6) -> list[Document]:
        """
        混合检索。alpha 控制向量权重（1-alpha 为 BM25 权重）。
        使用 RRF（倒数排名融合）合并两路结果。
        """
        if self.vector_store is None or self.bm25 is None:
            raise RuntimeError("索引未初始化，请先调用 build_index 或 load_existing_index")

        # 向量检索
        vec_results = self.vector_store.similarity_search(query, k=k * 2)
        vec_ids = {id(doc): rank for rank, doc in enumerate(vec_results)}

        # BM25 检索
        tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranked = sorted(range(len(self.all_docs)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ids = {id(self.all_docs[i]): rank for rank, i in enumerate(bm25_ranked[: k * 2])}

        # RRF 融合
        rrf_k = 60
        scores: dict[int, float] = {}
        all_candidate_ids = set(vec_ids) | set(bm25_ids)

        for doc_id in all_candidate_ids:
            vec_rank = vec_ids.get(doc_id, k * 2)
            bm25_rank = bm25_ids.get(doc_id, k * 2)
            scores[doc_id] = (
                alpha / (rrf_k + vec_rank) + (1 - alpha) / (rrf_k + bm25_rank)
            )

        # 构建 id→doc 映射
        id_to_doc = {id(doc): doc for doc in vec_results}
        id_to_doc.update({id(self.all_docs[i]): self.all_docs[i] for i in bm25_ranked[: k * 2]})

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        return [id_to_doc[doc_id] for doc_id in sorted_ids if doc_id in id_to_doc]

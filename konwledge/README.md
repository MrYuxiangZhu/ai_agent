# Modular RAG Knowledge Base

这是一个独立、模块化、可扩展的 RAG 知识库系统。默认实现只依赖 Python 标准库，可以直接运行；生产环境可以按接口替换 embedding、向量库、reranker 和 LLM。

## 能力覆盖

- 文档导入：Markdown、TXT、PDF、网页、SQLite 数据库。
- 文档解析与清洗：空白规范化、页眉页脚清理、内容 hash、去重。
- Chunk 切分：标题层级优先，长度切分，overlap 保留上下文。
- Embedding 向量化：本地 HashEmbeddingModel，HTTP EmbeddingModel 接口。
- 向量库入库：本地 InMemoryVectorStore + JSON 持久化，预留外部向量库适配器。
- 检索：向量检索 + BM25 关键词检索 + 元数据过滤。
- Rerank：默认轻量词法 rerank，可替换为 cross-encoder reranker。
- 问答生成：MockLLMClient 和通用 HTTP JSON LLM 客户端。
- 引用溯源：答案携带 citations、source、chunk_id、document_id。
- 评估与优化：Recall@K、citation hit rate、答案长度统计。

## 目录结构

```text
konwledge/
├── app.py
├── factory.py
├── core/
│   ├── config.py
│   ├── interfaces.py
│   └── models.py
├── loaders/
│   ├── db_loader.py
│   ├── file_loaders.py
│   └── web_loader.py
├── processing/
│   ├── chunker.py
│   ├── cleaner.py
│   └── text.py
├── embeddings/
│   └── models.py
├── stores/
│   ├── keyword.py
│   └── vector.py
├── retrieval/
│   ├── hybrid.py
│   └── rerank.py
├── llm/
│   ├── clients.py
│   └── prompting.py
├── pipelines/
│   ├── ingest.py
│   └── qa.py
├── evaluation/
│   └── metrics.py
├── data/
└── examples/
```

## 快速运行

从项目根目录执行：

```bash
python3 -m konwledge.app \
  --data-dir konwledge/data \
  --question "RAG 系统有哪些模块？"
```

运行示例脚本：

```bash
python3 konwledge/examples/run_demo.py
```

## 代码使用

```python
from konwledge.factory import build_local_rag_system

system = build_local_rag_system()
system.ingestion.ingest_directory()

answer = system.qa.ask("如何提高 RAG 检索质量？")
print(answer.answer)
for citation in answer.citations:
    print(citation.source, citation.title)
```

## 替换真实模型

Embedding 可替换为：

```python
from konwledge.embeddings.models import HttpEmbeddingModel

embedding = HttpEmbeddingModel(
    endpoint="http://127.0.0.1:8000/embeddings",
    model="bge-m3",
    dimension=1024,
)
```

LLM 可替换为：

```python
from konwledge.llm.clients import HttpJsonLLMClient

llm = HttpJsonLLMClient(
    endpoint="http://127.0.0.1:8000/generate",
    model="qwen",
    temperature=0,
)
```

向量库可通过继承 `ExternalVectorStoreAdapter` 接入 FAISS、Milvus、Qdrant、Chroma 或 pgvector，只要实现：

```python
add(chunks, vectors)
search(query_vector, top_k, metadata_filter)
persist(path)
load(path)
```

## 生产化建议

默认实现适合学习、原型和单机小规模知识库。生产环境建议增强：

- 使用真实 embedding 模型，例如 bge-m3、text-embedding-3-large。
- 使用 Milvus、Qdrant、Elasticsearch、OpenSearch 或 pgvector。
- 使用 cross-encoder reranker，例如 bge-reranker-v2-m3。
- 为所有 chunk 加入租户、权限、版本、更新时间元数据。
- 建立离线评测集，持续观察 Recall@K、引用准确率和幻觉率。
- 对 ingestion 做异步队列和增量索引。

# RAG 知识库设计框架思路

本文系统梳理一个高质量、高效率、可扩展 RAG 知识库系统的设计思路。RAG 的核心不是简单地“文档切片 + 向量库 + 大模型”，而是一个完整的信息处理、检索、生成和评估系统。

## 1. RAG 的核心目标

RAG，全称 Retrieval-Augmented Generation，即检索增强生成。它的目标是让大语言模型在回答问题前，先从外部知识库中检索相关资料，再基于检索结果生成答案。

一个高质量 RAG 系统需要同时满足三个目标：

```text
高质量：答案准确、引用可靠、尽量减少幻觉。
高效率：检索快、生成快、成本可控。
高全面：知识覆盖完整，能持续更新和扩展。
```

从系统角度看，RAG 可以拆成两条主链路：

```text
离线链路：文档导入 -> 清洗解析 -> Chunk 切分 -> Embedding -> 索引入库

在线链路：用户问题 -> 查询改写 -> 检索 -> Rerank -> 上下文组装 -> LLM 生成 -> 引用溯源
```

## 2. 总体架构

推荐整体架构如下：

```text
                       ┌──────────────────┐
                       │    数据源层       │
                       │ PDF/MD/TXT/Web/DB │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │   文档导入层      │
                       │ Loader/Connector  │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ 文档处理层        │
                       │ 清洗/解析/去重     │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Chunk 切分层      │
                       │ 语义/标题/长度     │
                       └─────────┬────────┘
                                 │
                                 ▼
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        ┌──────────────────┐           ┌──────────────────┐
        │   向量索引层      │           │  关键词索引层     │
        │ Vector Store      │           │ BM25/Inverted     │
        └─────────┬────────┘           └─────────┬────────┘
                  │                              │
                  └───────────────┬──────────────┘
                                  ▼
                       ┌──────────────────┐
                       │   混合检索层      │
                       │ Vector + BM25     │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │     Rerank 层     │
                       │ Cross Encoder 等  │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  上下文组装层     │
                       │ 去重/合并/压缩     │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │    LLM 生成层     │
                       │ Answer + Citation │
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  评估与反馈层     │
                       │ Recall/幻觉/引用   │
                       └──────────────────┘
```

## 3. 离线链路设计

离线链路决定知识库的上限。很多 RAG 效果差，不是因为模型不够强，而是因为文档处理和索引设计不合理。

### 3.1 文档导入

文档导入层负责把不同来源的数据统一转成标准 `Document` 对象。

常见数据源包括：

```text
PDF
Markdown
TXT
HTML / 网页
数据库
API
企业 Wiki
代码仓库
工单系统
聊天记录
```

标准文档结构建议包含：

```python
Document(
    id="doc_xxx",
    text="文档正文",
    source="原始来源路径或 URL",
    title="文档标题",
    metadata={
        "author": "作者",
        "updated_at": "更新时间",
        "department": "所属部门",
        "permission": ["可访问角色"],
        "version": "版本号"
    }
)
```

设计重点：

- Loader 只负责读取和初步转换，不应该承担复杂清洗逻辑。
- 不同数据源应该通过统一接口接入。
- 文档必须保留来源信息，方便后续引用溯源。
- 生产环境必须保留权限、版本和更新时间元数据。

### 3.2 文档解析与清洗

文档清洗的目标是去掉噪声，同时保留语义结构。

常见清洗任务：

```text
去除页眉页脚
去除重复目录
修复异常换行
合并被错误切断的段落
去除 HTML 标签和脚本
提取 PDF 正文
提取表格文本
OCR 图片文字
规范化空白字符
```

清洗后应补充元数据：

```python
metadata = {
    "content_hash": "内容指纹",
    "char_count": 12345,
    "loader": "PDFFileLoader",
    "cleaned_at": "2026-07-26"
}
```

设计重点：

- 清洗不能破坏标题、段落、列表和代码块结构。
- 对 PDF、网页、表格要尽量保留层级关系。
- 对重复文档要通过 hash 或相似度去重。
- 清洗前后的文档最好都可追踪，便于排查问题。

### 3.3 Chunk 切分

Chunk 切分是 RAG 中非常关键的一环。切得太大，召回不准；切得太小，上下文不完整。

推荐切分策略：

```text
优先按标题层级切分
再按段落切分
最后按长度兜底切分
```

例如：

```text
# 报销制度
## 差旅报销
### 交通费用
```

切分后的 chunk 应保留：

```python
Chunk(
    id="chunk_xxx",
    document_id="doc_xxx",
    text="chunk 内容",
    source="原始文件路径",
    title="交通费用",
    section_path=["报销制度", "差旅报销", "交通费用"],
    start_char=100,
    end_char=900,
    metadata={...}
)
```

推荐参数：

```text
中文 chunk：500-1200 字
英文 chunk：300-800 tokens
overlap：10%-20%
最小 chunk：100-200 字
```

设计重点：

- Chunk 要尽量语义完整。
- 每个 chunk 都要能独立回答一个小问题。
- 保留 section path，方便展示引用。
- 可以采用 parent-child chunk：小 chunk 负责召回，大 chunk 负责生成。

### 3.4 Embedding 向量化

Embedding 层负责把文本映射为向量：

```text
text -> vector
```

向量表示用于语义检索。例如：

```text
“怎么申请出差费用”
```

可以召回：

```text
“差旅报销流程”
```

生产环境常用模型：

```text
bge-m3
bge-large-zh
text-embedding-3-large
e5
jina-embeddings
```

设计重点：

- Embedding 模型必须和业务语言匹配。
- 中文知识库优先使用中文或多语言 embedding。
- 向量维度要和向量库 schema 一致。
- 批量 embedding 可以显著提高效率。
- 文档更新时只需要增量向量化变化部分。

### 3.5 索引入库

RAG 通常需要两类索引：

```text
向量索引：用于语义相似检索。
关键词索引：用于精确词、编号、术语、错误码检索。
```

向量库可以选择：

```text
FAISS
Milvus
Qdrant
Chroma
pgvector
Weaviate
```

关键词索引可以选择：

```text
BM25
Elasticsearch
OpenSearch
Lucene
```

设计重点：

- 向量库中要存 chunk id、vector 和 metadata。
- 原文最好存文档库或对象存储，向量库只存必要字段。
- 需要支持 metadata filter，例如部门、权限、时间、标签。
- 大规模场景要支持增量更新和删除。

## 4. 在线链路设计

在线链路决定用户实际体验。重点是查得准、排得好、答得稳。

### 4.1 用户问题理解

用户问题通常不完整、不规范，甚至依赖上下文。

例如：

```text
这个怎么申请？
```

如果上一轮对话在说“差旅报销”，系统应改写为：

```text
差旅报销怎么申请？
```

常见 Query 处理方式：

```text
Query Rewrite：问题改写
Query Expansion：扩展同义词
Multi Query：生成多个检索问题
HyDE：先生成假设答案，再用假设答案检索
Metadata Routing：根据问题选择知识域
```

设计重点：

- 简单问题不要过度改写。
- 多轮对话必须结合历史上下文。
- 改写后的 query 要保留原始用户意图。

### 4.2 混合检索

只用向量检索通常不够，因为向量检索擅长语义相似，但不擅长精确匹配。

推荐使用混合检索：

```text
Dense Vector Search + BM25 Keyword Search
```

融合得分可以表示为：

```text
score = α * vector_score + β * keyword_score + γ * metadata_score
```

其中：

```text
vector_score：语义相似度
keyword_score：关键词匹配分数
metadata_score：时间、权限、标签等加权
```

设计重点：

- 第一阶段召回要适当放宽，例如 top 30-100。
- 向量和 BM25 分数需要归一化后再融合。
- 精确术语、代码、错误码、政策编号应该提高 BM25 权重。
- 语义问答、概念解释可以提高向量权重。

### 4.3 Metadata Filter

元数据过滤是企业 RAG 的必需能力。

常见过滤条件：

```text
tenant_id
department
permission
doc_type
updated_at
language
product
version
```

例如：

```python
metadata_filter = {
    "department": "finance",
    "permission": "employee"
}
```

设计重点：

- 权限过滤必须发生在检索阶段或检索前。
- 不能先检索出无权限内容，再靠 LLM 不展示。
- 多租户系统必须做 tenant 隔离。

### 4.4 Rerank

第一阶段检索关注召回率，Rerank 关注排序准确性。

典型流程：

```text
Hybrid Retrieve top 50
-> Rerank top 10
-> Context Builder
```

Reranker 可以使用：

```text
bge-reranker
jina-reranker
Cohere Rerank
Cross Encoder
LLM Rerank
```

设计重点：

- Rerank 成本比普通检索高，不应对全量文档做。
- 一般只对 top 30-100 候选做 rerank。
- 高质量 RAG 中，rerank 往往比换更大的 LLM 更有效。

### 4.5 上下文组装

检索出的 chunk 不能直接全部塞给 LLM，需要做上下文整理。

上下文组装要做：

```text
去重相似 chunk
合并相邻 chunk
保留标题路径
保留来源 URL 或文件路径
按相关性和文档结构排序
控制 token budget
```

推荐上下文格式：

```text
[资料 1]
标题：报销制度 / 差旅报销 / 交通费用
来源：finance/travel.md
内容：
...

[资料 2]
标题：报销制度 / 审批流程
来源：finance/travel.md
内容：
...
```

设计重点：

- 给 LLM 的资料必须带编号，便于引用。
- 相同来源的连续 chunk 可以合并。
- 低相关 chunk 宁可丢弃，不要污染上下文。
- 上下文太长会增加成本，也可能降低回答质量。

### 4.6 LLM 生成

生成层负责基于上下文回答问题。

Prompt 应明确约束：

```text
只能基于给定资料回答。
如果资料不足，说明无法确认。
不要编造资料中没有的数字、日期、政策。
回答中必须标注引用来源。
```

推荐 Prompt 结构：

```text
角色说明
回答规则
用户问题
检索资料
输出要求
```

设计重点：

- 对事实类问题要求引用。
- 对资料不足的问题要允许拒答。
- 对多来源冲突要提示冲突，而不是强行合并。
- 对长答案可以要求分点输出。

### 4.7 引用溯源

引用溯源是高质量 RAG 的核心能力。

每条引用应包含：

```text
citation index
chunk_id
document_id
title
source
snippet
metadata
```

用户看到答案时，应该能追溯到原始文档。

设计重点：

- 引用必须对应真实检索资料。
- 引用片段应该能支持答案中的关键事实。
- 不要让 LLM 自己编造 citation。
- Citation 最好由系统侧根据检索结果生成。

## 5. 评估体系设计

没有评估体系，RAG 很难持续优化。

### 5.1 离线评估集

建议构建一批标准测试用例：

```python
EvaluationCase(
    question="差旅报销需要哪些材料？",
    expected_answer="需要审批单、发票、行程单等。",
    expected_document_ids=["doc_finance_travel"],
    expected_chunk_ids=["chunk_xxx"],
    tags=["finance", "policy"]
)
```

测试集要覆盖：

```text
高频问题
边界问题
容易混淆的问题
需要拒答的问题
多文档综合问题
权限隔离问题
```

### 5.2 检索指标

常用检索指标：

```text
Recall@K：正确文档是否出现在 top K。
MRR：正确结果排名是否靠前。
NDCG：排序整体质量。
Hit Rate：是否命中目标文档。
```

其中 Recall@K 最重要，因为如果第一阶段没有召回正确资料，后面的 LLM 基本无法答对。

### 5.3 生成指标

常用生成指标：

```text
Answer Correctness：答案是否正确。
Faithfulness：答案是否被上下文支持。
Citation Accuracy：引用是否真的支持答案。
Hallucination Rate：幻觉率。
Refusal Accuracy：资料不足时是否正确拒答。
```

设计重点：

- 不要只看答案是否流畅。
- 要重点检查答案是否被检索资料支持。
- 引用准确率比普通文本相似度更关键。

## 6. 效率与成本设计

RAG 的效率瓶颈主要来自：

```text
文档解析
Embedding
向量检索
Rerank
LLM 生成
```

优化策略：

```text
Embedding 批处理
增量索引
检索缓存
热门问题缓存
Rerank 候选数控制
上下文压缩
小模型处理简单问题
大模型处理复杂问题
异步文档导入
```

推荐在线链路延迟目标：

```text
检索：50-300ms
Rerank：100-800ms
首 token：500-2000ms
整体 P95：3-8s
```

设计重点：

- 不要为了召回更多资料无限增加 top_k。
- Rerank 数量要受控。
- Prompt token 是主要成本之一。
- 热门问题可以缓存最终答案，但要注意文档更新后的失效机制。

## 7. 安全与权限设计

企业级 RAG 必须考虑安全。

关键设计：

```text
租户隔离
用户权限过滤
敏感信息脱敏
访问审计
Prompt 注入防护
数据源权限继承
日志脱敏
```

特别注意：

```text
不能把用户无权限访问的 chunk 放入 prompt。
不能依赖 LLM 自觉不泄露敏感信息。
不能把密钥、Token、隐私数据写入日志。
```

权限过滤应该尽量前置：

```text
用户身份 -> 权限集合 -> metadata filter -> 检索候选
```

## 8. 模块化接口设计

一个可扩展 RAG 系统应该把模块拆开，而不是写成一个大函数。

推荐接口：

```text
DocumentLoader：加载数据源
DocumentCleaner：清洗文档
Chunker：切分文档
EmbeddingModel：文本向量化
VectorStore：向量索引
KeywordIndex：关键词索引
Retriever：混合检索
Reranker：重排序
ContextBuilder：上下文组装
LLMClient：生成答案
Evaluator：评估系统
```

这样做的好处：

```text
可以替换具体模型
可以替换向量库
可以独立测试每个模块
可以逐步从原型升级到生产
可以复用离线索引和在线检索逻辑
```

## 9. 当前项目中的实现映射

当前 `konwledge/` 目录已经按上述思路做了模块化拆分：

```text
konwledge/core/
  数据模型、配置、接口协议

konwledge/loaders/
  文件、网页、数据库 loader

konwledge/processing/
  文本清洗、去重、chunk 切分、文本工具

konwledge/embeddings/
  本地 hash embedding 和 HTTP embedding 客户端

konwledge/stores/
  本地向量库、BM25 索引、外部向量库适配接口

konwledge/retrieval/
  混合检索和 rerank

konwledge/llm/
  Prompt 构造、上下文组装、LLM 客户端

konwledge/pipelines/
  ingestion pipeline 和 QA pipeline

konwledge/evaluation/
  离线评估指标
```

默认本地链路为：

```text
DirectoryLoader
-> BasicDocumentCleaner
-> DocumentDeduplicator
-> HierarchicalChunker
-> HashEmbeddingModel
-> InMemoryVectorStore + BM25KeywordIndex
-> HybridRetriever
-> LexicalReranker
-> ContextBuilder
-> MockLLMClient
-> RAGAnswer
```

## 10. 从原型到生产的演进路线

### 阶段一：本地可运行原型

目标是跑通完整链路。

```text
Markdown/TXT 文档
本地 hash embedding 或轻量 embedding
本地向量库
BM25
Mock 或本地 LLM
基础引用
```

### 阶段二：真实语义检索

目标是提升召回质量。

```text
接入 bge-m3 / text-embedding-3-large
接入真实向量库
加入 reranker
建立评估集
```

### 阶段三：企业知识库

目标是支持真实业务。

```text
多数据源同步
权限过滤
增量索引
版本管理
引用打开原文
日志与监控
```

### 阶段四：高质量生产系统

目标是稳定、可评估、可运营。

```text
自动评估
人工反馈闭环
检索参数自动调优
Prompt 版本管理
灰度发布
成本监控
安全审计
```

## 11. 常见问题与设计取舍

### 11.1 为什么不能只用向量检索

向量检索适合语义相似，但对这些内容不稳定：

```text
错误码
人名
产品型号
政策编号
精确日期
函数名
数据库字段
```

因此需要 BM25 或关键词索引补充。

### 11.2 为什么需要 Rerank

第一阶段检索通常只做粗召回，结果中会有噪声。Rerank 可以更准确判断 query 和 chunk 是否真正相关。

好的排序通常比单纯增加 top_k 更有效。

### 11.3 为什么要保留引用

RAG 不是只要答案看起来对，而是要能证明答案来自哪里。

引用可以帮助：

```text
用户验证答案
排查错误召回
评估幻觉
支持审计
建立信任
```

### 11.4 为什么需要评估集

没有评估集，优化只能靠感觉。改了 chunk 大小、embedding 模型或 rerank 策略后，必须用固定测试集比较效果。

## 12. 总结

高质量 RAG 的关键不是某一个组件，而是整条链路的系统工程：

```text
干净的数据
合理的 chunk
准确的 embedding
混合检索
强 rerank
可控 prompt
可靠引用
持续评估
权限治理
增量更新
```

一句话总结：

```text
RAG = 可评估的检索系统 + 可控的生成系统 + 可追溯的知识治理。
```

向量库只是 RAG 的一部分。真正决定效果的，是文档质量、检索策略、上下文组织、引用机制和持续评估闭环。

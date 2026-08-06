# Milvus Bootcamp 自学大纲

> 基于 [milvus-io/bootcamp](https://github.com/milvus-io/bootcamp) 仓库整理。
> 主线使用 `bootcamp/` 目录（速查表 + 教程 notebook），再以 `tutorials/quickstart/`、`evaluation/`、`applications/` 补齐深度与实战。

## 总目标

学完后你应该能独立完成：

- 讲清楚向量、Embedding、向量检索、向量数据库的基本原理
- 用 Milvus / Zilliz 完成数据接入（分块 → Embedding → 建 Collection → 建索引 → 插入 → 检索）
- 搭建一个可评估、可优化的 RAG 应用
- 根据业务场景选择合适的索引、距离度量、一致性级别和部署形态

## 环境准备（阶段 0 完成时配好）

二选一：

1. **本地**：Docker 安装 Milvus Standalone，配 Attu 可视化界面
2. **云端**：Zilliz Cloud 免费 Serverless 实例（免运维，适合快速上手）

语言环境：Python 3.9+、`pymilvus`、Jupyter Notebook / Jupyter Lab。

---

## 阶段 0：预备知识与环境搭建（约 0.5 天）

**目标**：建立向量检索的直觉，跑通第一个 Milvus 程序。

**材料**

- 阅读 [Vector Database 101](https://zilliz.com/learn/what-is-vector-database)：向量、Embedding、向量检索、向量数据库是什么
- 安装 Milvus Standalone 或注册 Zilliz 免费实例
```powershell
curl -o docker-compose.yml https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
docker compose up -d
docker compose ps
docker run -d --name attu -p 3000:3000 -e MILVUS_ADDRESS=host.docker.internal:19530 -v attu-data:/data zilliz/attu:v3.0.0-beta.6
```
- 跑 `tutorials/quickstart/quickstart.ipynb`（hello milvus）

**产出**：本地能完成"创建 Collection → 插入向量 → 搜索"的最小闭环。

---

## 阶段 1：通读速查表，掌握核心概念（1 天）

**目标**：建立 Milvus 的完整概念框架，后续每个 notebook 都能对号入座。

**材料**

- 精读 `bootcamp/MilvusCheatSheet.md`，重点关注：
  - 数据接入最佳实践：分块策略、1 个 Collection 用 1 个 Embedding 模型
  - Collection / Schema / Partition（手动分区 vs 自动分区）
  - 索引：HNSW、IVF、DiskANN、AUTOINDEX 及参数（M、efConstruction）
  - 距离度量：COSINE / IP / L2 怎么选
  - 一致性级别：Strong / Eventually / Session / Bounded
  - 四大操作：Insert、Search、Upsert、Query 的区别
- 对照官方文档 [milvus.io/docs](https://milvus.io/docs) 查细节

**产出**：写一页概念笔记，画出 `数据 → 分块 → Embedding → Collection → Schema → Index → Search` 流程图。

---

## 阶段 2：连接与基础检索（1–2 天）

**目标**：掌握标准检索流水线。

**材料**

- `bootcamp/milvus_connect.ipynb`：本地与云端连接方式、Token 管理
- `bootcamp/Retrieval/imdb_milvus_client.ipynb`：主线教程 —— 选 Embedding 模型、建 Collection、配 HNSW 索引、批量加载数据、语义搜索
- `bootcamp/Retrieval/imdb_metadata_filtering.ipynb`、`imdb_metadata_json.ipynb`：标量元数据过滤（布尔表达式、JSON）
- 阅读 [如何选择向量索引](https://zilliz.com/learn/choosing-right-vector-index-for-your-project)

**产出**：把 IMDB 示例换成自己的小数据集，完成一个带元数据过滤的语义搜索 notebook。

---

## 阶段 3：更多检索形态（1–2 天）

**目标**：理解"多模态输入 → 同一向量空间"的通用范式。

**材料**

- `bootcamp/Retrieval/paintings.ipynb`：图像向量检索
- `bootcamp/Retrieval/white_house_speeches.ipynb`：长文本/演讲检索
- `bootcamp/model/embedding_functions.ipynb`：Embedding 函数的封装与调用
- `bootcamp/model/reranker.ipynb`：Reranker（重排）的基本用法
- `bootcamp/Integration/openai_embedding.ipynb` 与 `bge_m3_embedding.ipynb`：闭源 vs 开源 Embedding 模型对比

**产出**：写一篇"Embedding 模型选型"小结（开源/闭源、维度、成本、检索效果）。

---

## 阶段 4：构建 RAG（2–3 天）

**目标**：做出第一个带来源引用的 RAG 应用。

**材料**（按顺序跑）

- `bootcamp/RAG/readthedocs_zilliz_langchain.ipynb`：主线 RAG —— LangChain + Milvus，开源（HuggingFace）与闭源（OpenAI）LLM 都覆盖
- `bootcamp/RAG/readthedocs_openai_emb3.ipynb`：OpenAI Embedding 版本
- `bootcamp/RAG/multi_doc_qa_llamaindex.ipynb`：LlamaIndex 多文档问答
- `bootcamp/RAG/conv_mem_langchain.ipynb`：对话记忆（多轮对话）
- `bootcamp/RAG/sparse_dense_embeddings_tutorial.ipynb`：稀疏 + 稠密 Embedding（为混合检索打基础）

**产出**：换成自己的文档库，搭一个带引用来源的聊天机器人。

---

## 阶段 5：RAG 评估（1–2 天）

**目标**：能量化 RAG 效果，而不是"感觉还不错"。

**材料**

- `bootcamp/Evaluation/eval_ragas.ipynb`：用 Ragas 评估
- `evaluation/evaluate_fiqa_customized_RAG.ipynb`：基于 FiQA 数据集的自定义 RAG 评估
- 指标：Faithfulness、Answer Relevancy、Context Precision / Recall

**产出**：对阶段 4 的 RAG 跑一次评估，得到一组基线指标，为阶段 6 的优化做对照。

---

## 阶段 6：高级 RAG 优化（2–3 天）

**目标**：掌握主流 RAG 优化手段，并验证对指标的影响。

**材料**：`bootcamp/RAG/advanced_rag/`（先读该目录 README，再按顺序跑）

- 基线：`vanilla_rag_with_langchain` —— 先建基线
- 查询增强：HyDE、Sub-Query、Query Routing
- 索引/检索增强：Sentence Window、Hybrid Search + Rerank（BM25/稀疏 + 稠密 + RRF/Cross-Encoder）
- 图与智能体：LangChain GraphRAG、LangGraph RAG Agent

**产出**：在阶段 5 的基线上应用 2–3 种优化，记录指标前后对比。

---

## 阶段 7：Agent（1–2 天）

**目标**：把 RAG 升级为可调用工具的智能体。

**材料**

- `bootcamp/OpenAIAssistants/milvus_agent_llamaindex.ipynb`：LlamaIndex + OpenAI Assistant
- `bootcamp/OpenAIAssistants/custom_RAG_workflow.ipynb`：自定义 RAG 工作流

**产出**：做一个能自主检索并回答的问答 Agent。

---

## 阶段 8：官方 Quickstart 教程系统性补全（3–5 天，按需挑选）

**目标**：补齐各应用场景的 Milvus 能力组合。

**材料**：`tutorials/quickstart/`

- `full_text_search_with_milvus.ipynb`：全文检索
- `hybrid_search_with_milvus.ipynb`：混合检索（稠密 + 稀疏 + RRF）
- `image_search_with_milvus.ipynb` / `text_image_search_with_milvus.ipynb`：图文检索
- `multimodal_rag_with_milvus.ipynb`：多模态 RAG
- `graph_rag_with_milvus.ipynb`：图 RAG
- `movie_recommendation_with_milvus.ipynb`：推荐系统
- `contextual_retrieval_with_milvus.ipynb`、`vector_visualization.ipynb`、`hdbscan_clustering_with_milvus.ipynb`：进阶玩法
- `funnel_search_with_matryoshka.ipynb`、`generating_milvus_query_filter.ipynb`：Matryoshka 检索、LLM 生成查询过滤条件

**产出**：能说出每种场景适合哪些 Milvus 能力（向量 / 全文 / 混合 / 多向量）的组合。

---

## 阶段 9：端到端项目实战（3–5 天）

**目标**：把 notebook 变成可部署的产品。

**材料**

- `tutorials/quickstart/apps/`：可运行 Demo（Image Search、RAG Search、Hybrid Demo、Multimodal RAG、CIR 等，含前端）
- `applications/`：完整应用（图像检索、NLP 问答/推荐等，Docker 一键部署）

**产出**：挑一个方向，本地部署一个带界面/API 的完整应用，并换成自己的数据。

---

## 阶段 10：生态集成与生产化（按需）

**目标**：了解 Milvus 生态与生产要点。

**材料**

- `integration/`：LangChain、LlamaIndex、Haystack、DSPy、Ollama、Gemini、DeepSeek 等 50+ 集成示例
- `bootcamp/spark_milvus_demo/`：Spark 大规模数据写入/导入
- 性能与基准：[VectorDBBench](https://github.com/zilliztech/VectorDBBench)
- 官方文档：部署形态（Standalone / Cluster / Cloud）、监控、RBAC、数据一致性机制

**产出**：能写出一个"我的场景该用哪种部署 + 索引 + 数据规模"的选型说明。

---

## 时间规划

| 路线 | 建议周期 | 覆盖阶段 |
| --- | --- | --- |
| 全职速成 | 3–4 周 | 0–9，阶段 10 按需 |
| 业余学习 | 2–3 个月 | 0–9，阶段 10 按需 |
| 最低必学 | 1–1.5 周（全职） | 0–5 必做，6–7 至少各挑一个 notebook |

## 自学建议

- 每个阶段都要有**产出物**（笔记、demo、评估报告），避免只读不练。
- 优先使用自己的数据替换示例数据，效果更直观。
- 遇到 API 差异，以 [pymilvus 官方文档](https://milvus.io/docs) 为准；notebook 依赖版本较旧时可参考但不必死磕。
- 卡住时可用 [DeepWiki 的 bootcamp 结构化索引](https://deepwiki.com/milvus-io/bootcamp) 快速定位知识点。

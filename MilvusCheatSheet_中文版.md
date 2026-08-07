# Milvus 简介、最佳实践与速查表教程

- 目录
  - [Milvus 简介](#milvus-introduction)
    - [快速上手](#quick-start)
    - [架构](#architecture)
    - [文档与版本](#documentation--releases)
  - [Milvus 入门教程](#getting-started-with-milvus-tutorial)
    - [连接 Milvus](#start-up-milvus-server)
    - [分块与嵌入](#chunking--embedding)
    - [集合、Schema 与分区](#collections-schema--partitions)
    - [构建索引](#index)
    - [距离度量](#metric)
    - [一致性](#consistency)
    - [插入数据](#insert-data)
    - [搜索](#search)
    - [Upsert（更新插入）](#upsert)
    - [查询](#query)
  - [示例 Notebook](#example-notebooks)
  - [学习资源](#learning-resources)
  - [社区与帮助](#community--help)

<div>
<img src="../pics/milvus_zilliz_overview.png" width="90%"/>
</div>

## Milvus 简介 {#milvus-introduction}

🐦 [Milvus](https://zilliz.com/what-is-milvus) 是一个开源的（Apache License 2.0）向量数据库。它是一个强大的工具，用于将深度神经网络及其他机器学习（ML）模型生成的嵌入向量（embedding vector）形式的[非结构化数据](https://zilliz.com/learn/introduction-to-unstructured-data)进行存储、索引和管理。非结构化数据包括网页、文本文件、PDF、视频、图像或音频文件。

✴ [Zilliz Cloud](https://zilliz.com/) 是 Milvus 的商业化托管服务。

1. **成功的 AI 应用需要充分利用数据。** 嵌入模型（embedding model）是理解和检索非结构化数据的最先进工具。

2. **非结构化数据被嵌入为向量，Milvus 是为向量数据专门构建的数据库。** 🤖 AI 术语称之为[*向量数据库*](https://www.infoworld.com/article/3711281/how-to-evaluate-a-vector-database.html)。

3. **向量检索是许多 AI 应用的核心，例如检索增强生成（Retrieval Augmented Generation，RAG）。**

**集成**方面，Zilliz 支持 AWS、GCP 和 Azure 云。[Milvus](https://github.com/milvus-io/milvus) 的理念是成为 AI 技术栈中底层的"铲子"。🦙✨𑗗🤗 你应该能够自主选择自己想要的嵌入、融合、LLM 或生成模型。🦜⛓️ Milvus 也不绑定任何 RAG 框架，例如 LlamaIndex 或 LangChain 都可以。

> AI 领域的模型和工具变化非常快！⬱ 作为一家向量数据库公司，我们会有自己的观点，但你完全可以自由地为你的用例选择最新、最好的 AI 工具。

### 快速上手 {#quick-start}

💡[Zilliz Pipelines](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/zilliz_pipeline_rag.ipynb) 是快速体验 Milvus 的一种方式。它也已集成到 [LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/managed/zcpDemo.html) 中。它内置了：

- 开源嵌入模型 *[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)* 和 *[bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)*
- 基于我们在文档解析和分块策略方面的研究，开箱即用地提供良好的检索质量
- [AUTOINDEX](https://docs.zilliz.com/docs/autoindex-explained)，Zilliz 专有功能
- 元数据过滤能力

### 架构 {#architecture}

Milvus 采用共享存储的[架构](https://milvus.io/docs/architecture_overview.md)，包含 4 个彼此独立、可分别扩展或进行灾难恢复的层级：1）接入层，2）协调服务，3）工作节点，4）存储。Milvus 还包含数据分片（data sharding）、日志即数据（logs-as-data）持久化，以及流式数据接入。

<div>
<img src="../pics/oss_zilliz_architecture.png" width="90%"/>
</div>

### 文档与版本 {#documentation--releases}

- 开源 [Milvus 文档](https://milvus.io/docs)
- 开源 [Milvus Client 文档](https://pymilvus.readthedocs.io/en/latest/_modules/milvus/client/stub.html)（围绕 Milvus collection 的无 Schema 封装）
- 商业版 [Zilliz 文档](https://docs.zilliz.com/docs/quick-start)
- [Zilliz 版本发布说明](https://docs.zilliz.com/docs/release-notes-230)
- [Zilliz serverless 免费层](https://zilliz.com/zilliz-cloud-free-tier)
  - 最多 1 个集群
  - 每个集群最多 2 个集合（collection）
  - 每个集合最多 100 万个向量
  - 如需升级到 beta，可联系 Zilliz 技术支持，他们需要你的集群 ID
- 面向生产场景的 [Zilliz 企业版](https://zilliz.com/pricing)：
  - 99.9% 可用性
  - 多可用区
  - 传输和静态数据的企业级加密
  - [SOC 2 Type 2 合规](https://zilliz.com/security)
  - 组织级和项目级的 RBAC（基于角色的访问控制）
  - 资源监控和告警通知
  - 可自助升级到 beta，点击"try beta"按钮即可升级集群
  - 7×24 全年无休的邮件和 Discord 支持，并带有响应时间 SLA（紧急：1 小时；高：4 小时；普通：1 个工作日）
  - Zilliz 自带云（bring-your-own-cloud）计划于 2024 年第二季度推出

<br>

## Milvus 入门教程 {#getting-started-with-milvus-tutorial}

以下是将数据接入 Milvus 以便开发 AI 应用的最佳实践。

1. **启动 Milvus 服务并[连接](https://milvus.io/docs/manage_connection.md)。** <a class="anchor" id="start-up-milvus-server"></a>

- 💡👉🏼 最简单的方式是使用 [Zilliz serverless 免费层](https://zilliz.com/zilliz-cloud-free-tier)。无需担心连接时是否需要足够的 *.wait()*，它始终可用！
- Milvus 可以在本地运行。Milvus 的形态包括 lite、docker 或 k8s。
- Zilliz 在云端运行 Milvus。Zilliz 的形态包括免费层（serverless）或付费（托管在 aws、google、azure 上）。
- 参见本页底部的[示例连接 notebook](#example-notebooks)。 <br><br>

<a class="anchor" id="chunking--embedding"></a>
2. **根据数据类型选择分块策略。** 非结构化数据需要先分块（chunking），再嵌入（embedding，转换为向量），并将向量作为张量（tensor）存储——张量是与特定计算硬件（CPU、GPU、TPU 等）绑定的向量。张量是 AI 领域的通用语言。

- 关于分块策略的[背景知识](https://learn.deeplearning.ai/building-evaluating-advanced-rag)。 <br>
- 大多数通用 NLP 任务，使用 512 的块大小和 10-15% 的重叠效果最佳。
- **网页数据**最适合在分块时把标题（header）加入块中。由于标题很短，通常值得为每个块增加这些上下文。
  - 在 LangChain 中，参见 [ParentDocumentRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever) 和 HTMLHeaderTextSplitter。
  - 在 LlamaIndex 中，参见 [AutoMergingRetriever](https://docs.llamaindex.ai/en/latest/examples/retrievers/auto_merging_retriever.html) 和 HierarchicalNodeParser。<br><br>

3. **每个集合只使用 1 个嵌入模型。** 集合的向量空间通常来自深度神经网络模型的倒数第二层隐藏层。这一层的权重（数字）被用作转换函数，将输入的非结构化数据映射为数字向量（通常是 1024 维）。为了让向量相似度计算有效，所有数据（包括问题）都需要在同一向量空间中做 token 化（将输入映射为输出）。这样，该空间中的概念才能被搜索。因此，最佳实践是每个集合只使用 1 个嵌入模型。

- 💡👉🏼**开源嵌入模型的表现与商业嵌入模型相当。** 开源模型的好处是高召回率，并且可以自由使用自己的数据。例如，查看 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard)，按"Retrieval Average"列降序排序，注意 *[UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)* 排名第 4，且仅占用 1.34 MB 内存；相比之下，OpenAI 的 *ada-002* 排名第 25。（访问时间：2023 年 12 月 30 日。）
- 使用自己的数据和任务微调嵌入模型，检索效果可能提升 10-15%。只有开源嵌入模型可以被微调。<br><br>

<a class="anchor" id="collections-schema--partitions"></a>
4. **[创建](https://milvus.io/docs/create_collection.md)集合（collection）。** 集合类似于数据库表。每个集合都有名称、索引、Schema 和一致性级别。

- 💡👉🏼 最简单的方法是使用 [Milvus Client 无 Schema 模式](https://milvus.io/docs/using_milvusclient.md)。Milvus Client 是 Milvus 集合对象的封装，它使用灵活的 json *key:value* 格式，可以在无需预先定义 Schema 的情况下创建集合。这是入门时最不容易出错的方式。参见本页底部的[示例搜索 notebook](#example-notebooks)。

<div font size="1">

```python
from pymilvus import MilvusClient

COLLECTION_NAME = "MilvusDocs"
EMBEDDING_LENGTH = 1024

INDEX_PARAMS = dict({
    'M': 16,    # 每个节点在图中连接的最大边数
    "efConstruction": M * 2 })  # 构建索引时的动态候选列表大小
index_params = {
    "index_type": "HNSW",   # 基于图的近似最近邻索
    "metric_type": "COSINE",
    "params": INDEX_PARAMS
    }

# 使用无 Schema 的 Milvus Client，采用灵活的 json key:value 格式。
mc = MilvusClient(
    uri=CLUSTER_ENDPOINT,   # uri="http://localhost:19530"
    # API key 或冒号分隔的集群用户名和密码
    token=TOKEN)    # token="root:Milvus"

# 检查集合是否已存在，如果存在则删除。
has = utility.has_collection(COLLECTION_NAME)
if has:
    drop_result = utility.drop_collection(COLLECTION_NAME)

# 创建集合。
mc.create_collection(COLLECTION_NAME,
                     EMBEDDING_LENGTH,
                     consistency_level="Eventually",    # 最终一致性，读操作可能读到旧数据
                     auto_id=True,
                     overwrite=True,    # 当集合已存在时，直接删除并重新创建该集合
                     # 如果使用 AUTOINDEX，可跳过下面的参数
                     params=index_params    # 在创建集合的同时定义向量索引的参数
                    )
print(mc.describe_collection(COLLECTION_NAME))
```
</div>

- 元数据限制：每行最多 64 个字段。这是除了 "pk" 和 "vector" 之外的额外字段数。

- 如果预先定义 Schema，请查阅 [Schema 类型](https://milvus.io/docs/schema.md) 文档。
  - 主键（通常叫 "pk"），默认类型 INT64（注意：LangChain 期望 "pk" 是字符串类型。）
  - 嵌入向量（通常叫 "vector"），类型为 `numpy.ndarray` 的 `numpy.float32` 数字列表
  - 字符串，类型 VARCHAR，最大长度 65535 个字符。最佳实践：在 Schema 中直接使用最大长度。实际数据不会占用那么多空间。

```python
EMBEDDING_LENGTH = 1024
MAX_LENGTH = 65535
fields = [
  FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
  FieldSchema("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_LENGTH),
  FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=MAX_LENGTH),
]
```

- **分区（partition）** 用于将实体隔离到不同的物理路径，以缩小搜索范围。

- **Milvus 支持两种分区类型，两者速度一样快！所以选择权在你！** <br><br>a) 手动分区（MANUAL）——仅当你能保证每个分区大约有 20-100K 行时使用。由用户指定每个实体属于哪个分区。分区可以随时添加或删除。搜索时需要把分区名作为搜索参数传入。<br><br>b) 自动分区（AUTOMATIC）——Milvus 自动将实体分配到不同的分区。搜索时无需指定分区名，Milvus 会自动把你的元数据过滤表达式翻译成分区映射来查找数据。

- 分区技巧：
  - 💡👉🏼 最佳实践是让 Milvus 自动分区数据，并把元数据过滤器转换为搜索映射。
  - 目前 RBAC 只在集合或项目级别生效，因此无法控制不同用户对分区的可见性。
  - 手动分区时，建议每个分区 20-100K 行，否则搜索速度会比自动分区慢。
  - 一个集合中最多可以有 4096 个分区。<br><br>

<a class="anchor" id="index"></a>
5. **[构建索引](https://milvus.io/docs/build_index.md)（即在张量中查找最近邻的搜索算法）。** 数据按照特定的[搜索算法索引](https://milvus.io/docs/index.md)（哈希、树或图）保存在数据结构中。<br>

- 博客：[为你的项目选择合适的索引](https://zilliz.com/learn/choosing-right-vector-index-for-your-project)。

- 💡👉🏼**使用 Milvus Client 时，你需要自己定义 [HNSW 索引](https://github.com/milvus-io/knowhere/blob/main/src/index/hnsw/hnsw.cc)。** 否则搜索可能会很慢（Milvus Client 默认使用 [IVF_Flat](https://milvus.io/docs/index.md) 索引）。

- [HNSW 最佳实践参数](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md)：从 M: 4~64 开始，数据量越大、嵌入维度越高，M 越大。然后设置 ef = efConstruction = M * 2。

- **进阶提示：使用 [AUTOINDEX](https://docs.zilliz.com/docs/autoindex-explained)，除非你在用 Milvus Client。** AUTOINDEX 在 Milvus 中默认为 HNSW。在 Zilliz 中，AUTOINDEX 会根据你的数据和集群上运行的计算类型自动选择最佳索引。<br><br>

<a class="anchor" id="metric"></a>
6. **选择[距离度量](https://milvus.io/docs/metric.md)。**

- 💡👉🏼 "COSINE"（余弦相似度）适用于大多数用例。

- 大多数搜索算法在嵌入数据归一化后效果最佳。这意味着 L2 度量没有意义（因为所有向量长度相同）。**当向量归一化后，"IP"（内积）和"COSINE"是等价的。**

- 只有在你打算保持嵌入向量不归一化时，才选择 metric="L2"。
- 想要更快，可以微调搜索索引参数。
- 大数据想要更快，选择带向量压缩的索引，在[索引文档页](https://milvus.io/docs/index.md)搜索"Quantization-based index"（基于量化的索引）。<br><br>

<a class="anchor" id="consistency"></a>
7. **选择[一致性级别](https://milvus.io/docs/consistency.md)。**

- 💡👉🏼 **对于典型使用场景（例如表每 30 分钟或更长时间更新一次），使用 "Eventually"（最终一致性）以获得最快性能。**
- 4 种可用的一致性级别：
  - Strong（强一致性）——实时，所有人都看到相同的数据。
  - Eventually（最终一致性）——很快，所有人都看到相同的数据。
  - Session（会话一致性）——会话内，数据与本次会话中的所有写入保持一致。
  - Bounded（有界一致性）——比最终一致性更短的时间内，所有人都看到相同的数据。

- 在两处指定一致性：
  - 在 collection.create_collection() 中——设置默认值。
  - 在 collection.search() 中——可以覆盖默认值。<br><br>

<a class="anchor" id="insert-data"></a>
8. **向集合中[插入数据](https://milvus.io/docs/insert_data.md)。**

- Milvus 支持从以下来源加载数据：
  - pandas DataFrame，或
  - 字典列表

- 💡👉🏼 Milvus Client 封装只能处理字典列表形式的数据加载。

<div font size="1">

```python
# 将 DataFrame 转换为字典列表。
dict_list = []
for _, row in batch.iterrows():
    dictionary = row.to_dict()
    dict_list.append(dictionary)

print("Start inserting entities")
start_time = time.time()
insert_result = mc.insert(
    COLLECTION_NAME,
    data=dict_list,
    progress_bar=True)  # 显示一个动态进度条
end_time = time.time()
print(f"Milvus insert time for {batch.shape[0]} vectors: {end_time - start_time} seconds")
# 插入完所有实体后，调用 flush，避免 growing segments 一直留在内存中。
mc.flush(COLLECTION_NAME)
```
</div>
<br>

<a class="anchor" id="search"></a>
9. **[搜索](https://milvus.io/docs/search.md)你的全部数据。** Milvus 搜索默认是在向量空间中使用近似最近邻（ANN）距离的语义搜索，或者说随机模糊搜索。使用的搜索算法取决于你创建集合时选择的索引。

- 参见 [Milvus search API](https://milvus.io/api-reference/pymilvus/v2.3.x/Collection/search().md) 文档。
- 参见 [Milvus Client search API](https://pymilvus.readthedocs.io/en/latest/_modules/milvus/client/stub.html)。两者的 API 和搜索结果对象略有不同。

<div font size="1">

```python
# 使用同一个编码器对问题做嵌入。
query_embeddings = _utils.embed_query(encoder, [SAMPLE_QUESTION])

# 使用 HNSW 索引返回 top k 结果。
SEARCH_PARAMS = dict({
    "ef": INDEX_PARAMS['efConstruction']
    })

# 定义要返回的输出字段。
OUTPUT_FIELDS = ["h1", "h2", "source", "chunk"]

# 使用查询向量和向量数据库执行语义向量搜索。
start_time = time.time()
results = mc.search(
    COLLECTION_NAME,
    data=query_embeddings,
    search_params=SEARCH_PARAMS,
    output_fields=OUTPUT_FIELDS,
    # Milvus 可以利用布尔表达式形式的元数据过滤搜索。
    # filter="pk >= 0",
    limit=3,  # 默认 top_k = 10
    consistency_level="Eventually"
    )
elapsed_time = time.time() - start_time
print(f"Milvus Client search time for {len(chunk_list)} vectors: {elapsed_time} seconds")

# 检查搜索结果。
print(f"type: {type(results[0])}, count: {len(results[0])}")
```
</div>

- 与 SQL 数据库概念类似，除了向量搜索外，还可以使用[布尔表达式](2022-08-08-How-to-use-string-data-to-empower-your-similarity-search-applications.md)指定标量（[元数据过滤](https://milvus.io/docs/hybridsearch.md)）。
  - "filter": "boolean_expression"
  ```python
  "filter": "email == 'tom@zilliz.com' "
  ```
  - 任何字符串字面量都要用单引号 ' 包裹。
  - 使用 &&（与）或 ||（或）连接布尔表达式。
  - 字符串匹配只对**锚定字符串（anchored strings）**有效。
  ```python
  "filter":"((DatePublished >= 2000) && (RatingValue > 6.8)) || (MovieName != 'Deepsea Challenge%')"
  ```
  - 使用 "in" 和 "like" 的字符串匹配同样只支持**锚定字符串**。
    - "my_string in 'prefix%'"
    - "my_string like 'prefix%'"
  - 数组元数据支持从 Milvus v2.3 开始。
    - A in ["str1", "str2"]

- 如果需要手动控制语义搜索，可以使用[范围（特定向量距离）](https://milvus.io/docs/search.md#Prepare-search-parameters)搜索。

- 当数据集太大无法放入内存时，Milvus 提供 [DiskANN](https://zilliz.com/learn/DiskANN-and-the-Vamana-Algorithm)。<br><br>

<a class="anchor" id="upsert"></a>
10. **使用 ["upsert"（更新插入）](https://milvus.io/docs/upsert_entities.md) 操作更新数据。** 如果向量不存在则插入新向量，如果数据已存在则更新数据库中的现有数据。

- Upsert 从 milvus v2.3 开始支持。

- AutoID [不能为 True](https://milvus.io/docs/upsert_entities.md#Limits)！你的 pk 必须手动指定，即：
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False,)

- 要 upsert 的数据必须具有新的、唯一的 pk，否则当 pk 匹配时，upsert 的数据会[覆盖已有数据](https://github.com/milvus-io/milvus/discussions/28744)。

- 如果需要 [Strong "读后写"（read-after-write）](https://github.com/milvus-io/milvus/blob/f3f46d3bb2dcae2de0bdb7bc0f7b20a72efceaab/docs/developer_guides/how-guarantee-ts-works.md)一致性，请将 [ignore_growing segments](https://milvus.io/docs/search.md#Prepare-search-parameters) 设置为 True 后再执行 [upsert](https://milvus.io/docs/upsert_entities.md#Upsert-Entities)。 <br><br>

<a class="anchor" id="query"></a>
11. **["Query"（查询）](https://milvus.io/docs/query.md) 操作不使用模糊搜索（语义搜索）。**

- 例如，你想知道某个 productID 是否已存在。
res = collection.query(expr = "ProductID == 100")
如果 len(res) 为 0，我们就知道没有 product id 为 100 的条目。

## 示例 Notebook {#example-notebooks}

1. 连接 Milvus 入门：https://github.com/milvus-io/bootcamp/blob/master/bootcamp/milvus_connect.ipynb
2. 使用 Milvus Client 加载和搜索 IMDB 电影数据：https://github.com/milvus-io/bootcamp/blob/master/bootcamp/Retrieval/imdb_milvus_client.ipynb
3. 使用开源 LLM（也支持 OpenAI）在网站数据上构建 RAG 聊天机器人：https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/readthedocs_zilliz_langchain.ipynb
4. 使用 Ragas 和 OpenAI 评估 RAG：https://github.com/milvus-io/bootcamp/blob/master/evaluation/evaluate_fiqa_customized_RAG.ipynb
5. 使用 LlamaIndex 构建 OpenAI agent：https://github.com/milvus-io/bootcamp/blob/master/bootcamp/OpenAIAssistants/milvus_agent_llamaindex.ipynb

## 学习资源 {#learning-resources}

- **[OSSChat 演示](https://osschat.io/)**：[代码](https://github.com/zilliztech/akcio) 在 GitHub 上
- **[向量数据库 101](https://zilliz.com/learn/what-is-vector-database)** 博客系列
- **[Milvus bootcamp](https://github.com/milvus-io/bootcamp/tree/master/bootcamp)** 教程
- **[VectorDBBench](https://github.com/zilliztech/VectorDBBench)**，一个开源基准测试工具，允许用户使用自己的数据测量 Milvus 或 Zilliz Cloud 与其他产品的性能对比。
  - [Milvus vs PgVector](https://medium.com/@zilliz_learn/getting-started-with-pgvector-a-guide-for-developers-exploring-vector-databases-9c2295bb13e5)（Postgres 向量插件）——滚动到底部查看图表
  - [Milvus vs Qdrant vs Elastic vs Weaviate](https://zilliz.com/vector-database-benchmark-tool?database=ZillizCloud%2CMilvus%2CPgVector%2CElasticCloud%2CPinecone%2CQdrantCloud%2CWeaviateCloud&dataset=medium&filter=none%2Clow%2Chigh)

## 社区与帮助 {#community--help}

- [Milvus 公共 Discord](https://discord.gg/8uyFbECzPX) —— 聊天内容对所有人公开可见。
- [Milvus GitHub 讨论与 issue](https://github.com/milvus-io/milvus/discussions) —— 开源 Milvus 的 GitHub 讨论和问题。在那里点击"New discussion"或提交 issue。
- [Zilliz 云问题](https://support.zilliz.com/hc/en-us/requests/new) —— Zilliz 付费版支持工单会获得更多关注和更高优先级。可以从 [Zilliz 控制台](https://cloud.zilliz.com/) 创建工单。
- Zilliz 私有支持 Slack —— 联系你的 Zilliz 销售代表获取邀请。
- [非结构化数据线下聚会（旧金山和西雅图）](https://www.meetup.com/unstructured-data/) —— 每月一次，我们作为社区互相学习，主题与非结构化数据和 AI 相关。
- [YouTube 频道](https://www.youtube.com/@MilvusVectorDatabase/playlists)
- [社交媒体：Milvus 在 LinkedIn](https://www.linkedin.com/company/the-milvus-project/)
- [社交媒体：Zilliz 在 LinkedIn](https://www.linkedin.com/company/zilliz/)
- [社交媒体：Milvus 在 Twitter](https://twitter.com/milvusio)
- [社交媒体：Zilliz 在 Twitter](hhttps://twitter.com/zilliz_universe)

import time
import os
import numpy as np
import torch
from rich.jupyter import display
from torch.nn import functional as F

CUSTOM_CACHE = r'F:\Teewon\Milvue\models'
os.environ['HF_HOME'] = CUSTOM_CACHE
os.environ['HF_HUB_CACHE'] = os.path.join(CUSTOM_CACHE, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CUSTOM_CACHE, 'transformers')

##########
# Functions for IMDB demo notebook.
# Data source: Stanford AI Lab https://ai.stanford.edu/~amaas/data/sentiment/
##########

# Output words instead of scores.
def sentiment_score_to_name(score:float):
    if score > 0:
        return "Positive"
    elif score <= 0:
        return "Negative"

# Split data into train, valid, test.
def partition_dataset(df_input, new_columns, smoke_test=False):
    """按比例划分数据，假设原始输入的DataFrame包含5万行。

    参数：
    df_input (pandas.DataFrame)：输入的数据框
    smoke_test (bool)：如果为True，则用于测试的行数较少

    返回值：
    df_train, df_val, df_test (pandas.DataFrame)：训练集、验证集和测试集的划分。
    """

    # 打乱数据并划分成训练/验证/测试集
    df_shuffle=df_input.sample(frac=1,random_state=1).reset_index()
    df_shuffle.columns=new_columns

    df_train=df_shuffle.iloc[:35_000]
    df_val=df_shuffle.iloc[35_000:40_000]
    df_test=df_shuffle.iloc[40_000:]

    # 将训练/验证/测试数据分别保存为本地文件
    df_train.to_csv("train.csv",index=False,encoding="utf-8")
    df_val.to_csv("val.csv",index=False,encoding="utf-8")
    df_test.to_csv("test.csv",index=False,encoding="utf-8")

    return df_shuffle,df_train, df_val, df_test

# 用于测试 chunk_size 的函数
def imdb_chunk_test(encoder, batch_size, df, chunk_size, chunk_overlap):
    batch=df.head(batch_size).copy()
    print(f"chunk_size:{chunk_size}")
    print(f"original shape:{batch.shape}")

    start_time = time.time()
    # 1. Change primary key type to string
    batch["movie_index"]=batch["movie_index"].apply(lambda x: str(x))

    # 2. Split the documents into smaller chunks and add as new column to batch df
    batch['chunk']=batch['text'].apply(recursive_splitter_wrapper, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Explode the 'chunk' column to create new rows for each chunk
    batch=batch.explode('chunk', ignore_index=True)
    print(f"chunk shape:{batch.shape}")

    #  3. Add embeddings as new column in df
    review_embeddings=torch.tensor(encoder.encode(batch['chunk']))
        # Normalize embeddings to unit length
    review_embeddings=F.normalize(review_embeddings, p=2, dim=1) # 使用 L2 范数（欧几里得范数）进行归一化
        # Quick check if embeddings are normalized
    norms=np.linalg.norm(review_embeddings, axis=1) # 按行计算每个向量的欧几里得长度
    assert np.allclose(norms, 1.0, atol=1e-5)==True # 容差比较（而非直接等于）

    # 4. Convert embeddings to list of `numpy.ndarray`, each containing `numpy.float32` numbers
    converted_values=list(map(np.float32, review_embeddings))
    batch['vector']=converted_values

    # 5. Reorder columns for conveneince, so index first, labels at end 为方便起见重新排列Columns
    new_order=["movie_index","text","chunk","vector","label_int","label"]
    batch=batch[new_order]

    end_time=time.time()
    print(f"Chunking + embeddings time for {batch_size} docs: {end_time-start_time} sec")

    # Inspect the batch of data
    display(batch.head())

    # assert len(batch.chunk[0])<=MAX_SEQ_LENGTH-1
    # assert len(batch.vector[0])==EMBEDDING_LENGTH
    print(f"type embeddings: {type(batch.vector)} of {type(batch.vector[0])}")
    print(f"of numbers: {type(batch.vector[0][0])}")

    # Chunking looks good, drop the original text columns
    batch.drop(columns=["text"],inplace=True)

    return batch

# Function for embedding a query.
def embed_query(encoder, query):
    # Embed the query using same embedding model used to create the Milvus collections
    query_embeddings=torch.tensor(encoder.encode(query))
    # Normalize embeddings to unit length
    query_embeddings=F.normalize(query_embeddings, p=2, dim=1)
    # Quick check if embeddings are normalized
    norms=np.linalg.norm(query_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)==True
    # Convert the embeddings to list of list of np.float32
    query_embeddings=list(map(np.float32, query_embeddings))

    return query_embeddings

##########
# Functions for LangChain chunking and embedding.
##########
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_splitter_wrapper(text, chunk_size, chunk_overlap):
    # Define chunk overlap is 10% chunk_size
    chunk_overlap=np.round(chunk_size*0.10,0)

    # Use langchain's convenient recursive(递归) chunking method
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks:List[str]=text_splitter.split_text(text)

    # Replace special characters with spaces
    chunks=[text.replace("<br /><br />"," ")for text in chunks]
    return chunks

##########
# Functions to process Milvus Client API responses.
##########
def client_assemble_retrieved_context(retrieved_top_k, metadata_fields=[],num_shot_answers=3):
    """
    对于每个问题，从检索到的 top_k 段中组装上下文和元数据。
    retrieved_top_k：字典列表
    """
    # Assemble the context as a stuffed string 将上下文拼接成一个填充的字符串。
    distances=[]
    context=[]
    context_metadata=[]
    i=1
    for r in retrieved_top_k[0]:
        distances.append(r['distance'])
        if i<=num_shot_answers:
            if len(metadata_fields)>0:
                metadata={}
                for field in metadata_fields:
                    metadata[field]=r['entity'][field]
                context_metadata.append(metadata)
            context.append(r['entity']['chunk'])
        i+=1

    # Assemble formatted results in a zipped list 将格式化结果整理成压缩列表
    formatted_results=list(zip(distances, context, context_metadata))

    # Return all the thins for convenience
    return formatted_results,context,context_metadata

##########
# Functions to process Milvus Search API responses.
##########
# 从 Milvus 搜索响应中解析出答案和上下文元数据
def assemble_answer_sources(answer, context_metadata):
    """Assemble the answer and grounding sources into a string"""
    grounded_answer=f"Answer: {answer}"
    grounded_answer+="Grounding sources and citations:\n"

    for metadata in context_metadata:
        try:
            grounded_answer+=f"'h1': {metadata['h1']},'h2':{metadata['h2']}\n"
        except:
            pass
        try:
            grounded_answer+=f"'source': {metadata['source']}"
        except:
            pass

    return grounded_answer

# 将内容插入到上下文字符串中，并将元数据放入字典列表中
def assemble_retrieved_context(retrieved_result, metadata_fields=[],num_shot_answers=3):
    # 将上下文拼接成一个填充的字符串
    # 因此，保存上下文元数据以便与答案一起检索
    context=[]
    context_metadata=[]
    i=1
    for r in retrieved_result[0]:
        if i<=num_shot_answers:
            if len(metadata_fields)>0:
                metadata={}
                for field in metadata_fields:
                    metadata[field]=getattr(r.entity,field,None)
                context_metadata.append(metadata)
            context.append(r.entity.text)
        i+=1

    return context,context_metadata
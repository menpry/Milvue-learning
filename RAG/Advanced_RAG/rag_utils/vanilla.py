from typing import List
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv("../../.env")

CUSTOM_CACHE = r'F:\Teewon\Milvue\models'
os.environ['HF_HOME'] = CUSTOM_CACHE
os.environ['HF_HUB_CACHE'] = os.path.join(CUSTOM_CACHE, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CUSTOM_CACHE, 'transformers')
os.environ['TORCH_HOME'] = CUSTOM_CACHE

PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""
rag_prompt=PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

embeddings=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

llm=ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-v4-pro",
    temperature=0
)

vectorstore=Milvus(
    embedding_function=embeddings,
    connection_args={
        "uri":"../../milvus_demo.db"
    },
    auto_id=True,
    drop_old=True,
)

def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)
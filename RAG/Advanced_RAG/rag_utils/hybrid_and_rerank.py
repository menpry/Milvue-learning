import os

CUSTOM_CACHE = r'F:\Teewon\Milvue\models'
os.environ['HF_HOME'] = CUSTOM_CACHE
os.environ['HF_HUB_CACHE'] = os.path.join(CUSTOM_CACHE, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CUSTOM_CACHE, 'transformers')
os.environ['TORCH_HOME'] = CUSTOM_CACHE

from typing import List,Optional,Sequence
from pydantic import Field, ConfigDict

from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.runnables import Runnable,RunnableConfig
from langchain_core.runnables.utils import Input, Output
from sentence_transformers import CrossEncoder


class RerankerRunnable(Runnable):
    def __init__(self,compressor:BaseDocumentCompressor,top_k:int=4):
        self.compressor = compressor
        self.top_k = top_k

    def _remove_duplicates(self,retrieved_documents:List[Document]):
        seen_page_content=set()
        unique_documents=[]
        for doc in retrieved_documents:
            if doc.page_content not in seen_page_content:
                unique_documents.append(doc)
                seen_page_content.add(doc.page_content)
        return unique_documents

    def invoke(self,input:Input,config:Optional[RunnableConfig]=None)->Output:
        milvus_retrieved_doc:List[Document]=input.get("milvus_retrieved_doc")
        bm25_retrieved_doc:List[Document]=input.get("bm25_retrieved_doc")
        query:str=input.get("query")
        print(f"len(milvus_retrieved_doc)={len(milvus_retrieved_doc)}")
        print(f"len(bm25_retrieved_doc)={len(bm25_retrieved_doc)}")
        unique_documents=self._remove_duplicates(milvus_retrieved_doc+bm25_retrieved_doc)
        print(f"len(unique_documents)={len(unique_documents)}")
        result=self.compressor.compress_documents(unique_documents,query)
        return result


class CrossEncoderReranker(BaseDocumentCompressor):
    model_config=ConfigDict(arbitrary_types_allowed=True)
    model:Optional[CrossEncoder]=Field(default=None,exclude=True)
    top_k:int=4

    def __init__(self,model_name:str="cross-encoder/ms-marco-MiniLM-L-6-v2",top_k:int=4):
        super().__init__()
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def compress_documents(self,documents:List[Document],query:str,**kwargs)->Sequence[Document]:
        if not documents:
            return []

        pairs=[(query,doc.page_content)for doc in documents]
        scores=self.model.predict(pairs)

        scored_docs=sorted(
            zip(documents,scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        return [doc for doc,_ in scored_docs]
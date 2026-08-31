from typing import List
from langchain_core.documents import Document

def write_wider_window(split_docs: List[Document],original_documents:Document,offset:int=200):
   original_text=original_documents.page_content
   for doc in split_docs:
       doc_text=doc.page_content
       start_index=original_text.index(doc_text)
       end_index=start_index+len(doc_text)-1
       winder_text=original_text[max(start_index-offset,0):min(end_index+offset,len(original_text))]
       doc.metadata["start_index"]=start_index
       doc.metadata["end_index"]=end_index
       doc.metadata["winder_text"]=winder_text


def format_docs_with_winder_window(docs:List[Document]):
    return "\n\n".join([doc.metadata["winder_text"] for doc in docs])
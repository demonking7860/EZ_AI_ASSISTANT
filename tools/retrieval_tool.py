from typing import Optional
from langchain_chroma import Chroma

class DocumentRetriever:
    def __init__(self, vectorstore: Optional[Chroma] = None):
        self.vectorstore = vectorstore

    def retrieve(self, query: str) -> str:
        if self.vectorstore is None:
            return "No document has been processed yet. Cannot perform retrieval."
        
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)
            retrieved_content = "\n\n".join([doc.page_content for doc in docs])
            print(f"Retrieved {len(docs)} chunks for query: {query[:50]}...")
            return retrieved_content
        except Exception as e:
            return f"Error retrieving document content: {str(e)}"

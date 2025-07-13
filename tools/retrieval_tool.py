from langchain_chroma import Chroma

class DocumentRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(self, query: str) -> str:
        if not self.vectorstore:
            return "No document processed."
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

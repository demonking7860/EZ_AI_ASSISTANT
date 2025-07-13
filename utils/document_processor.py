import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persistent_directory = os.path.join(self.current_dir, "..", "db", "chroma_db")

    def clear_existing_data(self):
        """Clear existing vector database to ensure fresh start for new document"""
        try:
            if os.path.exists(self.persistent_directory):
                shutil.rmtree(self.persistent_directory, ignore_errors=True)
                print(f"Cleared existing ChromaDB directory: {self.persistent_directory}")
        except Exception as e:
            print(f"Warning: Could not clear existing data: {e}")

    def process_document(self, file_path: str, file_type: str) -> Chroma:
        # Clear existing data first to ensure only current document is processed
        self.clear_existing_data()
        
        # Create the directory if it doesn't exist
        os.makedirs(self.persistent_directory, exist_ok=True)
        
        # Load and process document
        loader = PyPDFLoader(file_path) if file_type == "pdf" else TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        
        # Create fresh vector store
        vectorstore = Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.persistent_directory
        )
        
        print(f"Processed {len(docs)} document chunks into ChromaDB")
        return vectorstore

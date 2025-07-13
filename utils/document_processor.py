import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class DocumentProcessor:
    def __init__(self):
        # Smaller model to reduce RAM use
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-albert-small-v2",
            cache_folder=os.getenv("HF_MODEL_CACHE", "./models")
        )
        base = os.path.dirname(os.path.abspath(__file__))
        # Allow override via env var (e.g., /mnt/data)
        self.persistent_directory = os.getenv("CHROMA_DIR",
                                              os.path.join(base, "..", "db", "chroma_db"))

    def clear_existing_data(self):
        if os.path.exists(self.persistent_directory):
            shutil.rmtree(self.persistent_directory, ignore_errors=True)

    def process_document(self, file_path: str, file_type: str) -> Chroma:
        self.clear_existing_data()
        os.makedirs(self.persistent_directory, exist_ok=True)
        loader = PyPDFLoader(file_path) if file_type == "pdf" \
                 else TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.persistent_directory
        )
        return vectorstore

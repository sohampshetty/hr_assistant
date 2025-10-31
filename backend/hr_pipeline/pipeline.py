from metaflow import FlowSpec, step
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import os

class HRPolicyPipeline(FlowSpec):
    @step
    def start(self):
        print("Starting HR policy document processing pipeline...")
        self.pdf_dir = "./policy_pdfs"
        self.next(self.load_pdfs)

    @step
    def load_pdfs(self):
        self.documents = []
        pdf_files = [os.path.join(self.pdf_dir, f) for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")]
        print(f"Found {len(pdf_files)} PDF files.")
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            self.documents.extend(loader.load())
        print(f"Loaded {len(self.documents)} document chunks.")
        self.next(self.split_documents)

    @step
    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.text_chunks = splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
        self.next(self.create_embeddings)

    @step
    def create_embeddings(self):
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(self.text_chunks, embedding_model)
        self.vector_store.save_local("faiss_hr_policy_index")
        print("Created and saved FAISS vector store index locally.")
        self.next(self.end)

    @step
    def end(self):
        print("Pipeline completed successfully. Vector store ready for queries.")

if __name__ == "__main__":
    HRPolicyPipeline()

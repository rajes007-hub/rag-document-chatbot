import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables (.env)
load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. Please create it and add your files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. Please add documents."
        )

    # Preview first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}")
        print(f"  Source : {doc.metadata['source']}")
        print(f"  Length : {len(doc.page_content)} chars")
        print(f"  Preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into chunks"""
    print("\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Preview first 5 chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source : {chunk.metadata['source']}")
        print(f"Length : {len(chunk.page_content)} chars")
        print(chunk.page_content)
        print("-" * 50)

    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist Chroma vector store"""
    print("\nCreating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector store created successfully")
    print(f"Saved to: {persist_directory}")

    return vectorstore


def main():
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    # If vector store already exists, load it
    if os.path.exists(persist_directory):
        print("✅ Existing vector store found. Loading...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print(f"Loaded vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    print("No existing vector store found. Creating a new one...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persist_directory)

    print("\n✅ Ingestion complete! Documents are ready for RAG.")
    return vectorstore


if __name__ == "__main__":
    main()

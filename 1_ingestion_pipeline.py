import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import time


load_dotenv()

def load_documents(path="docs"):
    """Load all the files from the docs directory"""
    print(f"Loading files from {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist. Plz create a new One")

    loader = DirectoryLoader(
        path=path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if (len(documents) == 0):
        raise FileNotFoundError(f"No .txt file found in {path}.Plz create a new One")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\n Document {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size = 1000, chunk_overlap = 0):
    """Split document into smaller chunks with overlap"""
    print("Splitting documenst into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---- Chunk {i+1} ----")
            print(f"Source: {chunk.metadata["source"]}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*30)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using rate-limit safe batching"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    print("---- Creating Vector Store ----")
    
    # 1. Initialize an empty Chroma instance configuration
    vectorstore = Chroma(
        collection_name="rag_collection",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # 2. Segment chunks into controlled sub-batches
    batch_size = 40
    total_chunks = len(chunks)
    
    print(f"Starting rate-limited ingestion loop for {total_chunks} chunks...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (total_chunks - 1) // batch_size + 1
        
        print(f"--> Uploading batch {current_batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        # Upload just this safe sub-batch to your database
        vectorstore.add_documents(batch)
        
        # 3. Introduce a cooldown pause if there are remaining batches left to run
        if i + batch_size < total_chunks:
            print("Respecting API limits: Cooling down for 15 seconds...")
            time.sleep(15)

    print("--- Finished creating vector store ----")
    print(f"Vector store successfully created and saved to {persist_directory}")

    return vectorstore


def main():
    print("Data Ingestion")

    documents = load_documents(path="docs")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()    

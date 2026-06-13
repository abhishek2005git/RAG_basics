from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

db = Chroma(
    collection_name="rag_collection",
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query = "Who is the founder of amazon"

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

print("---- Context ----")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n{doc.page_content}\n")



# if __name__ == "__main__":
#     main() 
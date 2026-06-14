from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
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

query = "What is amazon??"

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

print("---- Context ----")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n{doc.page_content}\n")


combined_input = f"""Based on the following retrieved documents, answer the question: {query}   
Documents:
{chr(10).join(doc.page_content for doc in relevant_docs)}
Please provide a concise and accurate answer to the question based on the information from the documents.If the documents do not contain relevant information, please state that you don't know the answer.
"""

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

print("---- LLM Generation ----")
print("Sending context to Groq...")

response = model.invoke(combined_input)
print("\n---- Response ----")
print(response.content)
print("======================\n")
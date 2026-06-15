from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

db = Chroma(
    collection_name="rag_collection",
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

chat_histroy = []

def ask_question(user_question):
    print(f"\n You asked: {user_question}---")

    if chat_histroy:
        messages = [
            SystemMessage(content="Given the chat history rewrite the new question to be standalone and searchable. Just return the rewritten question.")
        ] + chat_histroy + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for question: {search_question}")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k":3})

    docs = retriever.invoke(search_question)
    print(f"Found {len(docs)} relevant documents")

    for i, doc in enumerate(docs):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i+1}: {preview}")

    combined_input = f"""Based on the following retrieved documents, answer the question: {user_question}   
    Documents:
    {chr(10).join(doc.page_content for doc in docs)}
    Please provide a concise and accurate answer to the question based on the information from the documents.If the documents do not contain relevant information, please state that you don't know the answer.
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents.")] + chat_histroy + [
            HumanMessage(content=combined_input)
        ]
    
    result = model.invoke(messages)
    answer = result.content

    chat_histroy.append(HumanMessage(content=user_question))
    chat_histroy.append(AIMessage(content=answer))

    print(f"\nAnswer : {answer}")
    return answer

def start_chat():
    print("Ask me a question or type 'quit' to exit.")

    while(True):
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Exiting loop bye!!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()
    

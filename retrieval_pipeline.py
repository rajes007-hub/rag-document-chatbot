from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# 1️⃣ Load the SAME embeddings used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2️⃣ Load existing vector DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# 3️⃣ User query
query = "How much did Microsoft pay to acquire GitHub?"
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("\n--- Retrieved Context ---\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 60)

# 4️⃣ Build RAG prompt
combined_input = f"""
Answer the question using ONLY the information from the documents below.

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

If the answer is not present, say:
"I don't have enough information to answer that question."
"""

# 5️⃣ UPDATED: Use current Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",  # ✅ CURRENT MODEL (Feb 2025)
    temperature=0
)

messages = [
    SystemMessage(content="You are a precise assistant that answers strictly from provided context."),
    HumanMessage(content=combined_input),
]

# 6️⃣ Generate answer
result = model.invoke(messages)
print("\n--- Generated Response ---")
print(result.content)
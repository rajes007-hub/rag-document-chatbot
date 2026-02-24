import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

if os.path.exists('.env'):
    load_dotenv()
else:
    if hasattr(st, 'secrets'):
        os.environ['GROQ_API_KEY'] = st.secrets.get("GROQ_API_KEY", "")


# SESSION STATE INITIALIZATION
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

if 'use_default_docs' not in st.session_state:
    st.session_state.use_default_docs = True

if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = []


# PAGE SETUP

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Q&A Chatbot")
st.markdown("Ask questions about documents - use default docs or upload your own!")


# HELPER FUNCTIONS

@st.cache_resource
def get_embedding_model():
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

@st.cache_resource
def load_default_db():
    
    persistent_directory = "db/chroma_db"
    
    if not os.path.exists(persistent_directory):
        return None
    
    embedding_model = get_embedding_model()
    
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return db

def process_uploaded_file(uploaded_file):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_path, encoding='utf-8')
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None
        
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def create_vectordb_from_uploads(uploaded_files):
    all_chunks = []
    
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            chunks = process_uploaded_file(uploaded_file)
            if chunks:
                all_chunks.extend(chunks)
                st.success(f"✅ Processed: {uploaded_file.name} ({len(chunks)} chunks)")
    
    if not all_chunks:
        st.error("No valid documents to process!")
        return None
    
    with st.spinner("Building vector database..."):
        embedding_model = get_embedding_model()
        db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
    return db

def query_documents(user_question, db, model):
    if st.session_state.chat_history:
        messages = [
            SystemMessage(content="Rewrite this question to be standalone based on the chat history. Return ONLY the rewritten question."),
        ] + st.session_state.chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
    else:
        search_question = user_question
    
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{docs_text}

Instructions:
- Provide a clear, helpful answer using ONLY information from these documents
- If you cannot find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
- Be concise but complete
"""
    
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history. Be concise and accurate."),
    ] + st.session_state.chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    return answer, docs


# SIDEBAR for Document Selection

with st.sidebar:
    st.header("📂 Document Source")
    
    doc_source = st.radio(
        "Choose document source:",
        ["Use Default Documents", "Upload My Own Documents"],
        key="doc_source"
    )
    
    if doc_source == "Upload My Own Documents":
        st.session_state.use_default_docs = False
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF or TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload one or more PDF or TXT files to query"
        )
        
        if uploaded_files:
    
            current_file_names = [f.name for f in uploaded_files]
            
            if current_file_names != st.session_state.uploaded_files_processed:
                
                st.session_state.vector_db = create_vectordb_from_uploads(uploaded_files)
                st.session_state.uploaded_files_processed = current_file_names
                st.session_state.chat_history = []  
                
                if st.session_state.vector_db:
                    st.success(f"✅ Ready to query {len(uploaded_files)} file(s)!")
        else:
            st.info("👆 Upload files to get started")
            st.session_state.vector_db = None
    
    else:
        st.session_state.use_default_docs = True
        st.session_state.vector_db = load_default_db()
        st.session_state.uploaded_files_processed = []
        
        if st.session_state.vector_db:
            try:
                doc_count = st.session_state.vector_db._collection.count()
                st.success(f"✅ Using default database ({doc_count} documents)")
            except:
                st.success("✅ Using default database")
    
    st.divider()
    
    # Stats
    st.subheader("⚙️ Settings")
    st.metric("Chat Messages", len(st.session_state.chat_history))
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Info
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. Choose document source
    2. Ask your question
    3. AI retrieves relevant context
    4. Generates accurate answer
    """)
    
    st.divider()
    st.markdown("### 🔑 Tech Stack")
    st.markdown("""
    - **LLM:** Groq (Llama 3.1)
    - **Embeddings:** HuggingFace
    - **Vector DB:** ChromaDB
    - **Framework:** LangChain
    """)


# MAIN CHAT INTERFACE

if st.session_state.vector_db is None:
    if st.session_state.use_default_docs:
        st.error("❌ Default database not found! Please upload your own documents or check that db/chroma_db exists.")
    else:
        st.info("👈 Please upload documents from the sidebar to get started")
    st.stop()

model = get_llm()

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)
    
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer, source_docs = query_documents(
                    user_question,
                    st.session_state.vector_db,
                    model
                )
                
        
                st.write(answer)
                
                
                with st.expander("📄 View source documents"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Document {i}:**")
                        preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        st.text(preview)
                        st.divider()
                
                
                st.session_state.chat_history.append(HumanMessage(content=user_question))
                st.session_state.chat_history.append(AIMessage(content=answer))
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)
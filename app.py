import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables
# When running locally, it reads from .env file
# When deployed, we'll use Streamlit secrets instead
if os.path.exists('.env'):
    load_dotenv()
else:
    # Running on Streamlit Cloud - use secrets
    if hasattr(st, 'secrets'):
        os.environ['GROQ_API_KEY'] = st.secrets.get("GROQ_API_KEY", "")

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Session state stores data that persists across reruns
# Think of it like variables that don't reset when page refreshes
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Q&A Chatbot")
st.markdown("Ask questions about your documents using RAG!")

# =============================================================================
# LOAD RAG SYSTEM (CACHED)
# =============================================================================

@st.cache_resource  # This decorator caches the function so it only runs once
def load_rag_system():
    """
    Load the vector database and language model
    This is cached so it doesn't reload on every interaction
    """
    persistent_directory = "db/chroma_db"
    
    # Check if database exists
    if not os.path.exists(persistent_directory):
        st.error("❌ Database not found! Please run ingest.py first.")
        st.stop()
    
    # Load embeddings model (same one used during ingestion)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load vector database
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize Groq LLM
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    
    return db, model

# =============================================================================
# MAIN APP
# =============================================================================

try:
    # Load the RAG system
    db, model = load_rag_system()
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        
        # Chat input at the bottom
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    
                    # Step 1: Rewrite question if chat history exists
                    if st.session_state.chat_history:
                        messages = [
                            SystemMessage(content="Rewrite this question to be standalone based on the chat history. Return ONLY the rewritten question, nothing else."),
                        ] + st.session_state.chat_history + [
                            HumanMessage(content=f"New question: {user_question}")
                        ]
                        result = model.invoke(messages)
                        search_question = result.content.strip()
                    else:
                        search_question = user_question
                    
                    # Step 2: Retrieve relevant documents
                    retriever = db.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(search_question)
                    
                    # Step 3: Create context from documents
                    docs_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                    
                    # Step 4: Create the prompt
                    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{docs_text}

Instructions:
- Provide a clear, helpful answer using ONLY information from these documents
- If you cannot find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
- Be concise but complete
"""
                    
                    # Step 5: Get answer from LLM
                    messages = [
                        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history. Be concise and accurate."),
                    ] + st.session_state.chat_history + [
                        HumanMessage(content=combined_input)
                    ]
                    
                    result = model.invoke(messages)
                    answer = result.content
                    
                    # Display the answer
                    st.write(answer)
                    
                    # Optional: Show retrieved documents in an expander
                    with st.expander("📄 View source documents"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Document {i}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.divider()
                    
                    # Save to chat history
                    st.session_state.chat_history.append(HumanMessage(content=user_question))
                    st.session_state.chat_history.append(AIMessage(content=answer))
    
    # Sidebar
    with col2:
        st.subheader("⚙️ Settings")
        
        # Show database stats
        try:
            doc_count = db._collection.count()
            st.metric("Documents in DB", doc_count)
        except:
            st.metric("Documents in DB", "Unknown")
        
        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Info section
        st.divider()
        st.markdown("### 📖 How it works")
        st.markdown("""
        1. Your question is processed
        2. Relevant documents are retrieved
        3. AI generates answer based on documents
        4. Sources are shown below
        """)
        
        st.divider()
        st.markdown("### 🔑 Tech Stack")
        st.markdown("""
        - **LLM:** Groq (Llama 3.1)
        - **Embeddings:** HuggingFace
        - **Vector DB:** ChromaDB
        - **Framework:** LangChain
        """)

except Exception as e:
    st.error(f"❌ Error loading RAG system: {e}")
    st.info("""
    **Troubleshooting:**
    1. Make sure you've run `ingest.py` first
    2. Check that `db/chroma_db` folder exists
    3. Verify your GROQ_API_KEY is set correctly
    4. Install all requirements: `pip install -r requirements.txt`
    """)
    st.exception(e)  # Show full error for debugging
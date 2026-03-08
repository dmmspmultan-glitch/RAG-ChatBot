import os
import tempfile
import json
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------- Page config ----------
st.set_page_config(page_title="📄 RAG Q&A Pro", layout="wide")
load_dotenv()

# ---------- Persistent memory with JSON ----------
def load_history(session_id: str):
    """Return a list of (role, content) tuples from the session's JSON file."""
    filename = f"history_{session_id}.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [(item["role"], item["content"]) for item in data]
    return []

def save_message(session_id: str, role: str, content: str):
    """Append a message to the session's JSON file."""
    filename = f"history_{session_id}.json"
    history = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({"role": role, "content": content})
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def clear_history(session_id: str):
    """Delete the JSON file for the session."""
    filename = f"history_{session_id}.json"
    if os.path.exists(filename):
        os.remove(filename)

# ---------- Sidebar config ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs and ask questions about them.")

    st.subheader("💬 Session")
    session_id = st.text_input("Session ID", value="default_session", key="session_input")

    # Show number of messages in current session
    msg_count = len(load_history(session_id))
    st.caption(f"📊 {msg_count} messages in this session")

    if st.button("🗑️ Clear session history"):
        clear_history(session_id)
        if session_id in st.session_state.chat_history_objects:
            del st.session_state.chat_history_objects[session_id]
        st.rerun()

    # Placeholder for progress messages during indexing
    progress_placeholder = st.empty()

api_key = api_key_input or os.getenv("GROQ_API_KEY")
if not api_key:
    st.warning("Please enter your Groq API Key (or set GROQ_API_KEY in .env)")
    st.stop()

# ---------- Embeddings and LLM ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

# ---------- File upload and indexing ----------
uploaded_files = st.file_uploader(
    "📚 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin")
    st.stop()

# Helper to compute a hash of the filenames (simple change detection)
file_names = [f.name for f in uploaded_files]
files_hash = hashlib.md5("".join(sorted(file_names)).encode()).hexdigest()

INDEX_DIR = "chroma_index"
HASH_FILE = os.path.join(INDEX_DIR, "files_hash.txt")

# Check if we need to rebuild the index
rebuild_index = True
if os.path.exists(INDEX_DIR) and os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        stored_hash = f.read().strip()
    if stored_hash == files_hash:
        rebuild_index = False

if rebuild_index:
    # Show progress in sidebar
    progress_placeholder.info("🔄 Rebuilding index – this may take a moment...")

    with st.spinner("Processing PDFs and building index..."):
        all_docs = []
        tmp_paths = []
        for pdf in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(pdf.getvalue())
            tmp.close()
            tmp_paths.append(tmp.name)

            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = pdf.name
            all_docs.extend(docs)

        # Clean up temp files
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

        # Check if any text was extracted
        if not all_docs:
            st.error("No pages could be loaded from the PDFs. The files might be empty or corrupted.")
            st.stop()

        # Filter out pages with empty content
        non_empty_docs = [d for d in all_docs if d.page_content.strip()]
        if not non_empty_docs:
            st.error("All PDF pages appear to be empty or contain no extractable text. Please check your files.")
            st.stop()
        else:
            st.success(f"✅ Loaded {len(non_empty_docs)} pages with text from {len(uploaded_files)} PDFs")

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=120
        )
        splits = text_splitter.split_documents(non_empty_docs)

        # Check if splitting produced any chunks
        if not splits:
            st.error("Text splitting produced no chunks. The document content might be too short or empty.")
            st.stop()

        # Create vectorstore
        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=INDEX_DIR
        )
        # Save hash for next time
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(HASH_FILE, "w") as f:
            f.write(files_hash)

    progress_placeholder.success("✅ Index built successfully!")
else:
    # Load existing index
    vectorstore = Chroma(
        persist_directory=INDEX_DIR,
        embedding_function=embeddings
    )
    progress_placeholder.info("♻️ Using existing index (no changes in PDFs)")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# ---------- Helper functions ----------
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone search query using the chat history for context. "
     "Return only the rewritten query, no extra text."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. You must answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n"
     "Do NOT use outside knowledge.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ---------- Session state for in-memory history (synced with JSON) ----------
if "chat_history_objects" not in st.session_state:
    st.session_state.chat_history_objects = {}

def get_history(session_id: str):
    """Return a ChatMessageHistory for the session, loading from JSON if needed."""
    if session_id not in st.session_state.chat_history_objects:
        history = ChatMessageHistory()
        for role, content in load_history(session_id):
            if role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_ai_message(content)
        st.session_state.chat_history_objects[session_id] = history
    return st.session_state.chat_history_objects[session_id]

history = get_history(session_id)

# ---------- Main chat area ----------
st.markdown("# 📄 RAG Q&A Pro")
st.markdown("Ask questions based on your uploaded PDFs. The system remembers your conversation per session.")

# Display chat messages (from history)
chat_container = st.container()
with chat_container:
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

# ---------- Question input ----------
user_q = st.chat_input("Ask a question...")

if user_q:
    # 1) Rewrite question with history
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )
    standalone_q = llm.invoke(rewrite_msgs).content.strip()

    # 2) Retrieve chunks
    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of scope — not found in provided documents."
    else:
        # 3) Build context string
        context_str = _join_docs(docs)
        qa_msgs = qa_prompt.format_messages(
            chat_history=history.messages,
            input=user_q,
            context=context_str
        )
        answer = llm.invoke(qa_msgs).content

    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.write(user_q)
    # Display assistant message
    with chat_container:
        with st.chat_message("assistant"):
            st.write(answer)

    # Save to JSON and update in-memory history
    save_message(session_id, "user", user_q)
    save_message(session_id, "assistant", answer)
    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # Debug panels (collapsible, placed below chat)
    with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📑 Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
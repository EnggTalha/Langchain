# ============================================
# Basic Streamlit UI - Document Chat with MongoDB
# ============================================

import streamlit as st
from main import (
    get_documents_text, 
    get_text_chunks, 
    get_vector_store,
    get_answer_with_memory,
    get_llm,
    MongoDBManager,
    ConversationMemoryManager
)
import os


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Document Chat",
    page_icon="💬",
    layout="centered"
)


# ============================================
# Initialize Session State
# ============================================

def init_session():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'docs_ready' not in st.session_state:
        st.session_state.docs_ready = False
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "user_001"
    
    if 'mongo_manager' not in st.session_state:
        try:
            st.session_state.mongo_manager = MongoDBManager()
        except:
            st.session_state.mongo_manager = None
    
    if 'memory_manager' not in st.session_state:
        try:
            llm = get_llm()
            st.session_state.memory_manager = ConversationMemoryManager(llm)
        except:
            st.session_state.memory_manager = None


# ============================================
# Sidebar
# ============================================

def sidebar():
    """Simple sidebar for file upload."""
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # User ID
        user_id = st.text_input("User ID:", value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        # File upload
        files = st.file_uploader(
            "Choose files:",
            type=['pdf', 'xlsx', 'csv', 'txt', 'docx'],
            accept_multiple_files=True
        )
        
        # Process button
        if files:
            st.write(f"**{len(files)} files selected**")
            
            if st.button("Process Documents", type="primary"):
                process_files(files)
        
        st.divider()
        
        # Status
        if st.session_state.docs_ready:
            st.success("✅ Documents ready")
        else:
            st.info("📤 Upload documents to start")
        
        st.divider()
        
        # Clear buttons
        if st.button("Clear Chat"):
            st.session_state.messages = []
            if st.session_state.memory_manager:
                st.session_state.memory_manager.clear_memory()
            st.rerun()
        
        if st.button("Clear History"):
            if st.session_state.mongo_manager:
                st.session_state.mongo_manager.clear_user_history(st.session_state.user_id)
                st.success("History cleared!")


# ============================================
# Process Documents
# ============================================

def process_files(files):
    """Process uploaded files."""
    with st.spinner("Processing..."):
        try:
            # Extract text
            text = get_documents_text(files)
            
            if not text.strip():
                st.error("No text extracted!")
                return
            
            # Create chunks
            chunks = get_text_chunks(text)
            
            # Build vector store
            get_vector_store(chunks)
            
            st.session_state.docs_ready = True
            st.success(f"✅ Processed {len(chunks)} chunks")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")


# ============================================
# Main Chat Interface
# ============================================

def main():
    """Main application."""
    init_session()
    sidebar()
    
    # Header
    st.title("💬 Document Chat")
    
    # Check if ready
    if not st.session_state.docs_ready:
        st.info("👈 Upload documents in the sidebar to get started")
        return
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = get_answer_with_memory(
                        user_question=prompt,
                        user_id=st.session_state.user_id,
                        mongo_manager=st.session_state.mongo_manager,
                        memory_manager=st.session_state.memory_manager
                    )
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ============================================
# Run App
# ============================================

if __name__ == "__main__":
    main()
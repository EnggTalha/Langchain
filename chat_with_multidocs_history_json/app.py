# ============================================
# Simple Streamlit UI - Multi-Format Document Chat
# ============================================
# A clean, easy-to-use interface for chatting with documents
# Supports: PDF, Excel, CSV, TXT, Word
# ============================================

import streamlit as st
from main import (
    get_documents_text, 
    get_text_chunks, 
    get_vector_store, 
    get_answer,
    load_conversation_history,
    clear_conversation_history
)
import os


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# Simple Custom CSS
# ============================================

st.markdown("""
    <style>
    /* Clean, minimal styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Message boxes */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* File info */
    .file-item {
        background: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        border: 1px solid #E0E0E0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# Initialize Session State
# ============================================

def initialize_session_state():
    """
    Initialize all session state variables.
    Session state persists data across Streamlit reruns.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'docs_processed' not in st.session_state:
        st.session_state.docs_processed = False
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "default_user"
    
    if 'file_stats' not in st.session_state:
        st.session_state.file_stats = {}


# ============================================
# Sidebar - Document Upload & Processing
# ============================================

def show_sidebar():
    """
    Display the sidebar with file upload and processing options.
    Supports multiple file formats: PDF, Excel, CSV, TXT, Word
    """
    with st.sidebar:
        st.markdown("## 📁 Upload Documents")
        
        # User ID input (for conversation history)
        user_id = st.text_input(
            "User ID (for saving history):",
            value=st.session_state.user_id,
            help="Enter your user ID to save conversation history"
        )
        st.session_state.user_id = user_id
        
        st.markdown("---")
        
        # File uploader - supports multiple formats
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=['pdf', 'xlsx', 'xls', 'csv', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Supported: PDF, Excel, CSV, TXT, Word"
        )
        
        # Show uploaded files
        if uploaded_files:
            st.markdown(f"**📄 {len(uploaded_files)} file(s) selected:**")
            for i, file in enumerate(uploaded_files, 1):
                file_size = len(file.getvalue()) / 1024  # KB
                file_type = file.name.split('.')[-1].upper()
                st.markdown(f"""
                    <div class="file-item">
                        {i}. <strong>{file.name}</strong><br>
                        <small>Type: {file_type} | Size: {file_size:.1f} KB</small>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Process button
            if st.button("⚙️ Process Documents", use_container_width=True, type="primary"):
                process_documents(uploaded_files)
        
        else:
            st.markdown("""
                <div class="info-box">
                    <strong>ℹ️ No files uploaded</strong><br>
                    Upload documents above to get started
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Status display
        if st.session_state.docs_processed:
            st.markdown("""
                <div class="success-box">
                    <strong>✅ Documents Ready</strong><br>
                    You can now ask questions!
                </div>
            """, unsafe_allow_html=True)
            
            # Show stats if available
            if st.session_state.file_stats:
                st.markdown("**📊 Statistics:**")
                stats = st.session_state.file_stats
                st.write(f"- Files: {stats.get('num_files', 0)}")
                st.write(f"- Chunks: {stats.get('num_chunks', 0)}")
                st.write(f"- Characters: {stats.get('text_length', 0):,}")
        
        st.markdown("---")
        
        # History management
        st.markdown("### 📜 History")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Load", use_container_width=True):
                load_history()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_history()


# ============================================
# Process Documents Function
# ============================================

def process_documents(uploaded_files):
    """
    Process uploaded documents: extract text, create chunks, build vector store.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    """
    try:
        with st.spinner("⏳ Processing documents..."):
            
            # Step 1: Extract text from all files
            st.info("📄 Step 1/3: Extracting text from documents...")
            raw_text = get_documents_text(uploaded_files)
            
            if not raw_text.strip():
                st.error("❌ No text could be extracted from the files!")
                return
            
            text_length = len(raw_text)
            st.success(f"✅ Extracted {text_length:,} characters")
            
            # Step 2: Create text chunks
            st.info("✂️ Step 2/3: Creating text chunks...")
            text_chunks = get_text_chunks(raw_text)
            st.success(f"✅ Created {len(text_chunks)} chunks")
            
            # Step 3: Build vector store
            st.info("🔍 Step 3/3: Building searchable index...")
            get_vector_store(text_chunks)
            st.success("✅ Vector store created!")
            
            # Update session state
            st.session_state.docs_processed = True
            st.session_state.file_stats = {
                'num_files': len(uploaded_files),
                'num_chunks': len(text_chunks),
                'text_length': text_length
            }
            
            st.success("🎉 Processing complete! You can now ask questions.")
            
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.info("💡 Please check your files and try again.")


# ============================================
# Chat Interface
# ============================================

def show_chat_interface():
    """
    Display the main chat interface where users can ask questions.
    Shows conversation history and handles new questions.
    """
    st.markdown("""
        <div class="app-header">
            <h1>💬 Document Chat Assistant</h1>
            <p>Ask questions about your uploaded documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if documents are processed
    if not st.session_state.docs_processed:
        st.markdown("""
            <div class="info-box">
                <h3>👋 Welcome!</h3>
                <p><strong>To get started:</strong></p>
                <ol>
                    <li>Upload your documents using the sidebar (PDF, Excel, CSV, TXT, Word)</li>
                    <li>Click "Process Documents"</li>
                    <li>Ask questions about your documents</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Check if vector store exists
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please process documents first!")
        st.session_state.docs_processed = False
        return
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation")
        
        # Clear button
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Show messages (most recent first)
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
            # User question
            st.markdown(f"""
                <div class="user-message">
                    <strong>❓ You:</strong><br>
                    {question}
                </div>
            """, unsafe_allow_html=True)
            
            # Assistant answer
            st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {answer}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Question input
    st.markdown("### ❓ Ask a Question")
    
    # Create input form
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the main topic? What are the key findings?",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submit = st.form_submit_button("🔍 Ask", use_container_width=True)
        with col2:
            # This button doesn't submit the form
            pass
    
    # Handle question submission
    if submit and user_question:
        handle_question(user_question)


# ============================================
# Handle User Question
# ============================================

def handle_question(question):
    """
    Process user question and display answer.
    
    Args:
        question: The user's question as a string
    """
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
        return
    
    try:
        with st.spinner("🔍 Searching documents and generating answer..."):
            # Get answer using the core logic
            answer = get_answer(question, st.session_state.user_id)
            
            # Add to chat history
            st.session_state.chat_history.append((question, answer))
            
            # Show success message
            st.success("✅ Answer generated!")
            
            # Rerun to display new message
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Try processing documents again or check your API key.")


# ============================================
# History Management
# ============================================

def load_history():
    """
    Load conversation history from file for the current user.
    """
    try:
        history = load_conversation_history(st.session_state.user_id)
        
        if history['conversations']:
            # Convert to chat history format
            st.session_state.chat_history = [
                (conv['question'], conv['answer']) 
                for conv in history['conversations']
            ]
            st.success(f"✅ Loaded {len(history['conversations'])} conversations!")
            st.rerun()
        else:
            st.info("ℹ️ No saved history found for this user.")
            
    except Exception as e:
        st.error(f"❌ Error loading history: {str(e)}")


def clear_history():
    """
    Clear conversation history for the current user.
    """
    if st.session_state.chat_history:
        # Confirm before clearing
        st.session_state.chat_history = []
        clear_conversation_history(st.session_state.user_id)
        st.success("✅ History cleared!")
        st.rerun()
    else:
        st.info("ℹ️ No history to clear.")


# ============================================
# Main Application
# ============================================

def main():
    """
    Main application entry point.
    Initializes session state and displays the interface.
    """
    # Initialize session state variables
    initialize_session_state()
    
    # Show sidebar
    show_sidebar()
    
    # Show main chat interface
    show_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <small>💡 Powered by Groq LLM & LangChain | Supports PDF, Excel, CSV, TXT, Word</small>
        </div>
    """, unsafe_allow_html=True)


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    main()
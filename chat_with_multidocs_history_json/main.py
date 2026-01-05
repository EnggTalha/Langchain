# ============================================
# Multi-Format Document QA System with History
# ============================================
# This system allows users to:
# 1. Upload multiple document formats (PDF, Excel, CSV, TXT, Word)
# 2. Ask questions about the uploaded documents
# 3. Maintain conversation history per user
# 
# Tutorial: Each function is well-documented with:
# - Purpose explanation
# - Parameter descriptions
# - Return value details
# - Usage examples where helpful
# ============================================

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Document reading libraries
from PyPDF2 import PdfReader  # For PDF files
import pandas as pd  # For Excel and CSV files
import docx  # For Word documents (python-docx library)

# LangChain utilities for text processing and vector storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Groq LLM for question answering
from langchain_groq import ChatGroq

# Embeddings for semantic search
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================
# Load Environment Variables
# ============================================
# Load API keys and configuration from .env file
# Required: GROQ_API_KEY=your_api_key_here

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the directory where user histories will be saved
HISTORY_DIR = "user_histories"

# Create the history directory if it doesn't exist
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


# ============================================
# TEXT EXTRACTION FUNCTIONS
# ============================================
# These functions handle extracting text from different file formats
# Each function takes a file object and returns extracted text as a string

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file: File object or path to PDF file
        
    Returns:
        str: Extracted text from all pages
        
    Example:
        text = extract_text_from_pdf("document.pdf")
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:  # Only add if text was extracted
                text += f"\n--- Page {page_num} ---\n{page_text}"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text


def extract_text_from_excel(excel_file) -> str:
    """
    Extract text from an Excel file (.xlsx, .xls).
    Reads all sheets and converts them to text format.
    
    Args:
        excel_file: File object or path to Excel file
        
    Returns:
        str: Formatted text with sheet names and data
        
    Example:
        text = extract_text_from_excel("data.xlsx")
    """
    text = ""
    try:
        # Read all sheets from the Excel file
        excel_data = pd.read_excel(excel_file, sheet_name=None)  # None reads all sheets
        
        for sheet_name, df in excel_data.items():
            text += f"\n\n=== Sheet: {sheet_name} ===\n"
            # Convert dataframe to string with proper formatting
            text += df.to_string(index=False)
            text += "\n"
    except Exception as e:
        print(f"Error reading Excel: {e}")
    return text


def extract_text_from_csv(csv_file) -> str:
    """
    Extract text from a CSV file.
    
    Args:
        csv_file: File object or path to CSV file
        
    Returns:
        str: Formatted text representation of CSV data
        
    Example:
        text = extract_text_from_csv("data.csv")
    """
    text = ""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        text += "\n=== CSV Data ===\n"
        # Convert dataframe to string
        text += df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return text


def extract_text_from_txt(txt_file) -> str:
    """
    Extract text from a plain text file.
    
    Args:
        txt_file: File object or path to text file
        
    Returns:
        str: Raw text content
        
    Example:
        text = extract_text_from_txt("notes.txt")
    """
    text = ""
    try:
        # Handle both file objects and file paths
        if isinstance(txt_file, str):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = txt_file.read().decode('utf-8')
    except Exception as e:
        print(f"Error reading TXT: {e}")
    return text


def extract_text_from_word(word_file) -> str:
    """
    Extract text from a Word document (.docx).
    
    Args:
        word_file: File object or path to Word file
        
    Returns:
        str: Text from all paragraphs in the document
        
    Note:
        Requires python-docx library: pip install python-docx
        
    Example:
        text = extract_text_from_word("report.docx")
    """
    text = ""
    try:
        # Read Word document
        doc = docx.Document(word_file)
        text += "\n=== Word Document ===\n"
        # Extract text from each paragraph
        for para in doc.paragraphs:
            if para.text.strip():  # Only add non-empty paragraphs
                text += para.text + "\n"
    except Exception as e:
        print(f"Error reading Word document: {e}")
    return text


# ============================================
# UNIFIED DOCUMENT TEXT EXTRACTION
# ============================================

def get_documents_text(uploaded_files) -> str:
    """
    Extract text from multiple uploaded files of various formats.
    Automatically detects file type and uses appropriate extraction method.
    
    Supported formats:
    - PDF (.pdf)
    - Excel (.xlsx, .xls)
    - CSV (.csv)
    - Text (.txt)
    - Word (.docx)
    
    Args:
        uploaded_files: List of file objects with .name attribute
        
    Returns:
        str: Combined text from all uploaded files
        
    Example:
        files = [pdf_file, excel_file, txt_file]
        all_text = get_documents_text(files)
    """
    combined_text = ""
    
    for file in uploaded_files:
        # Get file extension to determine type
        file_extension = os.path.splitext(file.name)[1].lower()
        
        print(f"Processing: {file.name}")
        
        # Route to appropriate extraction function based on file type
        if file_extension == '.pdf':
            combined_text += extract_text_from_pdf(file)
        elif file_extension in ['.xlsx', '.xls']:
            combined_text += extract_text_from_excel(file)
        elif file_extension == '.csv':
            combined_text += extract_text_from_csv(file)
        elif file_extension == '.txt':
            combined_text += extract_text_from_txt(file)
        elif file_extension == '.docx':
            combined_text += extract_text_from_word(file)
        else:
            print(f"Unsupported file format: {file_extension}")
            continue
        
        # Add separator between different files
        combined_text += "\n\n" + "="*50 + "\n\n"
    
    return combined_text


# ============================================
# TEXT CHUNKING FOR VECTOR STORAGE
# ============================================

def get_text_chunks(text: str) -> List[str]:
    """
    Split large text into smaller, overlapping chunks.
    This is crucial for:
    1. Fitting within embedding model limits
    2. Maintaining context across chunk boundaries
    3. Improving semantic search accuracy
    
    Args:
        text: Large text string to be split
        
    Returns:
        List[str]: List of text chunks
        
    Chunking Strategy:
    - chunk_size: 800 characters per chunk (optimal for most models)
    - chunk_overlap: 150 characters overlap to preserve context
    - Separators: Prioritize splitting at paragraph/sentence boundaries
    
    Example:
        chunks = get_text_chunks(large_document_text)
        print(f"Created {len(chunks)} chunks")
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Maximum characters per chunk
        chunk_overlap=150,  # Overlap to maintain context between chunks
        length_function=len,
        # Split at natural boundaries (paragraph > sentence > word > character)
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ============================================
# VECTOR STORE CREATION
# ============================================

def get_vector_store(text_chunks: List[str]) -> None:
    """
    Convert text chunks into vector embeddings and store them in FAISS index.
    
    What happens:
    1. Each text chunk is converted to a numerical vector (embedding)
    2. These vectors capture semantic meaning of the text
    3. FAISS efficiently stores and indexes these vectors for fast retrieval
    4. The index is saved to disk for later use
    
    Args:
        text_chunks: List of text chunks to embed and store
        
    Returns:
        None (saves to disk at "faiss_index" directory)
        
    Technical Details:
    - Model: sentence-transformers/all-MiniLM-L6-v2
      (Fast, efficient, good for general text)
    - Embeddings are normalized for better cosine similarity
    - FAISS uses efficient nearest neighbor search
    
    Example:
        chunks = get_text_chunks(text)
        get_vector_store(chunks)
        print("Vector store created successfully!")
    """
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' for GPU)
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
    )

    # Create FAISS vector store from text chunks
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    # Save to disk for later retrieval
    vector_store.save_local("faiss_index")
    print(f"Vector store created with {len(text_chunks)} chunks")


# ============================================
# LLM INITIALIZATION
# ============================================

def get_llm():
    """
    Initialize and return the Groq LLM instance.
    
    LLM Configuration:
    - Model: llama-3.1-8b-instant (fast, accurate, cost-effective)
    - Temperature: 0.1 (low = more deterministic/accurate responses)
    
    Returns:
        ChatGroq: Configured language model instance
        
    Note:
        Requires GROQ_API_KEY in environment variables
        
    Example:
        llm = get_llm()
        response = llm.invoke("What is AI?")
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1  # Lower = more factual, Higher = more creative
    )
    return llm


# ============================================
# CONVERSATION HISTORY MANAGEMENT
# ============================================

def save_conversation_history(user_id: str, question: str, answer: str) -> None:
    """
    Save a Q&A pair to the user's conversation history file.
    
    What it does:
    1. Loads existing history for the user (if any)
    2. Adds new Q&A with timestamp
    3. Saves back to JSON file
    
    Args:
        user_id: Unique identifier for the user (e.g., "user_123")
        question: The question asked by the user
        answer: The answer provided by the system
        
    Returns:
        None (saves to disk)
        
    File Structure:
        user_histories/
            └── user_123.json
                {
                    "user_id": "user_123",
                    "conversations": [
                        {
                            "timestamp": "2024-01-15 10:30:45",
                            "question": "What is the revenue?",
                            "answer": "The revenue is $1M."
                        }
                    ]
                }
    
    Example:
        save_conversation_history("user_123", "What is AI?", "AI is...")
    """
    # Define file path for this user's history
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    
    # Load existing history or create new
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {
            "user_id": user_id,
            "conversations": []
        }
    
    # Create new conversation entry
    conversation_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    
    # Add to history
    history["conversations"].append(conversation_entry)
    
    # Save back to file
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"Conversation saved for user: {user_id}")


def load_conversation_history(user_id: str) -> Dict[str, Any]:
    """
    Load the complete conversation history for a specific user.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        dict: User's conversation history, or empty history if none exists
        
    Example:
        history = load_conversation_history("user_123")
        print(f"User has {len(history['conversations'])} conversations")
        
        for conv in history['conversations']:
            print(f"Q: {conv['question']}")
            print(f"A: {conv['answer']}")
    """
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "user_id": user_id,
            "conversations": []
        }


def clear_conversation_history(user_id: str) -> None:
    """
    Delete all conversation history for a specific user.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        None
        
    Example:
        clear_conversation_history("user_123")
        print("History cleared!")
    """
    history_file = os.path.join(HISTORY_DIR, f"{user_id}.json")
    
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"History cleared for user: {user_id}")
    else:
        print(f"No history found for user: {user_id}")


# ============================================
# QUESTION ANSWERING SYSTEM
# ============================================

def get_answer(user_question: str, user_id: str = "default_user") -> str:
    """
    Answer a user's question based on uploaded documents.
    Also saves the Q&A to conversation history.
    
    Process Flow:
    1. Load the FAISS vector store
    2. Convert user question to embedding
    3. Find most relevant document chunks (semantic search)
    4. Create prompt with context and question
    5. Get answer from LLM
    6. Save to conversation history
    7. Return answer
    
    Args:
        user_question: The question to answer
        user_id: Unique identifier for the user (for history tracking)
        
    Returns:
        str: The answer based on document context
        
    Search Strategy:
    - Uses MMR (Maximal Marginal Relevance) for diverse, relevant results
    - Balances relevance with diversity to avoid redundant information
    - Fallback to similarity search if MMR fails
    
    Example:
        answer = get_answer(
            "What is the total revenue?", 
            user_id="user_123"
        )
        print(answer)
    """
    # Initialize embeddings (same as used during vector store creation)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load the vector store from disk
    try:
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        error_msg = "Error: Please upload and process documents first!"
        print(f"Vector store loading error: {e}")
        return error_msg

    # ============================================
    # SEMANTIC SEARCH: Find Relevant Chunks
    # ============================================
    try:
        # MMR (Maximal Marginal Relevance) Search
        # Balances relevance with diversity
        docs = db.max_marginal_relevance_search(
            user_question, 
            k=6,  # Number of documents to retrieve
            fetch_k=20,  # Fetch 20 candidates, then select best 6
            lambda_mult=0.5  # 0=relevance only, 1=diversity only, 0.5=balanced
        )
    except:
        # Fallback: Standard similarity search with score filtering
        docs_with_scores = db.similarity_search_with_score(user_question, k=10)
        
        # Filter by relevance score (lower = more similar for cosine distance)
        filtered_docs = [doc for doc, score in docs_with_scores if score < 0.7]
        
        # Use filtered docs or top 6 if no good matches
        docs = filtered_docs[:6] if filtered_docs else [doc for doc, _ in docs_with_scores[:6]]
    
    # ============================================
    # PREPARE CONTEXT FROM RETRIEVED CHUNKS
    # ============================================
    # Format context with chunk numbering for clarity
    context = "\n\n".join([
        f"[Document Chunk {i+1}]:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    # ============================================
    # CREATE PROMPT TEMPLATE
    # ============================================
    # This prompt instructs the LLM on how to answer
    prompt_template = """
    SYSTEM PROMPT:
    You are a precise document analysis assistant. Your ONLY source of information is the provided document context below.
    
    CRITICAL RULES:
    1. Answer ONLY using information explicitly stated in the provided context.
    2. Be ACCURATE - quote or paraphrase directly from the context.
    3. Be CONCISE - provide direct answers without unnecessary elaboration.
    4. If the answer is NOT in the context, respond EXACTLY: "The answer is not available in the uploaded documents."
    5. Do NOT use external knowledge, assumptions, or general information.
    6. Do NOT make inferences beyond what is explicitly stated.
    7. If the context is insufficient, state that clearly.
    8. When referencing data from tables (Excel/CSV), present it clearly.
    
    Context from Documents:
    {context}

    User Question:
    {question}

    Answer (based ONLY on document context):
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # ============================================
    # GENERATE ANSWER USING LLM
    # ============================================
    llm = get_llm()
    formatted_prompt = prompt.format(context=context, question=user_question)
    response = llm.invoke(formatted_prompt)
    
    answer = response.content
    
    # ============================================
    # SAVE TO CONVERSATION HISTORY
    # ============================================
    save_conversation_history(user_id, user_question, answer)
    
    return answer


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage of the Document QA System.
    
    Workflow:
    1. Upload documents (PDF, Excel, CSV, TXT, Word)
    2. System extracts and processes text
    3. Creates searchable vector store
    4. Ask questions
    5. Get answers with conversation history saved
    """
    
    print("="*50)
    print("Multi-Format Document QA System")
    print("="*50)
    
    # Example: Process documents
    # In a real application, these would come from file upload
    # uploaded_files = [file1, file2, file3]
    # text = get_documents_text(uploaded_files)
    # chunks = get_text_chunks(text)
    # get_vector_store(chunks)
    
    # Example: Ask questions
    # user_id = "user_123"
    # answer = get_answer("What is the total revenue?", user_id)
    # print(f"Answer: {answer}")
    
    # Example: View history
    # history = load_conversation_history(user_id)
    # print(f"Total conversations: {len(history['conversations'])}")
    
    print("\nSystem ready! Upload documents to begin.")
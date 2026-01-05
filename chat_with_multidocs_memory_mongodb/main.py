# ============================================
# Multi-Format Document QA System with MongoDB & Memory
# ============================================
# Enhanced features:
# 1. MongoDB storage for conversation history
# 2. LangChain ConversationBufferMemory with summarization
# 3. Multi-format document support
# ============================================

import os
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# MongoDB
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Document reading libraries
from PyPDF2 import PdfReader
import pandas as pd
import docx

# LangChain utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain Memory
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryBufferMemory
# ============================================
# Load Environment Variables
# ============================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# ============================================
# MongoDB Connection
# ============================================
class MongoDBManager:
    """
    Manages MongoDB connections and operations for conversation history.
    
    Database: GENI_CLASS
    Collection: GROQ_CHAT
    """
    
    def __init__(self, uri: str = MONGODB_URI):
        """
        Initialize MongoDB connection.
        
        Args:
            uri: MongoDB connection string
        """
        try:
            self.client = MongoClient(uri)
            # Test connection
            self.client.admin.command('ping')
            print("✓ MongoDB connection successful!")
            
            # Set database and collection
            self.db = self.client["GENI_CLASS"]
            self.collection = self.db["GROQ_CHAT"]
            
            # Create indexes for better query performance
            self.collection.create_index("user_id")
            self.collection.create_index("timestamp")
            
        except ConnectionFailure as e:
            print(f"✗ MongoDB connection failed: {e}")
            raise
    
    def save_conversation(self, user_id: str, question: str, answer: str, 
                         context_chunks: int = 0, metadata: Dict = None) -> None:
        """
        Save a conversation to MongoDB.
        
        Args:
            user_id: Unique user identifier
            question: User's question
            answer: System's answer
            context_chunks: Number of document chunks used
            metadata: Additional metadata (optional)
        """
        conversation_doc = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "context_chunks_used": context_chunks,
            "metadata": metadata or {}
        }
        
        result = self.collection.insert_one(conversation_doc)
        print(f"✓ Conversation saved with ID: {result.inserted_id}")
    
    def load_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Load conversation history for a user.
        
        Args:
            user_id: Unique user identifier
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation documents
        """
        conversations = list(
            self.collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
        )
        
        # Convert ObjectId to string for JSON serialization
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
            
        return conversations
    
    def get_recent_context(self, user_id: str, limit: int = 5) -> str:
        """
        Get recent conversation context for memory.
        
        Args:
            user_id: Unique user identifier
            limit: Number of recent conversations
            
        Returns:
            Formatted conversation history string
        """
        conversations = self.load_user_history(user_id, limit)
        
        context = ""
        for conv in reversed(conversations):  # Oldest first
            context += f"Human: {conv['question']}\n"
            context += f"Assistant: {conv['answer']}\n\n"
        
        return context
    
    def clear_user_history(self, user_id: str) -> int:
        """
        Delete all conversations for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Number of deleted documents
        """
        result = self.collection.delete_many({"user_id": user_id})
        print(f"✓ Deleted {result.deleted_count} conversations for user: {user_id}")
        return result.deleted_count
    
    def get_conversation_stats(self, user_id: str) -> Dict:
        """
        Get statistics about user's conversations.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary with stats
        """
        total = self.collection.count_documents({"user_id": user_id})
        
        if total == 0:
            return {"total_conversations": 0}
        
        first_conv = self.collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", 1)]
        )
        
        last_conv = self.collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )
        
        return {
            "total_conversations": total,
            "first_conversation": first_conv["timestamp"],
            "last_conversation": last_conv["timestamp"]
        }


# ============================================
# Memory Manager with Summarization
# ============================================
class ConversationMemoryManager:
    """
    Manages conversation memory with automatic summarization.
    
    Uses ConversationSummaryBufferMemory to:
    1. Keep recent messages in full
    2. Summarize older messages to save tokens
    """
    
    def __init__(self, llm, max_token_limit: int = 500):
        """
        Initialize memory manager.
        
        Args:
            llm: Language model for summarization
            max_token_limit: Maximum tokens before summarization
        """
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def add_interaction(self, question: str, answer: str):
        """
        Add a Q&A pair to memory.
        
        Args:
            question: User's question
            answer: System's answer
        """
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )
    
    def get_memory_context(self) -> str:
        """
        Get the current memory context (recent + summary).
        
        Returns:
            Formatted memory string
        """
        memory_vars = self.memory.load_memory_variables({})
        return str(memory_vars.get("chat_history", ""))
    
    def clear_memory(self):
        """Clear all memory."""
        self.memory.clear()
    
    def get_buffer_string(self) -> str:
        """
        Get the memory as a formatted string.
        
        Returns:
            Human-readable memory content
        """
        try:
            # Try to get messages from memory
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            
            # Format messages into a readable string
            if not messages:
                return ""
            
            formatted = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    # LangChain message objects
                    role = "Human" if msg.type == "human" else "Assistant"
                    formatted.append(f"{role}: {msg.content}")
                else:
                    # String format
                    formatted.append(str(msg))
            
            return "\n".join(formatted)
        except Exception as e:
            print(f"Error getting buffer string: {e}")
            return ""


# ============================================
# TEXT EXTRACTION FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num} ---\n{page_text}"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text


def extract_text_from_excel(excel_file) -> str:
    """Extract text from Excel file."""
    text = ""
    try:
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        for sheet_name, df in excel_data.items():
            text += f"\n\n=== Sheet: {sheet_name} ===\n"
            text += df.to_string(index=False)
            text += "\n"
    except Exception as e:
        print(f"Error reading Excel: {e}")
    return text


def extract_text_from_csv(csv_file) -> str:
    """Extract text from CSV file."""
    text = ""
    try:
        df = pd.read_csv(csv_file)
        text += "\n=== CSV Data ===\n"
        text += df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return text


def extract_text_from_txt(txt_file) -> str:
    """Extract text from plain text file."""
    text = ""
    try:
        if isinstance(txt_file, str):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = txt_file.read().decode('utf-8')
    except Exception as e:
        print(f"Error reading TXT: {e}")
    return text


def extract_text_from_word(word_file) -> str:
    """Extract text from Word document."""
    text = ""
    try:
        doc = docx.Document(word_file)
        text += "\n=== Word Document ===\n"
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        print(f"Error reading Word document: {e}")
    return text


def get_documents_text(uploaded_files) -> str:
    """
    Extract text from multiple uploaded files.
    
    Supported formats: PDF, Excel, CSV, TXT, Word
    """
    combined_text = ""
    
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        print(f"Processing: {file.name}")
        
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
        
        combined_text += "\n\n" + "="*50 + "\n\n"
    
    return combined_text


# ============================================
# TEXT CHUNKING
# ============================================

def get_text_chunks(text: str) -> List[str]:
    """
    Split text into chunks for embedding.
    
    Args:
        text: Large text string
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ============================================
# VECTOR STORE
# ============================================

def get_vector_store(text_chunks: List[str]) -> None:
    """
    Create FAISS vector store from text chunks.
    
    Args:
        text_chunks: List of text chunks to embed
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")
    print(f"✓ Vector store created with {len(text_chunks)} chunks")


# ============================================
# LLM
# ============================================

def get_llm():
    """Initialize Groq LLM."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )
    return llm


# ============================================
# QUESTION ANSWERING WITH MEMORY
# ============================================

def get_answer_with_memory(
    user_question: str,
    user_id: str = "default_user",
    mongo_manager: MongoDBManager = None,
    memory_manager: ConversationMemoryManager = None
) -> str:
    """
    Answer questions using document context and conversation memory.
    
    Args:
        user_question: The question to answer
        user_id: Unique user identifier
        mongo_manager: MongoDB manager instance
        memory_manager: Memory manager instance
        
    Returns:
        Answer string
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load vector store
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

    # Semantic search
    try:
        docs = db.max_marginal_relevance_search(
            user_question,
            k=6,
            fetch_k=20,
            lambda_mult=0.5
        )
    except:
        docs_with_scores = db.similarity_search_with_score(user_question, k=10)
        filtered_docs = [doc for doc, score in docs_with_scores if score < 0.7]
        docs = filtered_docs[:6] if filtered_docs else [doc for doc, _ in docs_with_scores[:6]]
    
    # Prepare context
    context = "\n\n".join([
        f"[Document Chunk {i+1}]:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    
    # Get conversation memory if available
    memory_context = ""
    if memory_manager:
        memory_context = memory_manager.get_buffer_string()
    
    # Create prompt with memory
    prompt_template = """
    SYSTEM PROMPT:
    You are a precise document analysis assistant with conversation memory.
    
    CRITICAL RULES:
    1. Answer using information from the provided document context.
    2. Consider previous conversation context to provide coherent answers.
    3. Be ACCURATE and CONCISE.
    4. If the answer is NOT in the documents, respond: "The answer is not available in the uploaded documents."
    5. Do NOT use external knowledge beyond the documents.
    
    Previous Conversation Context:
    {memory_context}
    
    Document Context:
    {context}

    Current Question:
    {question}

    Answer (based on documents and conversation context):
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["memory_context", "context", "question"]
    )
    
    # Generate answer
    llm = get_llm()
    formatted_prompt = prompt.format(
        memory_context=memory_context,
        context=context,
        question=user_question
    )
    response = llm.invoke(formatted_prompt)
    answer = response.content
    
    # Save to memory
    if memory_manager:
        memory_manager.add_interaction(user_question, answer)
    
    # Save to MongoDB
    if mongo_manager:
        mongo_manager.save_conversation(
            user_id=user_id,
            question=user_question,
            answer=answer,
            context_chunks=len(docs),
            metadata={"has_memory": memory_manager is not None}
        )
    
    return answer


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("Document QA with MongoDB & Memory")
    print("="*50)
    
    # Initialize managers
    mongo_manager = MongoDBManager()
    llm = get_llm()
    memory_manager = ConversationMemoryManager(llm, max_token_limit=500)
    
    # Example workflow:
    # 1. Upload and process documents
    # uploaded_files = [file1, file2]
    # text = get_documents_text(uploaded_files)
    # chunks = get_text_chunks(text)
    # get_vector_store(chunks)
    
    # 2. Ask questions with memory
    # user_id = "user_123"
    # answer = get_answer_with_memory(
    #     "What is the revenue?",
    #     user_id=user_id,
    #     mongo_manager=mongo_manager,
    #     memory_manager=memory_manager
    # )
    
    # 3. View stats
    # stats = mongo_manager.get_conversation_stats(user_id)
    # print(f"Stats: {stats}")
    
    # 4. View memory
    # print(f"Memory: {memory_manager.get_buffer_string()}")
    
    print("\n✓ System ready!")
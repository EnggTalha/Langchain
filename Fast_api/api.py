# ============================================
# FastAPI Document QA System with MongoDB & Memory
# Port: 8005
# ============================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from datetime import datetime
from io import BytesIO

# Import your existing modules
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from PyPDF2 import PdfReader
import pandas as pd
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationSummaryBufferMemory

# ============================================
# Load Environment Variables
# ============================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# ============================================
# FastAPI App Initialization
# ============================================
app = FastAPI(
    title="Document QA System",
    description="Multi-format document QA with MongoDB persistence and conversation memory",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Pydantic Models
# ============================================
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default_user"

class QuestionResponse(BaseModel):
    answer: str
    context_chunks_used: int
    timestamp: datetime
    user_id: str

class StatsResponse(BaseModel):
    total_conversations: int
    first_conversation: Optional[datetime] = None
    last_conversation: Optional[datetime] = None

class HealthResponse(BaseModel):
    status: str
    mongodb_connected: bool
    vector_store_exists: bool
    timestamp: datetime

# ============================================
# MongoDB Manager Class
# ============================================
class MongoDBManager:
    def __init__(self, uri: str = MONGODB_URI):
        try:
            self.client = MongoClient(uri)
            self.client.admin.command('ping')
            self.db = self.client["GENI_CLASS"]
            self.collection = self.db["GROQ_CHAT"]
            self.collection.create_index("user_id")
            self.collection.create_index("timestamp")
            print("✓ MongoDB connection successful!")
        except ConnectionFailure as e:
            print(f"✗ MongoDB connection failed: {e}")
            raise
    
    def save_conversation(self, user_id: str, question: str, answer: str, 
                         context_chunks: int = 0, metadata: dict = None):
        conversation_doc = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "context_chunks_used": context_chunks,
            "metadata": metadata or {}
        }
        result = self.collection.insert_one(conversation_doc)
        return str(result.inserted_id)
    
    def load_user_history(self, user_id: str, limit: int = 50):
        conversations = list(
            self.collection.find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(limit)
        )
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
        return conversations
    
    def get_conversation_stats(self, user_id: str):
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
    
    def clear_user_history(self, user_id: str):
        result = self.collection.delete_many({"user_id": user_id})
        return result.deleted_count

# ============================================
# Memory Manager Class
# ============================================
class ConversationMemoryManager:
    def __init__(self, llm, max_token_limit: int = 500):
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def add_interaction(self, question: str, answer: str):
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )
    
    def get_buffer_string(self):
        try:
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            if not messages:
                return ""
            formatted = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    role = "Human" if msg.type == "human" else "Assistant"
                    formatted.append(f"{role}: {msg.content}")
                else:
                    formatted.append(str(msg))
            return "\n".join(formatted)
        except Exception as e:
            print(f"Error getting buffer string: {e}")
            return ""
    
    def clear_memory(self):
        self.memory.clear()

# ============================================
# Text Extraction Functions
# ============================================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num} ---\n{page_text}"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_excel(excel_bytes: bytes) -> str:
    text = ""
    try:
        excel_data = pd.read_excel(BytesIO(excel_bytes), sheet_name=None)
        for sheet_name, df in excel_data.items():
            text += f"\n\n=== Sheet: {sheet_name} ===\n"
            text += df.to_string(index=False)
            text += "\n"
    except Exception as e:
        print(f"Error reading Excel: {e}")
    return text

def extract_text_from_csv(csv_bytes: bytes) -> str:
    text = ""
    try:
        df = pd.read_csv(BytesIO(csv_bytes))
        text += "\n=== CSV Data ===\n"
        text += df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return text

def extract_text_from_txt(txt_bytes: bytes) -> str:
    text = ""
    try:
        text = txt_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error reading TXT: {e}")
    return text

def extract_text_from_word(word_bytes: bytes) -> str:
    text = ""
    try:
        doc = docx.Document(BytesIO(word_bytes))
        text += "\n=== Word Document ===\n"
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        print(f"Error reading Word document: {e}")
    return text

def get_documents_text(files: List[UploadFile]) -> str:
    combined_text = ""
    for file in files:
        file_extension = os.path.splitext(file.filename)[1].lower()
        print(f"Processing: {file.filename}")
        
        file_content = file.file.read()
        
        if file_extension == '.pdf':
            combined_text += extract_text_from_pdf(file_content)
        elif file_extension in ['.xlsx', '.xls']:
            combined_text += extract_text_from_excel(file_content)
        elif file_extension == '.csv':
            combined_text += extract_text_from_csv(file_content)
        elif file_extension == '.txt':
            combined_text += extract_text_from_txt(file_content)
        elif file_extension == '.docx':
            combined_text += extract_text_from_word(file_content)
        else:
            print(f"Unsupported file format: {file_extension}")
            continue
        
        combined_text += "\n\n" + "="*50 + "\n\n"
        file.file.seek(0)  # Reset file pointer
    
    return combined_text

# ============================================
# Vector Store Functions
# ============================================
def get_text_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks: List[str]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print(f"✓ Vector store created with {len(text_chunks)} chunks")

def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

# ============================================
# Global Variables
# ============================================
mongo_manager = None
memory_managers = {}  # Store memory per user

# ============================================
# Startup Event
# ============================================
@app.on_event("startup")
async def startup_event():
    global mongo_manager
    try:
        mongo_manager = MongoDBManager()
        print("✓ FastAPI server started on port 8005")
    except Exception as e:
        print(f"✗ Failed to initialize MongoDB: {e}")

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    mongodb_ok = mongo_manager is not None
    vector_store_exists = os.path.exists("faiss_index")
    
    return HealthResponse(
        status="healthy" if mongodb_ok else "degraded",
        mongodb_connected=mongodb_ok,
        vector_store_exists=vector_store_exists,
        timestamp=datetime.now()
    )

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple documents.
    Supported formats: PDF, Excel, CSV, TXT, Word
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Extract text from all files
        combined_text = get_documents_text(files)
        
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from files")
        
        # Create chunks and vector store
        chunks = get_text_chunks(combined_text)
        get_vector_store(chunks)
        
        return JSONResponse(content={
            "message": "Documents processed successfully",
            "files_processed": len(files),
            "chunks_created": len(chunks),
            "filenames": [f.filename for f in files]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents with conversation memory.
    """
    try:
        # Check if vector store exists
        if not os.path.exists("faiss_index"):
            raise HTTPException(
                status_code=400,
                detail="Please upload documents first using /upload-documents endpoint"
            )
        
        # Get or create memory manager for user
        if request.user_id not in memory_managers:
            llm = get_llm()
            memory_managers[request.user_id] = ConversationMemoryManager(llm, max_token_limit=500)
        
        memory_manager = memory_managers[request.user_id]
        
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Semantic search
        try:
            docs = db.max_marginal_relevance_search(
                request.question,
                k=6,
                fetch_k=20,
                lambda_mult=0.5
            )
        except:
            docs_with_scores = db.similarity_search_with_score(request.question, k=10)
            filtered_docs = [doc for doc, score in docs_with_scores if score < 0.7]
            docs = filtered_docs[:6] if filtered_docs else [doc for doc, _ in docs_with_scores[:6]]
        
        # Prepare context
        context = "\n\n".join([
            f"[Document Chunk {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        # Get memory context
        memory_context = memory_manager.get_buffer_string()
        
        # Create prompt
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
            question=request.question
        )
        response = llm.invoke(formatted_prompt)
        answer = response.content
        
        # Save to memory
        memory_manager.add_interaction(request.question, answer)
        
        # Save to MongoDB
        if mongo_manager:
            mongo_manager.save_conversation(
                user_id=request.user_id,
                question=request.question,
                answer=answer,
                context_chunks=len(docs),
                metadata={"has_memory": True}
            )
        
        return QuestionResponse(
            answer=answer,
            context_chunks_used=len(docs),
            timestamp=datetime.now(),
            user_id=request.user_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Get conversation history for a user"""
    try:
        if not mongo_manager:
            raise HTTPException(status_code=500, detail="MongoDB not connected")
        
        history = mongo_manager.load_user_history(user_id, limit)
        return JSONResponse(content={"user_id": user_id, "conversations": history})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/stats/{user_id}", response_model=StatsResponse)
async def get_stats(user_id: str):
    """Get conversation statistics for a user"""
    try:
        if not mongo_manager:
            raise HTTPException(status_code=500, detail="MongoDB not connected")
        
        stats = mongo_manager.get_conversation_stats(user_id)
        return StatsResponse(**stats)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear conversation history for a user"""
    try:
        if not mongo_manager:
            raise HTTPException(status_code=500, detail="MongoDB not connected")
        
        deleted_count = mongo_manager.clear_user_history(user_id)
        
        # Also clear memory
        if user_id in memory_managers:
            memory_managers[user_id].clear_memory()
        
        return JSONResponse(content={
            "message": "History cleared successfully",
            "conversations_deleted": deleted_count,
            "user_id": user_id
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    """Clear conversation memory for a user (without deleting MongoDB history)"""
    try:
        if user_id in memory_managers:
            memory_managers[user_id].clear_memory()
            return JSONResponse(content={
                "message": "Memory cleared successfully",
                "user_id": user_id
            })
        else:
            return JSONResponse(content={
                "message": "No memory found for user",
                "user_id": user_id
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Get current conversation memory for a user"""
    try:
        if user_id in memory_managers:
            memory_content = memory_managers[user_id].get_buffer_string()
            return JSONResponse(content={
                "user_id": user_id,
                "memory": memory_content
            })
        else:
            return JSONResponse(content={
                "user_id": user_id,
                "memory": "No memory found"
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory: {str(e)}")

# ============================================
# Run Server
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8005,
        reload=True
    )
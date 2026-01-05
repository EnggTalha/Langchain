# FastAPI Document QA System – Postman Testing Guide

This document explains how to test the **FastAPI Document QA System** (`api.py`) using **Postman**.

---

## Server Details

**File name:** `api.py`  
**Run command:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8005 --reload
Base URL:

cpp
Copy code
http://127.0.0.1:8005
1. Health Check
Endpoint: /
Method: GET

Purpose:
Verify that FastAPI, MongoDB, and vector store status are working.

Postman Setup:

Method: GET

URL: http://127.0.0.1:8005/

Response Example:

json
Copy code
{
  "status": "healthy",
  "mongodb_connected": true,
  "vector_store_exists": false,
  "timestamp": "2026-01-02T18:20:00"
}
2. Upload Documents
Endpoint: /upload-documents
Method: POST

Purpose:
Upload documents and create FAISS vector store.

Supported Formats:
PDF, DOCX, XLSX, CSV, TXT

Postman Setup:

Method: POST

URL: http://127.0.0.1:8005/upload-documents

Body → form-data

Key	Type	Value
files	File	Select one or multiple files

Response Example:

json
Copy code
{
  "message": "Documents processed successfully",
  "files_processed": 2,
  "chunks_created": 42,
  "filenames": ["policy.pdf", "rules.docx"]
}
⚠️ You MUST upload documents before asking questions.

3. Ask a Question
Endpoint: /ask
Method: POST

Purpose:
Ask questions from uploaded documents with conversation memory.

Postman Setup:

Method: POST

URL: http://127.0.0.1:8005/ask

Body → raw → JSON

json
Copy code
{
  "question": "What is the main topic of the document?",
  "user_id": "user123"
}
Response Example:

json
Copy code
{
  "answer": "The document mainly discusses company policies and procedures.",
  "context_chunks_used": 6,
  "timestamp": "2026-01-02T18:22:00",
  "user_id": "user123"
}
4. Get Conversation History
Endpoint: /history/{user_id}
Method: GET

Purpose:
Retrieve previous Q&A stored in MongoDB.

Postman Setup:

Method: GET

URL:

arduino
Copy code
http://127.0.0.1:8005/history/user123
Response Example:

json
Copy code
{
  "user_id": "user123",
  "conversations": [
    {
      "_id": "65a9c123abc",
      "question": "What is the main topic?",
      "answer": "The document mainly discusses company policies.",
      "context_chunks_used": 6,
      "timestamp": "2026-01-02T18:22:00",
      "metadata": {
        "has_memory": true
      }
    }
  ]
}
5. Get Conversation Statistics
Endpoint: /stats/{user_id}
Method: GET

Purpose:
Get total conversations and timestamps.

Postman Setup:

Method: GET

URL:

arduino
Copy code
http://127.0.0.1:8005/stats/user123
Response Example:

json
Copy code
{
  "total_conversations": 3,
  "first_conversation": "2026-01-02T18:20:00",
  "last_conversation": "2026-01-02T18:25:00"
}
6. Clear Conversation History
Endpoint: /history/{user_id}
Method: DELETE

Purpose:
Delete all MongoDB chat history and clear memory.

Postman Setup:

Method: DELETE

URL:

arduino
Copy code
http://127.0.0.1:8005/history/user123
Response Example:

json
Copy code
{
  "message": "History cleared successfully",
  "conversations_deleted": 3,
  "user_id": "user123"
}
7. Get Conversation Memory
Endpoint: /memory/{user_id}
Method: GET

Purpose:
View in-memory conversation summary.

Postman Setup:

Method: GET

URL:

arduino
Copy code
http://127.0.0.1:8005/memory/user123
Response Example:

json
Copy code
{
  "user_id": "user123",
  "memory": "Human: What is the main topic?\nAssistant: The document discusses company policies."
}
8. Clear Conversation Memory Only
Endpoint: /memory/{user_id}
Method: DELETE

Purpose:
Clear conversation memory without deleting MongoDB history.

Postman Setup:

Method: DELETE

URL:

arduino
Copy code
http://127.0.0.1:8005/memory/user123
Response Example:

json
Copy code
{
  "message": "Memory cleared successfully",
  "user_id": "user123"
}
Important Notes
MongoDB must be running and reachable.

Upload documents before calling /ask.

Use form-data (not raw JSON) for file uploads.

Use consistent user_id to maintain memory.

Check terminal logs if an endpoint fails.

Recommended Test Order
GET /

POST /upload-documents

POST /ask

GET /history/{user_id}

GET /stats/{user_id}

GET /memory/{user_id}

DELETE /memory/{user_id}

DELETE /history/{user_id}

yaml
Copy code

---

If you want next, I can:
- ✅ Create a **Postman Collection JSON**
- ✅ Create **Swagger test examples**
- ✅ Add **curl commands**
- ✅ Add **automated API test cases**

Just tell me 👍
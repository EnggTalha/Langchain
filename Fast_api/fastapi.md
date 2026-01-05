# FastAPI Complete Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Building Your First API](#first-api)
5. [Path Parameters & Query Parameters](#parameters)
6. [Request Bodies with Pydantic](#request-bodies)
7. [Response Models](#response-models)
8. [File Uploads](#file-uploads)
9. [Error Handling](#error-handling)
10. [CORS Configuration](#cors)
11. [Database Integration](#database)
12. [Authentication](#authentication)
13. [Running the Application](#running)

---

## 1. Introduction {#introduction}

**FastAPI** is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

### Key Features:
- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Increase development speed by 200-300%
- **Fewer bugs**: Reduce human-induced errors by about 40%
- **Intuitive**: Great editor support with auto-completion
- **Easy**: Designed to be easy to use and learn
- **Automatic docs**: Interactive API documentation (Swagger UI)

---

## 2. Installation {#installation}

### Basic Installation
```bash
pip install fastapi
pip install "uvicorn[standard]"
```

### With All Features
```bash
pip install fastapi[all]
```

### Additional Dependencies
```bash
# For file uploads
pip install python-multipart

# For MongoDB
pip install pymongo

# For PostgreSQL
pip install sqlalchemy psycopg2-binary

# For authentication
pip install python-jose[cryptography] passlib[bcrypt]
```

---

## 3. Basic Concepts {#basic-concepts}

### HTTP Methods
- **GET**: Read data
- **POST**: Create data
- **PUT**: Update/replace data
- **PATCH**: Partial update
- **DELETE**: Delete data

### Status Codes
- **200**: OK
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized
- **404**: Not Found
- **500**: Internal Server Error

---

## 4. Building Your First API {#first-api}

### Minimal Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run it:
```bash
uvicorn main:app --reload
```

Visit: `http://localhost:8000` and `http://localhost:8000/docs`

### Multiple Endpoints

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="This is my awesome API",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to my API"}

@app.get("/items")
async def get_items():
    return {"items": ["item1", "item2", "item3"]}

@app.get("/users")
async def get_users():
    return {"users": ["Alice", "Bob", "Charlie"]}
```

---

## 5. Path Parameters & Query Parameters {#parameters}

### Path Parameters

```python
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
async def get_user_item(user_id: int, item_id: str):
    return {"user_id": user_id, "item_id": item_id}
```

### Query Parameters

```python
from typing import Optional

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/search/")
async def search_items(q: Optional[str] = None, min_price: float = 0):
    if q:
        return {"query": q, "min_price": min_price}
    return {"message": "No query provided"}
```

### Mixed Parameters

```python
@app.get("/users/{user_id}/items/")
async def get_user_items(
    user_id: int,
    skip: int = 0,
    limit: int = 10,
    q: Optional[str] = None
):
    return {
        "user_id": user_id,
        "skip": skip,
        "limit": limit,
        "query": q
    }
```

---

## 6. Request Bodies with Pydantic {#request-bodies}

### Basic Pydantic Model

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

### Nested Models

```python
from typing import List, Optional
from pydantic import BaseModel

class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tags: List[str] = []
    images: Optional[List[Image]] = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### Field Validation

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=0, le=120)
    full_name: Optional[str] = Field(None, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "age": 25,
                "full_name": "John Doe"
            }
        }

@app.post("/users/")
async def create_user(user: User):
    return user
```

---

## 7. Response Models {#response-models}

### Basic Response Model

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class ItemIn(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

class ItemOut(BaseModel):
    name: str
    price: float
    total_price: float

@app.post("/items/", response_model=ItemOut)
async def create_item(item: ItemIn):
    total = item.price + (item.tax or 0)
    return {
        "name": item.name,
        "price": item.price,
        "total_price": total
    }
```

### Response Model with Status Code

```python
from fastapi import FastAPI, status

@app.post("/items/", response_model=ItemOut, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemIn):
    # ... create logic
    return item
```

### Multiple Response Models

```python
from typing import Union
from fastapi import FastAPI

class BaseResponse(BaseModel):
    success: bool
    message: str

class SuccessResponse(BaseResponse):
    data: dict

class ErrorResponse(BaseResponse):
    error_code: str

@app.get("/items/{item_id}", response_model=Union[SuccessResponse, ErrorResponse])
async def get_item(item_id: int):
    if item_id < 0:
        return ErrorResponse(
            success=False,
            message="Invalid item ID",
            error_code="INVALID_ID"
        )
    return SuccessResponse(
        success=True,
        message="Item found",
        data={"id": item_id, "name": "Sample Item"}
    )
```

---

## 8. File Uploads {#file-uploads}

### Single File Upload

```python
from fastapi import FastAPI, File, UploadFile

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }
```

### Multiple Files Upload

```python
from typing import List

@app.post("/upload-multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    return {
        "filenames": [file.filename for file in files],
        "count": len(files)
    }
```

### Save Uploaded File

```python
import shutil
from pathlib import Path

@app.post("/save-file/")
async def save_file(file: UploadFile = File(...)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "location": str(file_path)
    }
```

---

## 9. Error Handling {#error-handling}

### HTTPException

```python
from fastapi import FastAPI, HTTPException

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )
    return items_db[item_id]
```

### Custom Exception Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."}
    )

@app.get("/custom-error/{name}")
async def trigger_error(name: str):
    if name == "error":
        raise CustomException(name=name)
    return {"name": name}
```

---

## 10. CORS Configuration {#cors}

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow all origins (development only!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 11. Database Integration {#database}

### MongoDB Example

```python
from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List

app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["users"]

class User(BaseModel):
    name: str
    email: str
    age: int

@app.post("/users/")
async def create_user(user: User):
    user_dict = user.dict()
    result = collection.insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    return user_dict

@app.get("/users/")
async def get_users():
    users = list(collection.find())
    for user in users:
        user["_id"] = str(user["_id"])
    return users
```

### SQLAlchemy Example (PostgreSQL)

```python
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = "postgresql://user:password@localhost/dbname"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/")
async def create_user(user: User, db: Session = Depends(get_db)):
    db_user = UserDB(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

---

## 12. Authentication {#authentication}

### Basic API Key Authentication

```python
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

app = FastAPI()

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/protected/")
async def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "Access granted"}
```

### JWT Token Authentication

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Verify username/password (simplified)
    if form_data.username != "test" or form_data.password != "test":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/protected/")
async def protected(username: str = Depends(verify_token)):
    return {"message": f"Hello {username}"}
```

---

## 13. Running the Application {#running}

### Development Mode

```bash
# Basic run
uvicorn main:app --reload

# Custom host and port
uvicorn main:app --host 0.0.0.0 --port 8005 --reload

# With custom log level
uvicorn main:app --reload --log-level debug
```

### Production Mode

```bash
# Single worker
uvicorn main:app --host 0.0.0.0 --port 8005

# Multiple workers (using Gunicorn)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8005
```

### Using Python Script

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )
```

---

## Complete Example: TODO API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="TODO API", version="1.0.0")

class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None
    completed: bool = False

class Todo(TodoCreate):
    id: int
    created_at: datetime

todos_db = []
todo_id_counter = 1

@app.post("/todos/", response_model=Todo, status_code=201)
async def create_todo(todo: TodoCreate):
    global todo_id_counter
    new_todo = Todo(
        id=todo_id_counter,
        title=todo.title,
        description=todo.description,
        completed=todo.completed,
        created_at=datetime.now()
    )
    todos_db.append(new_todo)
    todo_id_counter += 1
    return new_todo

@app.get("/todos/", response_model=List[Todo])
async def get_todos():
    return todos_db

@app.get("/todos/{todo_id}", response_model=Todo)
async def get_todo(todo_id: int):
    for todo in todos_db:
        if todo.id == todo_id:
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@app.put("/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, todo_update: TodoCreate):
    for idx, todo in enumerate(todos_db):
        if todo.id == todo_id:
            updated_todo = Todo(
                id=todo_id,
                title=todo_update.title,
                description=todo_update.description,
                completed=todo_update.completed,
                created_at=todo.created_at
            )
            todos_db[idx] = updated_todo
            return updated_todo
    raise HTTPException(status_code=404, detail="Todo not found")

@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int):
    for idx, todo in enumerate(todos_db):
        if todo.id == todo_id:
            todos_db.pop(idx)
            return {"message": "Todo deleted successfully"}
    raise HTTPException(status_code=404, detail="Todo not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Testing Your API

### Using cURL

```bash
# GET request
curl http://localhost:8005/

# POST request with JSON
curl -X POST http://localhost:8005/items/ \
  -H "Content-Type: application/json" \
  -d '{"name":"Item1","price":10.5}'

# Upload file
curl -X POST http://localhost:8005/upload/ \
  -F "file=@document.pdf"
```

### Using Python Requests

```python
import requests

# GET
response = requests.get("http://localhost:8005/")
print(response.json())

# POST
data = {"name": "Item1", "price": 10.5}
response = requests.post("http://localhost:8005/items/", json=data)
print(response.json())

# Upload file
files = {"file": open("document.pdf", "rb")}
response = requests.post("http://localhost:8005/upload/", files=files)
print(response.json())
```

---

## Best Practices

1. **Use Pydantic models** for request/response validation
2. **Use async/await** for I/O operations
3. **Implement proper error handling** with HTTPException
4. **Use dependency injection** for reusable components
5. **Add API documentation** with proper descriptions
6. **Use environment variables** for configuration
7. **Implement rate limiting** for production APIs
8. **Add proper logging** for debugging
9. **Use response models** to control output
10. **Write tests** using pytest and TestClient

---

## Resources

- **Official Docs**: https://fastapi.tiangolo.com/
- **GitHub**: https://github.com/tiangolo/fastapi
- **Community**: https://github.com/tiangolo/fastapi/discussions

---

**Happy coding with FastAPI! 🚀**
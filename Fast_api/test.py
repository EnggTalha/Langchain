from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="This is my awesome API",
    version="1.0.0"
)
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
async def get_user_item(user_id: int, item_id: str):
    return {"user_id": user_id, "item_id": item_id}
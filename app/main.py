from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import image

app = FastAPI(title="画像認識API")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# ルーターの登録
app.include_router(image.router, prefix="/api/v1")

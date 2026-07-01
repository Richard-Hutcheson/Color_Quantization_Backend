from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import health, images

app = FastAPI(title="Paint By Numbers API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(images.router)

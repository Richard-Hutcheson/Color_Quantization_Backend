from fastapi import FastAPI

from src.routers import health, images

app = FastAPI(title="Paint By Numbers API")

app.include_router(health.router)
app.include_router(images.router)

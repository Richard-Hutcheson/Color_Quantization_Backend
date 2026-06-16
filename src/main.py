from fastapi import FastAPI

from src.routers import health

app = FastAPI(title="Color Quantization API")

app.include_router(health.router)

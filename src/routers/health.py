from fastapi import APIRouter

from src.models.health import HealthResponse
from src.services.health import get_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return get_health()

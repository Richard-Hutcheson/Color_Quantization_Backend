from src.models.health import HealthResponse


def get_health() -> HealthResponse:
    return HealthResponse(status="ok")

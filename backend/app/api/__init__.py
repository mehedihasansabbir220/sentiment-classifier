# HTTP layer. Routes delegate to app.services; no inference logic here.

from app.api.routes import router

__all__ = ["router"]

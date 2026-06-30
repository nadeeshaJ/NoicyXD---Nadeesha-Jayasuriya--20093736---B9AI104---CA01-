from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.openapi import API_DESCRIPTION, CONTACT, LICENSE_INFO, OPENAPI_TAGS, SERVERS
from app.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sound Analytics API",
        description=API_DESCRIPTION,
        version="1.0.0",
        contact=CONTACT,
        license_info=LICENSE_INFO,
        openapi_tags=OPENAPI_TAGS,
        servers=SERVERS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "Sound Analytics API",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/health",
        }

    return app


app = create_app()

"""Главное FastAPI приложение"""
from typing import Optional, Dict, Any
import logging
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Annotated
from sqlmodel import Field, Session, SQLModel, create_engine, select, JSON, Column
from datetime import datetime
import contextlib
from config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    HOST, PORT, RELOAD, LOG_LEVEL, LOG_FORMAT
)
from schemas import (
    AtmData, PredictionResponse, 
    HealthResponse, PipelineInfoResponse
)
from services.pipeline_service import pipeline_service

# Настройка логирования
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Счетчик запросов
request_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Запуск
    logger.info("Запуск ATM Popularity API...")
    logger.info(f"Документация: http://{HOST}:{PORT}/docs")
    SQLModel.metadata.create_all(engine)
    
    yield
    
    # Остановка
    logger.info("Остановка ATM Popularity API...")

SQLModel.metadata.clear()

class History(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    timestamp: str = Field(index=True)
    content: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

sqlite_file_name = "history.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

# Создание приложения
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/history/")
def create_history(history: History, session: SessionDep) -> History:
    session.add(history)
    session.commit()
    session.refresh(history)
    return history

@app.get("/history/")
def read_history(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[History]:
    history = session.exec(select(History).offset(offset).limit(limit)).all()
    return history

# Эндпоинты
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Добро пожаловать в ATM Popularity Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "pipeline_loaded": pipeline_service.is_ready()
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Проверка здоровья сервиса"""
    return HealthResponse(
        status="OK" if pipeline_service.is_ready() else "ERROR",
        pipeline_loaded=pipeline_service.is_ready(),
        features_count=len(pipeline_service.feature_names)
    )


@app.get("/pipeline-info", response_model=PipelineInfoResponse, tags=["Pipeline"])
async def get_pipeline_info():
    """Информация о загруженном пайплайне"""
    if not pipeline_service.is_ready():
        raise HTTPException(status_code=503, detail="Пайплайн не загружен")
    
    info = pipeline_service.get_pipeline_info()
    return PipelineInfoResponse(**info)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

@app.post("/forward", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: AtmData, session: SessionDep):
    """Предсказание индекса популярности банкомата"""
    global request_count
    request_count += 1
    
    logger.info(f"Запрос #{request_count}: "
                f"lat={data.lat}, lon={data.lon}, "
                f"group={data.atm_group}, address={data.address_rus[:30]}...")
    
    if not pipeline_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Пайплайн не загружен. Запустите сначала обучение."
        )
    
    try:
        # Выполняем предсказание
        prediction = pipeline_service.predict(
            [[data.atm_group, data.lat, data.lon]]
        )

        history_entry = History(
            timestamp=datetime.now().strftime("%d.%m.%Y %H:%M"),
            content={'lat': data.lat, 'lon': data.lon, 'atm_group': data.atm_group}
        )
        
        logger.info(f"Результат #{request_count}: {prediction:.4f}")

        create_history(history_entry, session)

        return PredictionResponse(
            predicted_index=prediction,
            request_id=request_count,
            coordinates={"lat": data.lat, "lon": data.lon},
            atm_group=data.atm_group,
            address=data.address_rus[:50] + "..." if len(data.address_rus) > 50 else data.address_rus
        )
        
    except Exception as e:
        logger.error(f"Ошибка #{request_count}: {e}")
        raise HTTPException(status_code=403, detail='модель не смогла обработать данные')


# Запуск сервера
if __name__ == "__main__":
    logger.info(f"Запуск сервера на {HOST}:{PORT}")
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL.lower()
    )
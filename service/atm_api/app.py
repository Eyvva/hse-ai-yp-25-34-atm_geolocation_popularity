"""Главное FastAPI приложение"""

import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

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
    
    yield
    
    # Остановка
    logger.info("Остановка ATM Popularity API...")


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
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Зависимости
def get_pipeline_service():
    """Получение сервиса пайплайна"""
    return pipeline_service


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
async def predict(data: AtmData):
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
        
        logger.info(f"Результат #{request_count}: {prediction:.4f}")
        
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
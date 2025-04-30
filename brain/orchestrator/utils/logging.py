"""
Sistema de logging avanzado para el Orchestrator
"""
import os
import logging
import json
import datetime
import time
from logging.handlers import RotatingFileHandler
import functools

# Configuración de directorios
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Configuración de loggers
def setup_logging(level=logging.INFO):
    """Configura el sistema de logging"""
    # Logger principal
    logger = logging.getLogger("orchestrator")
    logger.setLevel(level)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "orchestrator.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Handler para JSON estructurado
    json_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "orchestrator_structured.json"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    json_handler.setLevel(level)
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            if hasattr(record, 'extra'):
                log_record.update(record.extra)
            return json.dumps(log_record)
    
    json_handler.setFormatter(JsonFormatter())
    
    # Añadir handlers al logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(json_handler)
    
    # Logger de actividad
    activity_logger = logging.getLogger("activity")
    activity_logger.setLevel(level)
    
    activity_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "activity.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    activity_format = logging.Formatter('%(asctime)s - %(message)s')
    activity_handler.setFormatter(activity_format)
    activity_logger.addHandler(activity_handler)
    
    return logger, activity_logger

# Decorador para monitorear rendimiento
def monitor_performance(func):
    """Decorador para monitorear el rendimiento de funciones"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("orchestrator.performance")
        start_time = time.time()
        result = None
        error = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            log_data = {
                "function": func.__name__,
                "duration_ms": round(duration * 1000, 2),
                "success": error is None,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if error:
                log_data["error"] = str(error)
                log_data["error_type"] = error.__class__.__name__
            
            logger.info(f"Performance: {json.dumps(log_data)}")
    return wrapper

# Decorador para reintentos con backoff exponencial
def retry_with_backoff(max_retries=3, initial_backoff=1, max_backoff=60, jitter=True):
    """Decorador para reintentos con backoff exponencial"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    
                    # Calcular tiempo de espera con backoff exponencial
                    wait_time = min(backoff * (2 ** (retries - 1)), max_backoff)
                    
                    # Añadir jitter si está habilitado
                    if jitter:
                        wait_time = wait_time * (0.5 + random.random())
                    
                    logger = logging.getLogger("orchestrator.retry")
                    logger.warning(
                        f"Reintento {retries}/{max_retries} para {func.__name__} "
                        f"después de {wait_time:.2f}s. Error: {str(e)}"
                    )
                    
                    time.sleep(wait_time)
        return wrapper
    return decorator

# Inicialización de loggers
logger, activity_logger = setup_logging()
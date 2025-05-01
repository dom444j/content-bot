# Scheduler System - Sistema de Planificación de Tareas

## Descripción
Sistema modular de planificación de tareas para la automatización de contenido multimedia. Diseñado para gestionar miles de tareas concurrentes con priorización inteligente, reintentos automáticos y distribución de carga.

## Uso Básico

### Programar una tarea
```python
from brain.scheduler.core.scheduler_facade import SchedulerFacade

scheduler = SchedulerFacade()

# Programar tarea simple
task_id = scheduler.schedule_task(
    task_type="content_creation",
    task_data={
        "channel_id": "channel123",
        "content_type": "short_video",
        "theme": "tech_news"
    },
    execution_time=datetime.datetime.now() + datetime.timedelta(hours=2),
    priority=TaskPriority.HIGH
)

# Verificar estado
status = scheduler.get_task_status(task_id)
print(f"Task status: {status}")

# Cancelar tarea
scheduler.cancel_task(task_id)
```

## Arquitectura
## Módulos Clave
### Core
- scheduler_facade.py : Punto de entrada principal que simplifica la interacción con el sistema
- task_queue.py : Implementación de cola de prioridades con soporte para operaciones atómicas
- task_model.py : Definición de modelos de datos y estados para tareas
- config.py : Gestión de configuración con validación y valores por defecto
### Executors
Diferentes estrategias para ejecutar tareas:

- local_executor.py : Ejecución síncrona en el mismo hilo
- thread_executor.py : Ejecución en hilos separados (para tareas I/O bound)
- process_executor.py : Ejecución en procesos separados (para tareas CPU bound)
- distributed_executor.py : Ejecución distribuida con Redis/RabbitMQ
### Strategies
Comportamientos configurables:

- retry_strategy.py : Estrategias de reintento con backoff exponencial
- priority_strategy.py : Algoritmos de priorización dinámica
- circuit_breaker.py : Prevención de fallos en cascada
## Integración con otros componentes
- Orchestrator : Coordinación de flujos de trabajo complejos
- Decision Engine : Priorización inteligente basada en aprendizaje
- Analytics Engine : Optimización de horarios basada en engagement
- Notifier : Alertas sobre fallos y anomalías
## Métricas y Monitoreo
Accede a métricas detalladas:

# Obtener métricas de rendimiento
metrics = scheduler.get_system_load_metrics()
print(f"Tareas por minuto: {metrics['tasks_per_minute']}")
print(f"Tasa de éxito: {metrics['success_rate_1h']}")

# Pronóstico de ejecución
forecast = scheduler.get_task_execution_forecast(lookahead_hours=24)


## 2. TaskLifecycle Enum en task_model.py

Esta sugerencia es fundamental para estandarizar los estados de las tareas. Actualmente, el código usa strings para estados, lo que puede llevar a inconsistencias. Un Enum proporcionará tipado seguro y documentación integrada:

```python:c%3A%5CUsers%5CDOM%5CDesktop%5Ccontent-bot%5Cbrain%5Cscheduler%5Ccore%5Ctask_model.py
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class TaskLifecycle(Enum):
    """
    Estados del ciclo de vida de una tarea
    
    Flujo típico:
    CREATED -> QUEUED -> SCHEDULED -> RUNNING -> COMPLETED
    
    Flujos alternativos:
    CREATED -> QUEUED -> SCHEDULED -> RUNNING -> FAILED -> RETRYING -> RUNNING -> ...
    CREATED -> QUEUED -> PAUSED -> QUEUED -> ...
    CREATED -> QUEUED -> CANCELLED
    """
    CREATED = auto()    # Tarea creada pero no añadida a ninguna cola
    QUEUED = auto()     # En cola, esperando programación
    SCHEDULED = auto()  # Programada para ejecución futura
    PAUSED = auto()     # Pausada (ej: esperando dependencias)
    RUNNING = auto()    # En ejecución actualmente
    RETRYING = auto()   # Fallida pero programada para reintento
    FAILED = auto()     # Fallida definitivamente
    COMPLETED = auto()  # Completada exitosamente
    CANCELLED = auto()  # Cancelada manualmente

class TaskPriority(Enum):
    """Niveles de prioridad para tareas"""
    CRITICAL = 0    # Tareas críticas del sistema (ej: recuperación)
    HIGH = 1        # Alta prioridad (ej: respuesta a tendencias virales)
    MEDIUM = 2      # Prioridad media (ej: publicaciones programadas)
    LOW = 3         # Baja prioridad (ej: análisis periódicos)
    BACKGROUND = 4  # Tareas en segundo plano (ej: limpieza, optimización)

@dataclass
class TaskMetadata:
    """Metadatos asociados a una tarea"""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    source_ip: Optional[str] = None
    source_component: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    audit_trail: List[Dict] = field(default_factory=list)

@dataclass
class Task:
    """
    Modelo de datos para una tarea en el sistema de planificación
    """
    id: str
    type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    lifecycle: TaskLifecycle = TaskLifecycle.CREATED
    
    # Tiempos
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Ejecución
    executor_type: str = "local"
    retries: int = 0
    max_retries: int = 3
    backoff_factor: int = 2
    
    # Dependencias
    dependencies: List[str] = field(default_factory=list)
    
    # Resultados
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Metadatos
    metadata: TaskMetadata = field(default_factory=TaskMetadata)
    
    def to_dict(self) -> Dict:
        """Convierte la tarea a diccionario para serialización"""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "priority": self.priority.name,
            "lifecycle": self.lifecycle.name,
            "created_at": self.created_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "executor_type": self.executor_type,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "dependencies": self.dependencies,
            "result": self.result,
            "error": self.error,
            "metadata": {
                "created_by": self.metadata.created_by,
                "source_ip": self.metadata.source_ip,
                "source_component": self.metadata.source_component,
                "tags": self.metadata.tags,
                "audit_trail": self.metadata.audit_trail
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Crea una tarea desde un diccionario"""
        metadata = TaskMetadata(
            created_by=data.get("metadata", {}).get("created_by", "system"),
            source_ip=data.get("metadata", {}).get("source_ip"),
            source_component=data.get("metadata", {}).get("source_component"),
            tags=data.get("metadata", {}).get("tags", []),
            audit_trail=data.get("metadata", {}).get("audit_trail", [])
        )
        
        return cls(
            id=data["id"],
            type=data["type"],
            data=data["data"],
            priority=TaskPriority[data["priority"]] if isinstance(data["priority"], str) else data["priority"],
            lifecycle=TaskLifecycle[data["lifecycle"]] if isinstance(data["lifecycle"], str) else data["lifecycle"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]) if data.get("scheduled_for") and isinstance(data["scheduled_for"], str) else data.get("scheduled_for"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") and isinstance(data["started_at"], str) else data.get("started_at"),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") and isinstance(data["completed_at"], str) else data.get("completed_at"),
            executor_type=data.get("executor_type", "local"),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
            backoff_factor=data.get("backoff_factor", 2),
            dependencies=data.get("dependencies", []),
            result=data.get("result"),
            error=data.get("error"),
            metadata=metadata
        )
```

## 3. Middleware para auditoría
Esta sugerencia es excelente para cumplimiento y análisis. Implementar un sistema de middleware permitirá interceptar y registrar información valiosa sin modificar el código principal:
import time
import socket
import inspect
import threading
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import logging

from ..core.task_model import Task, TaskMetadata

logger = logging.getLogger("scheduler.audit")

class AuditMiddleware:
    """
    Middleware para auditoría de tareas
    
    Registra información detallada sobre quién programa tareas, cuándo y desde dónde.
    Útil para análisis, cumplimiento normativo y seguridad.
    """
    
    def __init__(self, enabled: bool = True, log_level: int = logging.INFO):
        """
        Inicializa el middleware de auditoría
        
        Args:
            enabled: Si el middleware está activo
            log_level: Nivel de logging para eventos de auditoría
        """
        self.enabled = enabled
        self.log_level = log_level
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura el logger específico para auditoría"""
        handler = logging.FileHandler("logs/scheduler_audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        audit_logger = logging.getLogger("scheduler.audit")
        audit_logger.setLevel(self.log_level)
        audit_logger.addHandler(handler)
    
    def process_task(self, task: Task) -> Task:
        """
        Procesa una tarea para añadir información de auditoría
        
        Args:
            task: Tarea a procesar
            
        Returns:
            Tarea con información de auditoría añadida
        """
        if not self.enabled:
            return task
        
        # Obtener información del llamador
        caller_info = self._get_caller_info()
        
        # Añadir información a los metadatos
        if not task.metadata:
            task.metadata = TaskMetadata()
        
        task.metadata.source_ip = socket.gethostbyname(socket.gethostname())
        task.metadata.source_component = caller_info.get("module", "unknown")
        
        # Añadir entrada en el audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "task_created",
            "user": caller_info.get("caller", "system"),
            "source_file": caller_info.get("filename"),
            "source_line": caller_info.get("lineno"),
            "thread_id": threading.get_ident()
        }
        
        task.metadata.audit_trail.append(audit_entry)
        
        # Registrar en el log
        logger.log(
            self.log_level,
            f"Task {task.id} created by {audit_entry['user']} from {audit_entry['source_file']}:{audit_entry['source_line']}"
        )
        
        return task
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el código que está llamando al scheduler
        
        Returns:
            Diccionario con información del llamador
        """
        # Obtener stack de llamadas
        stack = inspect.stack()
        
        # Buscar el primer llamador fuera del módulo scheduler
        caller_frame = None
        for frame in stack[1:]:  # Ignorar esta función
            module = inspect.getmodule(frame[0])
            if module and "scheduler" not in module.__name__:
                caller_frame = frame
                break
        
        if not caller_frame:
            return {"module": "unknown", "caller": "system"}
        
        # Extraer información
        module = inspect.getmodule(caller_frame[0])
        module_name = module.__name__ if module else "unknown"
        
        return {
            "module": module_name,
            "filename": caller_frame.filename,
            "lineno": caller_frame.lineno,
            "function": caller_frame.function,
            "caller": self._get_username()
        }
    
    def _get_username(self) -> str:
        """
        Intenta obtener el nombre del usuario actual
        
        Returns:
            Nombre de usuario o 'system' si no se puede determinar
        """
        try:
            import getpass
            return getpass.getuser()
        except:
            return "system"

## 4. Integración con cron estándar
Esta sugerencia es muy valiosa para tareas recurrentes complejas. APScheduler es una excelente opción que se integra bien con el sistema existente:

from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from ..core.task_model import Task, TaskLifecycle, TaskPriority
from .base_executor import BaseExecutor

logger = logging.getLogger("scheduler.cron_executor")

class CronExecutor(BaseExecutor):
    """
    Ejecutor para tareas recurrentes basadas en expresiones cron
    
    Utiliza APScheduler para programar tareas con patrones complejos de recurrencia.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Inicializa el ejecutor cron
        
        Args:
            max_workers: Número máximo de hilos para ejecutar tareas
        """
        self.scheduler = BackgroundScheduler(
            jobstores={'default': MemoryJobStore()},
            executors={'default': ThreadPoolExecutor(max_workers)},
            job_defaults={'coalesce': True, 'max_instances': 1}
        )
        
        self.jobs = {}  # Mapeo de task_id a job_id
        self._lock = threading.RLock()
        self._started = False
    
    def start(self):
        """Inicia el scheduler de APScheduler"""
        if not self._started:
            self.scheduler.start()
            self._started = True
            logger.info("CronExecutor iniciado")
    
    def shutdown(self):
        """Detiene el scheduler de APScheduler"""
        if self._started:
            self.scheduler.shutdown()
            self._started = False
            logger.info("CronExecutor detenido")
    
    def execute(self, task: Task, callback: callable) -> bool:
        """
        Programa una tarea recurrente con expresión cron
        
        Args:
            task: Tarea a ejecutar
            callback: Función a llamar cuando la tarea se ejecute
            
        Returns:
            True si se programó correctamente, False en caso contrario
        """
        if not self._started:
            self.start()
        
        # Extraer expresión cron de los datos de la tarea
        cron_expression = task.data.get('cron_expression')
        if not cron_expression:
            logger.error(f"Tarea {task.id} no tiene expresión cron")
            return False
        
        try:
            # Crear trigger cron
            trigger = CronTrigger.from_crontab(cron_expression)
            
            # Función que se ejecutará en cada activación
            def job_func():
                # Crear una copia de la tarea para esta ejecución
                execution_task = Task(
                    id=f"{task.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    type=task.type,
                    data=task.data.copy(),
                    priority=task.priority,
                    lifecycle=TaskLifecycle.RUNNING,
                    created_at=datetime.now(),
                    scheduled_for=datetime.now(),
                    started_at=datetime.now(),
                    executor_type="cron",
                    metadata=task.metadata
                )
                
                # Eliminar la expresión cron para evitar recursión
                if 'cron_expression' in execution_task.data:
                    execution_task.data['original_cron'] = execution_task.data.pop('cron_expression')
                
                try:
                    # Ejecutar la tarea
                    result = callback(execution_task)
                    
                    # Actualizar estado
                    execution_task.lifecycle = TaskLifecycle.COMPLETED
                    execution_task.completed_at = datetime.now()
                    execution_task.result = result
                    
                    logger.info(f"Ejecución programada de tarea {task.id} completada: {execution_task.id}")
                    return True
                except Exception as e:
                    # Manejar error
                    execution_task.lifecycle = TaskLifecycle.FAILED
                    execution_task.error = str(e)
                    execution_task.completed_at = datetime.now()
                    
                    logger.error(f"Error en ejecución programada de tarea {task.id}: {str(e)}")
                    return False
            
            # Añadir trabajo al scheduler
            with self._lock:
                job = self.scheduler.add_job(
                    job_func,
                    trigger=trigger,
                    id=f"task_{task.id}",
                    replace_existing=True
                )
                self.jobs[task.id] = job.id
            
            # Actualizar estado de la tarea
            task.lifecycle = TaskLifecycle.SCHEDULED
            
            logger.info(f"Tarea recurrente {task.id} programada con expresión '{cron_expression}'")
            return True
            
        except Exception as e:
            logger.error(f"Error al programar tarea recurrente {task.id}: {str(e)}")
            return False
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancela una tarea recurrente
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            True si se canceló correctamente, False en caso contrario
        """
        with self._lock:
            job_id = self.jobs.get(task_id)
            if not job_id:
                logger.warning(f"No se encontró job para tarea {task_id}")
                return False
            
            try:
                self.scheduler.remove_job(job_id)
                del self.jobs[task_id]
                logger.info(f"Tarea recurrente {task_id} cancelada")
                return True
            except Exception as e:
                logger.error(f"Error al cancelar tarea recurrente {task_id}: {str(e)}")
                return False
    
    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de una tarea recurrente
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Diccionario con información del estado o None si no existe
        """
        with self._lock:
            job_id = self.jobs.get(task_id)
            if not job_id:
                return None
            
            try:
                job = self.scheduler.get_job(job_id)
                if not job:
                    return None
                
                return {
                    "id": task_id,
                    "job_id": job_id,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "cron_expression": str(job.trigger),
                    "active": job.next_run_time is not None
                }
            except Exception as e:
                logger.error(f"Error al obtener estado de tarea recurrente {task_id}: {str(e)}")
                return None
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """
        Obtiene información de todas las tareas recurrentes
        
        Returns:
            Lista de diccionarios con información de las tareas
        """
        jobs = []
        
        with self._lock:
            for task_id, job_id in self.jobs.items():
                try:
                    job = self.scheduler.get_job(job_id)
                    if job:
                        jobs.append({
                            "id": task_id,
                            "job_id": job_id,
                            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                            "cron_expression": str(job.trigger),
                            "active": job.next_run_time is not None
                        })
                except:
                    pass
        
        return jobs

## Alineación con MonetizationSystem.md
Estas mejoras se alinean perfectamente con los objetivos de MonetizationSystem.md :

1. Automatización 24/7 : El sistema de cron avanzado permitirá programar tareas complejas para mantener el sistema funcionando continuamente.
2. Escalabilidad a 5-20 canales : El modelo de tareas mejorado con TaskLifecycle y la auditoría facilitarán la gestión de múltiples canales con priorización inteligente.
3. Auto-Mejora : La integración con APScheduler permitirá programar tareas de análisis y optimización en patrones complejos.
4. Cumplimiento : El middleware de auditoría proporcionará trazabilidad completa para cumplimiento normativo y análisis de rendimiento.
5. Sostenibilidad : La estructura modular y bien documentada facilitará el mantenimiento y evolución del sistema.
## Conclusión
Las sugerencias propuestas son altamente recomendables y representan mejoras significativas para el sistema de planificación. Implementarlas proporcionará:

1. Mejor documentación con el README.md específico
2. Mayor robustez con el Enum TaskLifecycle
3. Mejor trazabilidad con el middleware de auditoría
4. Mayor flexibilidad con la integración de APScheduler
Estas mejoras no solo fortalecerán el sistema actual, sino que también lo prepararán para futuras expansiones y adaptaciones a nuevas plataformas y requisitos, alineándose perfectamente con la visión de escalabilidad y automatización descrita en MonetizationSystem.md .
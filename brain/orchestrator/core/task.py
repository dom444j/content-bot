"""
Task - Sistema de tareas para el Orchestrator

Este módulo implementa un sistema de tareas con prioridades, estados y reintentos
para el Orchestrator, permitiendo la ejecución asíncrona y ordenada de operaciones.
"""

import uuid
import logging
import datetime
import threading
import queue
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict

# Configuración de logging
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enumeración de estados de tareas."""
    PENDING = auto()      # Tarea pendiente de ejecución
    PROCESSING = auto()   # Tarea en procesamiento
    COMPLETED = auto()    # Tarea completada con éxito
    FAILED = auto()       # Tarea fallida definitivamente
    CANCELLED = auto()    # Tarea cancelada manualmente
    WAITING = auto()      # Tarea esperando dependencias

class TaskPriority(Enum):
    """Enumeración de prioridades de tareas."""
    CRITICAL = 0   # Máxima prioridad (0 es más prioritario)
    HIGH = 1       # Alta prioridad
    NORMAL = 2     # Prioridad normal
    LOW = 3        # Baja prioridad

class TaskType(Enum):
    """Enumeración de tipos de tareas."""
    CONTENT_CREATION = auto()    # Creación de contenido
    CONTENT_VERIFICATION = auto() # Verificación de contenido
    CONTENT_PUBLICATION = auto() # Publicación de contenido
    ANALYTICS = auto()           # Análisis de métricas
    MONETIZATION = auto()        # Acciones de monetización
    SHADOWBAN_CHECK = auto()     # Verificación de shadowban
    RECOVERY_ACTION = auto()     # Acción de recuperación
    TREND_DETECTION = auto()     # Detección de tendencias
    MAINTENANCE = auto()         # Mantenimiento del sistema
    NOTIFICATION = auto()        # Envío de notificaciones

@dataclass
class Task:
    """
    Clase que representa una tarea en el sistema.
    
    Attributes:
        id: Identificador único de la tarea
        type: Tipo de tarea
        priority: Prioridad de la tarea
        status: Estado actual de la tarea
        data: Datos específicos de la tarea
        channel_id: ID del canal asociado (opcional)
        dependencies: IDs de tareas de las que depende esta tarea
        result: Resultado de la ejecución (cuando está completada)
        error: Mensaje de error (cuando falla)
        retry_count: Número de reintentos realizados
        max_retries: Número máximo de reintentos permitidos
        created_at: Timestamp de creación
        updated_at: Timestamp de última actualización
        completed_at: Timestamp de completado (si está completada)
        callback: Función a llamar cuando la tarea se complete
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType = TaskType.CONTENT_CREATION
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    data: Dict[str, Any] = field(default_factory=dict)
    channel_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    completed_at: Optional[str] = None
    callback: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la tarea a un diccionario serializable.
        
        Returns:
            Dict[str, Any]: Representación de la tarea como diccionario
        """
        result = asdict(self)
        # Convertir enumeraciones a strings para serialización
        result["type"] = self.type.name
        result["priority"] = self.priority.name
        result["status"] = self.status.name
        # Eliminar callback que no es serializable
        result.pop("callback", None)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Crea una tarea a partir de un diccionario.
        
        Args:
            data: Diccionario con datos de la tarea
            
        Returns:
            Task: Instancia de tarea
        """
        # Convertir strings a enumeraciones
        if "type" in data and isinstance(data["type"], str):
            data["type"] = TaskType[data["type"]]
        if "priority" in data and isinstance(data["priority"], str):
            data["priority"] = TaskPriority[data["priority"]]
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus[data["status"]]
        
        # Eliminar callback si existe en los datos
        data.pop("callback", None)
        
        return cls(**data)

class TaskManager:
    """
    Gestor de tareas para el Orchestrator.
    
    Esta clase se encarga de crear, actualizar, eliminar y consultar tareas,
    así como de gestionar su estado, dependencias y ejecución.
    """
    
    def __init__(self, persistence_manager=None, config_manager=None, num_workers: int = 5):
        """
        Inicializa el gestor de tareas.
        
        Args:
            persistence_manager: Gestor de persistencia para guardar tareas
            config_manager: Gestor de configuración
            num_workers: Número de hilos de trabajo
        """
        self.tasks = {}  # Diccionario de tareas por ID
        self.persistence = persistence_manager
        self.config_manager = config_manager
        self.task_queue = queue.PriorityQueue()  # Cola de prioridades
        self.lock = threading.RLock()  # Lock para operaciones thread-safe
        self.worker_threads = []  # Lista de hilos de trabajo
        self.shutdown_event = threading.Event()  # Evento para señalizar apagado
        
        # Cargar tareas desde persistencia si está disponible
        self._load_tasks_from_persistence()
        
        logger.info("TaskManager inicializado")
    
    def _load_tasks_from_persistence(self) -> None:
        """
        Carga tareas desde el sistema de persistencia.
        """
        if not self.persistence:
            logger.debug("No hay gestor de persistencia configurado")
            return
        
        try:
            tasks_data = self.persistence.load_collection("tasks")
            if tasks_data:
                for task_data in tasks_data:
                    task = Task.from_dict(task_data)
                    self.tasks[task.id] = task
                    
                    # Añadir a la cola si está pendiente y no tiene dependencias pendientes
                    if task.status == TaskStatus.PENDING:
                        all_deps_completed = True
                        for dep_id in task.dependencies:
                            dep_task = self.tasks.get(dep_id)
                            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                                all_deps_completed = False
                                break
                        
                        if all_deps_completed:
                            self.task_queue.put((task.priority.value, task.created_at, task.id))
                
                logger.info(f"Cargadas {len(tasks_data)} tareas desde persistencia")
        
        except Exception as e:
            logger.error(f"Error al cargar tareas desde persistencia: {str(e)}")
    
    def _save_task_to_persistence(self, task: Task) -> None:
        """
        Guarda una tarea en el sistema de persistencia.
        
        Args:
            task: Tarea a guardar
        """
        if not self.persistence:
            return
        
        try:
            self.persistence.save_document("tasks", task.id, task.to_dict())
        except Exception as e:
            logger.error(f"Error al guardar tarea {task.id} en persistencia: {str(e)}")
    
    def create_task(self, task_type: TaskType, priority: TaskPriority = TaskPriority.NORMAL,
                   channel_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None,
                   dependencies: Optional[List[str]] = None, max_retries: int = 3,
                   callback: Optional[Callable] = None) -> Task:
        """
        Crea una nueva tarea.
        
        Args:
            task_type: Tipo de tarea
            priority: Prioridad de la tarea
            channel_id: ID del canal asociado (opcional)
            data: Datos específicos de la tarea
            dependencies: IDs de tareas de las que depende esta tarea
            max_retries: Número máximo de reintentos permitidos
            callback: Función a llamar cuando la tarea se complete
            
        Returns:
            Task: Tarea creada
        """
        with self.lock:
            task = Task(
                type=task_type,
                priority=priority,
                channel_id=channel_id,
                data=data or {},
                dependencies=dependencies or [],
                max_retries=max_retries,
                callback=callback
            )
            
            # Guardar tarea
            self.tasks[task.id] = task
            
            # Añadir a la cola si no tiene dependencias o todas están completadas
            if not task.dependencies or all(
                self.tasks.get(dep_id, Task(status=TaskStatus.FAILED)).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            ):
                self.task_queue.put((task.priority.value, task.created_at, task.id))
            
            # Persistir
            self._save_task_to_persistence(task)
            
            logger.info(f"Tarea {task.id} creada: {task_type.name}, prioridad {priority.name}")
            return task
    
    def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """
        Verifica si hay tareas dependientes que ahora pueden ejecutarse.
        
        Args:
            completed_task_id: ID de la tarea completada
        """
        with self.lock:
            for task_id, task in self.tasks.items():
                if (task.status == TaskStatus.PENDING and 
                    completed_task_id in task.dependencies):
                    
                    # Verificar si todas las dependencias están completadas
                    all_deps_completed = all(
                        self.tasks.get(dep_id, Task(status=TaskStatus.FAILED)).status == TaskStatus.COMPLETED
                        for dep_id in task.dependencies
                    )
                    
                    if all_deps_completed:
                        # Añadir a la cola
                        self.task_queue.put((task.priority.value, task.created_at, task.id))
                        logger.debug(f"Tarea {task_id} lista para ejecución, todas las dependencias completadas")
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea existente.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si la cancelación fue exitosa, False en caso contrario
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.warning(f"Intento de cancelar tarea inexistente: {task_id}")
                return False
            
            # Solo se pueden cancelar tareas pendientes o en procesamiento
            if task.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                logger.warning(f"No se puede cancelar tarea {task_id} en estado {task.status.name}")
                return False
            
            # Actualizar estado
            task.status = TaskStatus.CANCELLED
            task.updated_at = datetime.datetime.now().isoformat()
            
            # Persistir
            self._save_task_to_persistence(task)
            
            logger.info(f"Tarea {task_id} cancelada")
            return True
    
    def get_next_task(self) -> Optional[Task]:
        """
        Obtiene la siguiente tarea de mayor prioridad de la cola.
        
        Returns:
            Optional[Task]: Siguiente tarea o None si la cola está vacía
        """
        try:
            # Intentar obtener la siguiente tarea con timeout para no bloquear
            priority, created_at, task_id = self.task_queue.get(timeout=0.1)
            
            with self.lock:
                task = self.tasks.get(task_id)
                
                # Verificar si la tarea existe y sigue pendiente
                if not task:
                    logger.warning(f"Tarea {task_id} no encontrada en el diccionario de tareas")
                    return None
                
                if task.status != TaskStatus.PENDING:
                    logger.warning(f"Tarea {task_id} ya no está pendiente, estado actual: {task.status.name}")
                    return None
                
                # Actualizar estado
                task.status = TaskStatus.PROCESSING
                task.updated_at = datetime.datetime.now().isoformat()
                
                # Persistir
                self._save_task_to_persistence(task)
                
                return task
                
        except queue.Empty:
            # Cola vacía
            return None
        except Exception as e:
            logger.error(f"Error al obtener siguiente tarea: {str(e)}")
            return None
    
    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Marca una tarea como completada.
        
        Args:
            task_id: ID de la tarea
            result: Resultado de la ejecución (opcional)
            
        Returns:
            bool: True si se completó correctamente, False en caso contrario
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.warning(f"Intento de completar tarea inexistente: {task_id}")
                return False
            
            # Solo se pueden completar tareas en procesamiento
            if task.status != TaskStatus.PROCESSING:
                logger.warning(f"No se puede completar tarea {task_id} en estado {task.status.name}")
                return False
            
            # Actualizar estado
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.updated_at = datetime.datetime.now().isoformat()
            task.completed_at = datetime.datetime.now().isoformat()
            
            # Persistir
            self._save_task_to_persistence(task)
            
            # Verificar dependencias
            self._check_dependent_tasks(task_id)
            
            # Ejecutar callback si existe
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Error en callback de tarea {task_id}: {str(e)}")
            
            logger.info(f"Tarea {task_id} completada")
            return True
    
    def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """
        Marca una tarea como fallida, con opción de reintento.
        
        Args:
            task_id: ID de la tarea
            error: Mensaje de error
            retry: Si se debe reintentar la tarea (si no excede max_retries)
            
        Returns:
            bool: True si se marcó como fallida correctamente, False en caso contrario
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.warning(f"Intento de fallar tarea inexistente: {task_id}")
                return False
            
            # Solo se pueden fallar tareas en procesamiento
            if task.status != TaskStatus.PROCESSING:
                logger.warning(f"No se puede fallar tarea {task_id} en estado {task.status.name}")
                return False
            
            # Verificar si se debe reintentar
            if retry and task.retry_count < task.max_retries:
                # Incrementar contador de reintentos
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = error
                task.updated_at = datetime.datetime.now().isoformat()
                
                # Añadir a la cola con prioridad ajustada (más alta para reintentos)
                new_priority = max(0, task.priority.value - 1)  # Aumentar prioridad en reintentos
                self.task_queue.put((new_priority, datetime.datetime.now().isoformat(), task.id))
                
                logger.info(f"Tarea {task_id} fallida, reintentando ({task.retry_count}/{task.max_retries})")
            else:
                # Marcar como fallida definitivamente
                task.status = TaskStatus.FAILED
                task.error = error
                task.updated_at = datetime.datetime.now().isoformat()
                
                logger.warning(f"Tarea {task_id} fallida definitivamente: {error}")
            
            # Persistir
            self._save_task_to_persistence(task)
            
            return True
    
    def get_all_tasks(self, status: Optional[TaskStatus] = None, 
                     channel_id: Optional[str] = None,
                     task_type: Optional[TaskType] = None) -> List[Task]:
        """
        Obtiene todas las tareas, opcionalmente filtradas.
        
        Args:
            status: Filtrar por estado (opcional)
            channel_id: Filtrar por canal (opcional)
            task_type: Filtrar por tipo (opcional)
            
        Returns:
            List[Task]: Lista de tareas que cumplen los filtros
        """
        with self.lock:
            filtered_tasks = []
            
            for task in self.tasks.values():
                # Aplicar filtros
                if status and task.status != status:
                    continue
                if channel_id and task.channel_id != channel_id:
                    continue
                if task_type and task.type != task_type:
                    continue
                
                filtered_tasks.append(task)
            
            return filtered_tasks
    
    def get_task_count(self, status: Optional[TaskStatus] = None) -> int:
        """
        Obtiene el número de tareas, opcionalmente filtradas por estado.
        
        Args:
            status: Estado para filtrar (opcional)
            
        Returns:
            int: Número de tareas
        """
        with self.lock:
            if status:
                return sum(1 for task in self.tasks.values() if task.status == status)
            return len(self.tasks)
    
    def clear_completed_tasks(self, older_than_days: int = 7) -> int:
        """
        Elimina tareas completadas o fallidas más antiguas que el número de días especificado.
        
        Args:
            older_than_days: Eliminar tareas más antiguas que este número de días
            
        Returns:
            int: Número de tareas eliminadas
        """
        with self.lock:
            cutoff_date = (datetime.datetime.now() - 
                          datetime.timedelta(days=older_than_days)).isoformat()
            
            tasks_to_remove = []
            
            for task_id, task in self.tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.updated_at < cutoff_date):
                    tasks_to_remove.append(task_id)
            
            # Eliminar tareas
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                
                # Eliminar de persistencia si está disponible
                if self.persistence:
                    try:
                        self.persistence.delete_document("tasks", task_id)
                    except Exception as e:
                        logger.error(f"Error al eliminar tarea {task_id} de persistencia: {str(e)}")
            
            logger.info(f"Eliminadas {len(tasks_to_remove)} tareas antiguas")
            return len(tasks_to_remove)
    
    def start_workers(self, num_workers: int = None) -> None:
        """
        Inicia hilos de trabajo para procesar tareas.
        
        Args:
            num_workers: Número de hilos de trabajo (opcional, usa valor de configuración por defecto)
        """
        if num_workers is None:
            num_workers = self.config_manager.get("system.num_workers", 5) if hasattr(self, "config_manager") else 5
        
        self.shutdown_event = threading.Event()
        self.worker_threads = []  # Inicializar lista de hilos
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Iniciados {num_workers} hilos de trabajo para tareas")
    
    def stop_workers(self) -> None:
        """
        Detiene los hilos de trabajo de forma segura.
        """
        logger.info("Deteniendo hilos de trabajo...")
        self.shutdown_event.set()
        
        # Esperar a que todos los hilos terminen
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # Limpiar lista de hilos
        self.worker_threads = []
        logger.info("Hilos de trabajo detenidos")
    
    def _worker_loop(self) -> None:
        """
        Bucle principal de los hilos de trabajo.
        """
        logger.debug(f"Hilo de trabajo {threading.current_thread().name} iniciado")
        
        while not self.shutdown_event.is_set():
            try:
                # Obtener siguiente tarea
                task = self.get_next_task()
                
                if task:
                    logger.debug(f"Procesando tarea {task.id} ({task.type.name})")
                    
                    # Aquí normalmente se delegaría la ejecución a un handler específico
                    # según el tipo de tarea, pero en este ejemplo solo simulamos el procesamiento
                    time.sleep(0.5)  # Simular procesamiento
                    
                    # Completar tarea con resultado simulado
                    self.complete_task(task.id, {"status": "success", "processed_at": time.time()})
                else:
                    # No hay tareas, esperar un poco
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error en hilo de trabajo: {str(e)}")
                time.sleep(1.0)  # Esperar antes de continuar para evitar bucles de error
        
        logger.debug(f"Hilo de trabajo {threading.current_thread().name} finalizado")
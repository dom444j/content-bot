"""
Batch Processor - Sistema de procesamiento por lotes

Este módulo implementa un sistema de procesamiento por lotes que permite
optimizar recursos al procesar múltiples elementos de contenido (videos, imágenes,
audio, etc.) en grupos, reduciendo costos de API y mejorando la eficiencia.

Características principales:
- Procesamiento paralelo y secuencial configurable
- Priorización inteligente de tareas
- Gestión de dependencias entre tareas
- Optimización de recursos (CPU, GPU, memoria, API)
- Reintentos automáticos con backoff exponencial
- Monitoreo y reportes de rendimiento
"""

import os
import json
import time
import logging
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import queue
import concurrent.futures
import random
import traceback
import psutil
import numpy as np
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/batch_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("batch_processor")

class TaskPriority(Enum):
    """Niveles de prioridad para las tareas en el procesador por lotes."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class TaskStatus(Enum):
    """Estados posibles de una tarea en el procesador por lotes."""
    PENDING = 0
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    RETRYING = 5
    CANCELLED = 6
    SKIPPED = 7

class ResourceType(Enum):
    """Tipos de recursos que pueden ser requeridos por las tareas."""
    CPU = 0
    GPU = 1
    MEMORY = 2
    API_CALL = 3
    NETWORK = 4
    DISK_IO = 5

class BatchTask:
    """
    Representa una tarea individual dentro de un lote de procesamiento.
    
    Cada tarea tiene metadatos, dependencias, requisitos de recursos,
    y una función de procesamiento asociada.
    """
    
    def __init__(self, 
                task_id: str,
                task_type: str,
                processor_func: Callable,
                input_data: Any,
                priority: TaskPriority = TaskPriority.NORMAL,
                dependencies: List[str] = None,
                resource_requirements: Dict[ResourceType, float] = None,
                max_retries: int = 3,
                timeout: int = 300,
                metadata: Dict[str, Any] = None):
        """
        Inicializa una tarea de procesamiento por lotes.
        
        Args:
            task_id: Identificador único de la tarea
            task_type: Tipo de tarea (video, audio, imagen, etc.)
            processor_func: Función que procesará la tarea
            input_data: Datos de entrada para la tarea
            priority: Prioridad de la tarea
            dependencies: Lista de IDs de tareas de las que depende esta tarea
            resource_requirements: Requisitos de recursos para la tarea
            max_retries: Número máximo de reintentos si la tarea falla
            timeout: Tiempo máximo de ejecución en segundos
            metadata: Metadatos adicionales para la tarea
        """
        self.task_id = task_id
        self.task_type = task_type
        self.processor_func = processor_func
        self.input_data = input_data
        self.priority = priority
        self.dependencies = dependencies or []
        self.resource_requirements = resource_requirements or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self.metadata = metadata or {}
        
        # Estado y resultados
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.retry_count = 0
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.execution_time = None
        self.next_retry_time = None
        
    def __lt__(self, other):
        """
        Comparador para ordenar tareas por prioridad.
        Permite que las tareas se ordenen automáticamente en colas de prioridad.
        """
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la tarea a un diccionario para serialización."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority.name,
            "dependencies": self.dependencies,
            "resource_requirements": {r.name: v for r, v in self.resource_requirements.items()},
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "status": self.status.name,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "next_retry_time": self.next_retry_time.isoformat() if self.next_retry_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], processor_func: Callable, input_data: Any) -> 'BatchTask':
        """Crea una tarea a partir de un diccionario serializado."""
        task = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            processor_func=processor_func,
            input_data=input_data,
            priority=TaskPriority[data["priority"]],
            dependencies=data["dependencies"],
            resource_requirements={ResourceType[k]: v for k, v in data.get("resource_requirements", {}).items()},
            max_retries=data["max_retries"],
            timeout=data["timeout"],
            metadata=data["metadata"]
        )
        
        task.status = TaskStatus[data["status"]]
        task.retry_count = data["retry_count"]
        task.created_at = datetime.fromisoformat(data["created_at"])
        
        if data["started_at"]:
            task.started_at = datetime.fromisoformat(data["started_at"])
        
        if data["completed_at"]:
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        
        task.execution_time = data["execution_time"]
        
        if data["next_retry_time"]:
            task.next_retry_time = datetime.fromisoformat(data["next_retry_time"])
        
        return task

class BatchProcessor:
    """
    Sistema de procesamiento por lotes para optimizar recursos y eficiencia.
    
    Permite procesar múltiples tareas en paralelo o secuencialmente,
    gestionando dependencias, prioridades y recursos disponibles.
    """
    
    def __init__(self, 
                config_path: str = "config/batch_processor_config.json",
                max_workers: int = None,
                max_parallel_tasks: int = None,
                resource_limits: Dict[ResourceType, float] = None,
                api_rate_limits: Dict[str, Dict[str, Any]] = None,
                batch_size: int = 10,
                auto_start: bool = True,
                checkpoint_interval: int = 300,
                checkpoint_path: str = "data/batch_processor_state.json"):
        """
        Inicializa el procesador por lotes.
        
        Args:
            config_path: Ruta al archivo de configuración
            max_workers: Número máximo de workers para procesamiento paralelo
            max_parallel_tasks: Número máximo de tareas en paralelo
            resource_limits: Límites de recursos disponibles
            api_rate_limits: Límites de tasa para APIs
            batch_size: Tamaño predeterminado de lote
            auto_start: Si es True, inicia automáticamente el procesador
            checkpoint_interval: Intervalo en segundos para guardar el estado
            checkpoint_path: Ruta para guardar el estado del procesador
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Configurar límites de recursos
        self.max_workers = max_workers or self.config.get("max_workers", multiprocessing.cpu_count())
        self.max_parallel_tasks = max_parallel_tasks or self.config.get("max_parallel_tasks", self.max_workers * 2)
        self.resource_limits = resource_limits or self.config.get("resource_limits", self._get_default_resource_limits())
        self.api_rate_limits = api_rate_limits or self.config.get("api_rate_limits", {})
        self.batch_size = batch_size or self.config.get("batch_size", 10)
        self.checkpoint_interval = checkpoint_interval or self.config.get("checkpoint_interval", 300)
        
        # Inicializar colas de tareas
        self.task_queue = queue.PriorityQueue()
        self.pending_tasks = {}  # task_id -> BatchTask
        self.running_tasks = {}  # task_id -> (BatchTask, Future)
        self.completed_tasks = {}  # task_id -> BatchTask
        self.failed_tasks = {}  # task_id -> BatchTask
        self.retry_queue = queue.PriorityQueue()
        
        # Recursos utilizados actualmente
        self.used_resources = {resource_type: 0.0 for resource_type in ResourceType}
        
        # Contadores de API
        self.api_counters = {api_name: {"count": 0, "last_reset": datetime.now()} 
                            for api_name in self.api_rate_limits.keys()}
        
        # Estadísticas
        self.stats = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "resource_utilization": {resource_type.name: 0.0 for resource_type in ResourceType},
            "api_calls": {api_name: 0 for api_name in self.api_rate_limits.keys()},
            "start_time": datetime.now().isoformat(),
            "last_checkpoint": None
        }
        
        # Inicializar executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Flags de control
        self.running = False
        self.paused = False
        
        # Locks y eventos
        self.resource_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.checkpoint_event = threading.Event()
        
        # Hilos de trabajo
        self.scheduler_thread = None
        self.checkpoint_thread = None
        
        # Cargar estado anterior si existe
        self._load_checkpoint()
        
        # Iniciar procesador si auto_start es True
        if auto_start:
            self.start()
        
        logger.info(f"BatchProcessor inicializado. Workers: {self.max_workers}, Tareas paralelas: {self.max_parallel_tasks}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración del procesador por lotes."""
        default_config = {
            "max_workers": multiprocessing.cpu_count(),
            "max_parallel_tasks": multiprocessing.cpu_count() * 2,
            "resource_limits": self._get_default_resource_limits(),
            "api_rate_limits": {
                "youtube": {"limit": 10000, "period": "day"},
                "tiktok": {"limit": 100, "period": "day"},
                "instagram": {"limit": 25, "period": "day"},
                "leonardo": {"limit": 150, "period": "day"},
                "elevenlabs": {"limit": 100, "period": "day"}
            },
            "batch_size": 10,
            "checkpoint_interval": 300,
            "retry_backoff_factor": 2,
            "retry_jitter": 0.1,
            "task_timeout_multiplier": 1.5
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Combinar con configuración predeterminada
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            else:
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                # Guardar configuración predeterminada
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return default_config
    
    def _get_default_resource_limits(self) -> Dict[ResourceType, float]:
        """Obtiene los límites de recursos predeterminados basados en el sistema."""
        try:
            cpu_count = multiprocessing.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Intentar detectar GPU
            gpu_memory = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
            except ImportError:
                pass
            
            return {
                ResourceType.CPU: cpu_count,
                ResourceType.GPU: gpu_memory,
                ResourceType.MEMORY: memory_gb,
                ResourceType.API_CALL: 1000,  # Valor arbitrario
                ResourceType.NETWORK: 100,    # MB/s (valor arbitrario)
                ResourceType.DISK_IO: 100     # MB/s (valor arbitrario)
            }
        except Exception as e:
            logger.error(f"Error al obtener límites de recursos predeterminados: {str(e)}")
            return {
                ResourceType.CPU: 4,
                ResourceType.GPU: 0,
                ResourceType.MEMORY: 8,
                ResourceType.API_CALL: 1000,
                ResourceType.NETWORK: 100,
                ResourceType.DISK_IO: 100
            }
    
    def _save_checkpoint(self) -> None:
        """Guarda el estado actual del procesador por lotes."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            
            # Preparar datos para serialización
            checkpoint_data = {
                "stats": self.stats,
                "pending_tasks": {task_id: task.to_dict() for task_id, task in self.pending_tasks.items()},
                "completed_tasks": {task_id: task.to_dict() for task_id, task in self.completed_tasks.items()},
                "failed_tasks": {task_id: task.to_dict() for task_id, task in self.failed_tasks.items()},
                "api_counters": self.api_counters,
                "timestamp": datetime.now().isoformat()
            }
            
            # Guardar checkpoint
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=4)
            
            # Actualizar estadísticas
            with self.stats_lock:
                self.stats["last_checkpoint"] = datetime.now().isoformat()
            
            logger.debug(f"Checkpoint guardado en {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Error al guardar checkpoint: {str(e)}")
    
    def _load_checkpoint(self) -> None:
        """Carga el estado anterior del procesador por lotes si existe."""
        try:
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Cargar estadísticas
                self.stats = checkpoint_data["stats"]
                self.stats["start_time"] = datetime.now().isoformat()
                
                # Cargar contadores de API
                self.api_counters = checkpoint_data["api_counters"]
                for api_name in self.api_counters:
                    self.api_counters[api_name]["last_reset"] = datetime.fromisoformat(
                        self.api_counters[api_name]["last_reset"]
                    )
                
                logger.info(f"Checkpoint cargado desde {self.checkpoint_path}")
                logger.info(f"Estadísticas previas: Procesadas={self.stats['tasks_processed']}, "
                           f"Exitosas={self.stats['tasks_succeeded']}, Fallidas={self.stats['tasks_failed']}")
        except Exception as e:
            logger.error(f"Error al cargar checkpoint: {str(e)}")
    
    def _checkpoint_worker(self) -> None:
        """Hilo de trabajo para guardar checkpoints periódicamente."""
        while self.running:
            # Esperar hasta el próximo checkpoint o hasta que se solicite uno
            self.checkpoint_event.wait(timeout=self.checkpoint_interval)
            self.checkpoint_event.clear()
            
            if not self.running:
                break
            
            self._save_checkpoint()
    
    def _can_run_task(self, task: BatchTask) -> bool:
        """
        Verifica si una tarea puede ejecutarse basándose en dependencias y recursos.
        
        Args:
            task: Tarea a verificar
            
        Returns:
            True si la tarea puede ejecutarse, False en caso contrario
        """
        # Verificar dependencias
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        # Verificar recursos disponibles
        with self.resource_lock:
            for resource_type, amount in task.resource_requirements.items():
                if self.used_resources[resource_type] + amount > self.resource_limits.get(resource_type, float('inf')):
                    return False
        
        # Verificar límites de API si es necesario
        if ResourceType.API_CALL in task.resource_requirements:
            api_name = task.metadata.get("api_name")
            if api_name and api_name in self.api_rate_limits:
                limit_info = self.api_rate_limits[api_name]
                counter_info = self.api_counters[api_name]
                
                # Reiniciar contador si ha pasado el período
                period_seconds = self._get_period_seconds(limit_info["period"])
                if (datetime.now() - counter_info["last_reset"]).total_seconds() > period_seconds:
                    counter_info["count"] = 0
                    counter_info["last_reset"] = datetime.now()
                
                # Verificar si se ha alcanzado el límite
                if counter_info["count"] >= limit_info["limit"]:
                    return False
        
        return True
    
    def _get_period_seconds(self, period: str) -> int:
        """Convierte un período a segundos."""
        if period == "second":
            return 1
        elif period == "minute":
            return 60
        elif period == "hour":
            return 3600
        elif period == "day":
            return 86400
        elif period == "week":
            return 604800
        elif period == "month":
            return 2592000
        else:
            return 86400  # Predeterminado: día
    
    def _allocate_resources(self, task: BatchTask) -> None:
        """Reserva recursos para una tarea."""
        with self.resource_lock:
            for resource_type, amount in task.resource_requirements.items():
                self.used_resources[resource_type] += amount
    
    def _release_resources(self, task: BatchTask) -> None:
        """Libera recursos utilizados por una tarea."""
        with self.resource_lock:
            for resource_type, amount in task.resource_requirements.items():
                self.used_resources[resource_type] = max(0, self.used_resources[resource_type] - amount)
    
    def _update_api_counter(self, api_name: str) -> None:
        """Incrementa el contador de una API."""
        if api_name in self.api_counters:
            counter_info = self.api_counters[api_name]
            
            # Reiniciar contador si ha pasado el período
            limit_info = self.api_rate_limits[api_name]
            period_seconds = self._get_period_seconds(limit_info["period"])
            if (datetime.now() - counter_info["last_reset"]).total_seconds() > period_seconds:
                counter_info["count"] = 0
                counter_info["last_reset"] = datetime.now()
            
            # Incrementar contador
            counter_info["count"] += 1
            
            # Actualizar estadísticas
            with self.stats_lock:
                self.stats["api_calls"][api_name] += 1
    
    def _calculate_retry_delay(self, task: BatchTask) -> float:
        """Calcula el retraso para el próximo reintento de una tarea fallida."""
        backoff_factor = self.config.get("retry_backoff_factor", 2)
        jitter = self.config.get("retry_jitter", 0.1)
        
        # Calcular retraso base con backoff exponencial
        delay = backoff_factor ** task.retry_count
        
        # Añadir jitter aleatorio
        delay = delay * (1 + random.uniform(-jitter, jitter))
        
        return delay
    
    def _process_task(self, task: BatchTask) -> Any:
        """
        Procesa una tarea y maneja su resultado.
        
        Args:
            task: Tarea a procesar
            
        Returns:
            Resultado de la tarea
        """
        # Marcar inicio de ejecución
        task.started_at = datetime.now()
        task.status = TaskStatus.RUNNING
        
        try:
            # Ejecutar función de procesamiento
            result = task.processor_func(task.input_data)
            
            # Marcar finalización exitosa
            task.completed_at = datetime.now()
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
            
            # Actualizar estadísticas
            with self.stats_lock:
                self.stats["tasks_processed"] += 1
                self.stats["tasks_succeeded"] += 1
                self.stats["total_processing_time"] += task.execution_time
                self.stats["avg_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["tasks_processed"]
                )
            
            # Actualizar contador de API si es necesario
            api_name = task.metadata.get("api_name")
            if api_name:
                self._update_api_counter(api_name)
            
            logger.info(f"Tarea {task.task_id} completada en {task.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Marcar error
            task.completed_at = datetime.now()
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
            
            # Actualizar estadísticas
            with self.stats_lock:
                self.stats["tasks_processed"] += 1
                self.stats["tasks_failed"] += 1
                self.stats["total_processing_time"] += task.execution_time
                self.stats["avg_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["tasks_processed"]
                )
            
            # Determinar si se debe reintentar
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Calcular tiempo para el próximo reintento
                retry_delay = self._calculate_retry_delay(task)
                task.next_retry_time = datetime.now() + timedelta(seconds=retry_delay)
                
                # Actualizar estadísticas
                with self.stats_lock:
                    self.stats["tasks_retried"] += 1
                
                logger.warning(f"Tarea {task.task_id} fallida, reintento {task.retry_count}/{task.max_retries} "
                              f"en {retry_delay:.2f}s. Error: {task.error}")
                
                # Agregar a la cola de reintentos
                self.retry_queue.put((task.next_retry_time, task))
            else:
                logger.error(f"Tarea {task.task_id} fallida definitivamente después de {task.max_retries} intentos. "
                            f"Error: {task.error}")
                
                # Guardar en tareas fallidas
                self.failed_tasks[task.task_id] = task
            
            # Propagar excepción
            raise
    
    def _task_done_callback(self, future, task_id):
        """Callback para cuando una tarea termina de ejecutarse."""
        try:
            # Obtener tarea y futuro
            task, _ = self.running_tasks.pop(task_id, (None, None))
            if task is None:
                return
            
            # Liberar recursos
            self._release_resources(task)
            
            # Procesar resultado o error
            try:
                future.result()  # Esto lanzará excepción si la tarea falló
                
                # Mover a tareas completadas
                self.completed_tasks[task_id] = task
                
            except Exception:
                # El error ya fue manejado en _process_task
                if task.status != TaskStatus.RETRYING:
                    self.failed_tasks[task_id] = task
            
            # Solicitar checkpoint si es necesario
            if len(self.completed_tasks) % 10 == 0:
                self.checkpoint_event.set()
                
        except Exception as e:
            logger.error(f"Error en callback de tarea {task_id}: {str(e)}")
    
    def _scheduler_worker(self) -> None:
        """Hilo de trabajo principal para programar y ejecutar tareas."""
        while self.running:
            try:
                # Si está pausado, esperar
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Procesar tareas en cola de reintentos que estén listas
                current_time = datetime.now()
                while not self.retry_queue.empty():
                    try:
                        retry_time, task = self.retry_queue.get_nowait()
                        if retry_time <= current_time:
                            # La tarea está lista para reintentarse
                            self.task_queue.put(task)
                        else:
                            # Devolver a la cola de reintentos
                            self.retry_queue.put((retry_time, task))
                            break
                    except queue.Empty:
                        break
                
                # Verificar si hay espacio para más tareas
                if len(self.running_tasks) >= self.max_parallel_tasks:
                    time.sleep(0.1)
                    continue
                
                # Intentar obtener una tarea de la cola
                try:
                    task = self.task_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                
                # Verificar si la tarea puede ejecutarse
                if not self._can_run_task(task):
                    # Devolver a la cola
                    self.task_queue.put(task)
                    time.sleep(0.1)
                    continue
                
                # Reservar recursos
                self._allocate_resources(task)
                
                # Ejecutar tarea
                future = self.executor.submit(self._process_task, task)
                future.add_done_callback(lambda f, tid=task.task_id: self._task_done_callback(f, tid))
                
                # Registrar tarea en ejecución
                self.running_tasks[task.task_id] = (task, future)
                
                logger.debug(f"Tarea {task.task_id} iniciada")
                
            except Exception as e:
                logger.error(f"Error en scheduler: {str(e)}")
                time.sleep(1)
    
    def start(self) -> None:
        """Inicia el procesador por lotes."""
        if self.running:
            return
        
        self.running = True
        self.paused = False
        
        # Iniciar hilos de trabajo
        self.scheduler_thread = threading.Thread(target=self._scheduler_worker)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.checkpoint_thread = threading.Thread(target=self._checkpoint_worker)
        self.checkpoint_thread.daemon = True
        self.checkpoint_thread.start()
        
        logger.info("BatchProcessor iniciado")
    
    def stop(self) -> None:
        """Detiene el procesador por lotes."""
        if not self.running:
            return
        
        self.running = False
        
        # Señalizar hilos para que terminen
        self.checkpoint_event.set()
        
        # Esperar a que terminen los hilos
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        if self.checkpoint_thread:
            self.checkpoint_thread.join(timeout=5)
        
                # Cancelar tareas en ejecución
        for task_id, (task, future) in list(self.running_tasks.items()):
            if not future.done():
                future.cancel()
            self._release_resources(task)
            task.status = TaskStatus.CANCELLED
            self.failed_tasks[task_id] = task
        
        self.running_tasks.clear()
        
        # Guardar estado final
        self._save_checkpoint()
        
        # Cerrar executor
        self.executor.shutdown(wait=False)
        
        logger.info("BatchProcessor detenido")
    
    def pause(self) -> None:
        """Pausa el procesador por lotes."""
        if not self.running or self.paused:
            return
        
        self.paused = True
        logger.info("BatchProcessor pausado")
    
    def resume(self) -> None:
        """Reanuda el procesador por lotes."""
        if not self.running or not self.paused:
            return
        
        self.paused = False
        logger.info("BatchProcessor reanudado")
    
    def add_task(self, task: BatchTask) -> Dict[str, Any]:
        """
        Agrega una tarea al procesador por lotes.
        
        Args:
            task: Tarea a agregar
            
        Returns:
            Información sobre la tarea agregada
        """
        try:
            # Verificar si la tarea ya existe
            if task.task_id in self.pending_tasks or task.task_id in self.running_tasks or \
               task.task_id in self.completed_tasks or task.task_id in self.failed_tasks:
                return {
                    "status": "error",
                    "message": f"Ya existe una tarea con ID {task.task_id}"
                }
            
            # Agregar a tareas pendientes
            self.pending_tasks[task.task_id] = task
            
            # Agregar a cola de tareas
            task.status = TaskStatus.QUEUED
            self.task_queue.put(task)
            
            logger.info(f"Tarea {task.task_id} agregada a la cola")
            
            return {
                "status": "success",
                "message": "Tarea agregada correctamente",
                "task_id": task.task_id
            }
            
        except Exception as e:
            logger.error(f"Error al agregar tarea: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al agregar tarea: {str(e)}"
            }
    
    def add_batch(self, tasks: List[BatchTask]) -> Dict[str, Any]:
        """
        Agrega un lote de tareas al procesador.
        
        Args:
            tasks: Lista de tareas a agregar
            
        Returns:
            Información sobre las tareas agregadas
        """
        try:
            added_tasks = []
            skipped_tasks = []
            
            for task in tasks:
                # Verificar si la tarea ya existe
                if task.task_id in self.pending_tasks or task.task_id in self.running_tasks or \
                   task.task_id in self.completed_tasks or task.task_id in self.failed_tasks:
                    skipped_tasks.append(task.task_id)
                    continue
                
                # Agregar a tareas pendientes
                self.pending_tasks[task.task_id] = task
                
                # Agregar a cola de tareas
                task.status = TaskStatus.QUEUED
                self.task_queue.put(task)
                
                added_tasks.append(task.task_id)
            
            logger.info(f"Lote de tareas agregado. Agregadas: {len(added_tasks)}, Omitidas: {len(skipped_tasks)}")
            
            return {
                "status": "success",
                "message": f"Lote de tareas agregado. {len(added_tasks)} agregadas, {len(skipped_tasks)} omitidas",
                "added_tasks": added_tasks,
                "skipped_tasks": skipped_tasks
            }
            
        except Exception as e:
            logger.error(f"Error al agregar lote de tareas: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al agregar lote de tareas: {str(e)}"
            }
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancela una tarea específica.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si la tarea está en ejecución
            if task_id in self.running_tasks:
                task, future = self.running_tasks[task_id]
                
                # Cancelar futuro
                if not future.done():
                    future.cancel()
                
                # Liberar recursos
                self._release_resources(task)
                
                # Actualizar estado
                task.status = TaskStatus.CANCELLED
                self.failed_tasks[task_id] = task
                
                # Eliminar de tareas en ejecución
                del self.running_tasks[task_id]
                
                logger.info(f"Tarea {task_id} cancelada durante ejecución")
                
                return {
                    "status": "success",
                    "message": "Tarea cancelada durante ejecución",
                    "task_id": task_id
                }
            
            # Verificar si la tarea está pendiente
            elif task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                
                # Actualizar estado
                task.status = TaskStatus.CANCELLED
                self.failed_tasks[task_id] = task
                
                # Eliminar de tareas pendientes
                del self.pending_tasks[task_id]
                
                logger.info(f"Tarea {task_id} cancelada antes de ejecución")
                
                return {
                    "status": "success",
                    "message": "Tarea cancelada antes de ejecución",
                    "task_id": task_id
                }
            
            # La tarea no existe o ya ha terminado
            else:
                if task_id in self.completed_tasks:
                    return {
                        "status": "error",
                        "message": "La tarea ya ha sido completada",
                        "task_id": task_id
                    }
                elif task_id in self.failed_tasks:
                    return {
                        "status": "error",
                        "message": "La tarea ya ha fallado o ha sido cancelada",
                        "task_id": task_id
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Tarea no encontrada",
                        "task_id": task_id
                    }
            
        except Exception as e:
            logger.error(f"Error al cancelar tarea {task_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al cancelar tarea: {str(e)}",
                "task_id": task_id
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de una tarea.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Información sobre el estado de la tarea
        """
        try:
            # Buscar tarea en todas las colecciones
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                status = "pending"
            elif task_id in self.running_tasks:
                task, _ = self.running_tasks[task_id]
                status = "running"
            elif task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                status = "completed"
            elif task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                status = "failed" if task.status == TaskStatus.FAILED else "cancelled"
            else:
                return {
                    "status": "error",
                    "message": "Tarea no encontrada",
                    "task_id": task_id
                }
            
            # Preparar información de la tarea
            task_info = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status.name,
                "priority": task.priority.name,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "execution_time": task.execution_time,
                "retry_count": task.retry_count,
                "error": task.error,
                "dependencies": task.dependencies,
                "metadata": task.metadata
            }
            
            return {
                "status": "success",
                "task_status": status,
                "task_info": task_info
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estado de tarea {task_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener estado de tarea: {str(e)}",
                "task_id": task_id
            }
    
    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Obtiene el resultado de una tarea completada.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Resultado de la tarea
        """
        try:
            # Verificar si la tarea está completada
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                
                return {
                    "status": "success",
                    "task_id": task_id,
                    "result": task.result,
                    "execution_time": task.execution_time,
                    "completed_at": task.completed_at.isoformat()
                }
            
            # Verificar si la tarea falló
            elif task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                
                return {
                    "status": "error",
                    "message": "La tarea falló o fue cancelada",
                    "task_id": task_id,
                    "error": task.error,
                    "task_status": task.status.name
                }
            
            # Verificar si la tarea está en ejecución
            elif task_id in self.running_tasks:
                return {
                    "status": "pending",
                    "message": "La tarea está en ejecución",
                    "task_id": task_id
                }
            
            # Verificar si la tarea está pendiente
            elif task_id in self.pending_tasks:
                return {
                    "status": "pending",
                    "message": "La tarea está pendiente",
                    "task_id": task_id
                }
            
            # La tarea no existe
            else:
                return {
                    "status": "error",
                    "message": "Tarea no encontrada",
                    "task_id": task_id
                }
            
        except Exception as e:
            logger.error(f"Error al obtener resultado de tarea {task_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener resultado de tarea: {str(e)}",
                "task_id": task_id
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del procesador por lotes.
        
        Returns:
            Estadísticas de uso y rendimiento
        """
        try:
            # Calcular estadísticas adicionales
            pending_count = len(self.pending_tasks)
            running_count = len(self.running_tasks)
            completed_count = len(self.completed_tasks)
            failed_count = len(self.failed_tasks)
            total_count = pending_count + running_count + completed_count + failed_count
            
            # Calcular utilización de recursos
            resource_utilization = {}
            for resource_type in ResourceType:
                used = self.used_resources[resource_type]
                limit = self.resource_limits.get(resource_type, float('inf'))
                if limit == float('inf'):
                    utilization = 0.0
                else:
                    utilization = (used / limit) * 100.0
                
                resource_utilization[resource_type.name] = {
                    "used": used,
                    "limit": limit if limit != float('inf') else "unlimited",
                    "utilization_percent": utilization
                }
            
            # Calcular tiempo de ejecución
            uptime_seconds = (datetime.now() - datetime.fromisoformat(self.stats["start_time"])).total_seconds()
            
            # Preparar estadísticas
            stats = {
                "status": "success",
                "tasks": {
                    "total": total_count,
                    "pending": pending_count,
                    "running": running_count,
                    "completed": completed_count,
                    "failed": failed_count,
                    "processed": self.stats["tasks_processed"],
                    "succeeded": self.stats["tasks_succeeded"],
                    "failed_total": self.stats["tasks_failed"],
                    "retried": self.stats["tasks_retried"]
                },
                "performance": {
                    "avg_processing_time": self.stats["avg_processing_time"],
                    "total_processing_time": self.stats["total_processing_time"],
                    "uptime_seconds": uptime_seconds,
                    "tasks_per_second": self.stats["tasks_processed"] / uptime_seconds if uptime_seconds > 0 else 0
                },
                "resources": resource_utilization,
                "api_calls": self.stats["api_calls"],
                "api_limits": {
                    api_name: {
                        "limit": limit_info["limit"],
                        "period": limit_info["period"],
                        "current_count": self.api_counters[api_name]["count"],
                        "reset_at": self.api_counters[api_name]["last_reset"].isoformat()
                    }
                    for api_name, limit_info in self.api_rate_limits.items()
                },
                "system": {
                    "max_workers": self.max_workers,
                    "max_parallel_tasks": self.max_parallel_tasks,
                    "batch_size": self.batch_size,
                    "checkpoint_interval": self.checkpoint_interval,
                    "last_checkpoint": self.stats["last_checkpoint"],
                    "status": "running" if self.running else "stopped",
                    "paused": self.paused
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener estadísticas: {str(e)}"
            }
    
    def clear_completed_tasks(self, older_than_days: int = None) -> Dict[str, Any]:
        """
        Elimina tareas completadas del historial.
        
        Args:
            older_than_days: Si se especifica, solo elimina tareas más antiguas que este número de días
            
        Returns:
            Resultado de la operación
        """
        try:
            count = 0
            
            if older_than_days is not None:
                cutoff_time = datetime.now() - timedelta(days=older_than_days)
                
                # Eliminar tareas completadas antiguas
                for task_id in list(self.completed_tasks.keys()):
                    task = self.completed_tasks[task_id]
                    if task.completed_at and task.completed_at < cutoff_time:
                        del self.completed_tasks[task_id]
                        count += 1
                
                # Eliminar tareas fallidas antiguas
                for task_id in list(self.failed_tasks.keys()):
                    task = self.failed_tasks[task_id]
                    if task.completed_at and task.completed_at < cutoff_time:
                        del self.failed_tasks[task_id]
                        count += 1
                
                logger.info(f"Eliminadas {count} tareas más antiguas que {older_than_days} días")
            else:
                # Eliminar todas las tareas completadas
                count = len(self.completed_tasks)
                self.completed_tasks.clear()
                
                logger.info(f"Eliminadas {count} tareas completadas")
            
            # Guardar checkpoint
            self._save_checkpoint()
            
            return {
                "status": "success",
                "message": f"Se eliminaron {count} tareas del historial",
                "count": count
            }
            
        except Exception as e:
            logger.error(f"Error al limpiar tareas completadas: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al limpiar tareas completadas: {str(e)}"
            }
    
    def reset_stats(self) -> Dict[str, Any]:
        """
        Reinicia las estadísticas del procesador.
        
        Returns:
            Resultado de la operación
        """
        try:
            with self.stats_lock:
                # Guardar estadísticas anteriores
                old_stats = self.stats.copy()
                
                # Reiniciar estadísticas
                self.stats = {
                    "tasks_processed": 0,
                    "tasks_succeeded": 0,
                    "tasks_failed": 0,
                    "tasks_retried": 0,
                    "total_processing_time": 0.0,
                    "avg_processing_time": 0.0,
                    "resource_utilization": {resource_type.name: 0.0 for resource_type in ResourceType},
                    "api_calls": {api_name: 0 for api_name in self.api_rate_limits.keys()},
                    "start_time": datetime.now().isoformat(),
                    "last_checkpoint": None
                }
                
                # Reiniciar contadores de API
                for api_name in self.api_counters:
                    self.api_counters[api_name]["count"] = 0
                    self.api_counters[api_name]["last_reset"] = datetime.now()
                
                logger.info("Estadísticas reiniciadas")
                
                return {
                    "status": "success",
                    "message": "Estadísticas reiniciadas correctamente",
                    "previous_stats": old_stats
                }
                
        except Exception as e:
            logger.error(f"Error al reiniciar estadísticas: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al reiniciar estadísticas: {str(e)}"
            }
    
    def find_similar_tasks(self, task_type: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Busca tareas similares basadas en tipo y metadatos.
        
        Args:
            task_type: Tipo de tarea a buscar
            metadata: Metadatos para filtrar tareas
            
        Returns:
            Lista de tareas que coinciden con los criterios
        """
        try:
            matching_tasks = []
            
            # Función para verificar si una tarea coincide con los criterios
            def task_matches(task):
                if task_type and task.task_type != task_type:
                    return False
                
                if metadata:
                    for key, value in metadata.items():
                        if key not in task.metadata or task.metadata[key] != value:
                            return False
                
                return True
            
            # Buscar en todas las colecciones
            for task_id, task in self.pending_tasks.items():
                if task_matches(task):
                    matching_tasks.append({
                        "task_id": task_id,
                        "status": "pending",
                        "task_info": task.to_dict()
                    })
            
            for task_id, (task, _) in self.running_tasks.items():
                if task_matches(task):
                    matching_tasks.append({
                        "task_id": task_id,
                        "status": "running",
                        "task_info": task.to_dict()
                    })
            
            for task_id, task in self.completed_tasks.items():
                if task_matches(task):
                    matching_tasks.append({
                        "task_id": task_id,
                        "status": "completed",
                        "task_info": task.to_dict()
                    })
            
            for task_id, task in self.failed_tasks.items():
                if task_matches(task):
                    matching_tasks.append({
                        "task_id": task_id,
                        "status": "failed" if task.status == TaskStatus.FAILED else "cancelled",
                        "task_info": task.to_dict()
                    })
            
            return {
                "status": "success",
                "count": len(matching_tasks),
                "tasks": matching_tasks
            }
            
        except Exception as e:
            logger.error(f"Error al buscar tareas similares: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al buscar tareas similares: {str(e)}"
            }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear procesador por lotes
    processor = BatchProcessor(
        config_path="config/batch_processor_config.json",
        checkpoint_path="data/batch_processor_state.json",
        auto_start=True
    )
    
    # Definir una función de procesamiento de ejemplo
    def process_image(data):
        """Función de ejemplo para procesar una imagen."""
        # Simular procesamiento
        time.sleep(2)
        return {"width": data["width"], "height": data["height"], "processed": True}
    
    # Crear una tarea
    task = BatchTask(
        task_id="image_process_1",
        task_type="image_processing",
        processor_func=process_image,
        input_data={"width": 800, "height": 600, "format": "jpg"},
        priority=TaskPriority.NORMAL,
        resource_requirements={
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 0.5,
            ResourceType.API_CALL: 1.0
        },
        metadata={"source": "example", "api_name": "leonardo"}
    )
    
    # Agregar tarea al procesador
    result = processor.add_task(task)
    print(f"Tarea agregada: {result}")
    
    # Esperar un momento para que se procese
    time.sleep(5)
    
    # Obtener estado de la tarea
    status = processor.get_task_status(task.task_id)
    print(f"Estado de la tarea: {status}")
    
    # Obtener estadísticas
    stats = processor.get_stats()
    print(f"Estadísticas: {stats}")
    
    # Detener procesador
    processor.stop()
"""
Cola de prioridades para el sistema de planificación de tareas.

Este módulo implementa una cola de prioridades thread-safe que permite:
- Añadir tareas con diferentes niveles de prioridad
- Obtener la siguiente tarea a ejecutar basada en prioridad y tiempo
- Actualizar el estado de tareas existentes
- Cancelar tareas programadas
"""

import threading
import heapq
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

from .task_model import Task, TaskStatus, TaskPriority

logger = logging.getLogger('Scheduler.TaskQueue')

class TaskQueue:
    """
    Implementación thread-safe de una cola de prioridades para tareas.
    Permite operaciones atómicas y soporta priorización dinámica.
    """
    
    def __init__(self):
        self._queue = []  # Cola de prioridad (heap)
        self._task_map = {}  # Mapeo de task_id a tareas para acceso rápido
        self._lock = threading.RLock()  # Lock para operaciones thread-safe
        self._condition = threading.Condition(self._lock)  # Para wait/notify
        logger.info("TaskQueue inicializada")
    
    def add_task(self, task: Task) -> str:
        """
        Añade una tarea a la cola con la prioridad especificada.
        
        Args:
            task: Objeto Task a añadir
            
        Returns:
            str: ID de la tarea añadida
        """
        with self._lock:
            # Asegurar que la tarea tenga un ID único
            if not task.task_id:
                task.task_id = str(uuid.uuid4())
            
            # Crear entrada para la cola de prioridad
            # Formato: (prioridad, tiempo_ejecución, tiempo_creación, id_tarea)
            # El tiempo_creación se usa como desempate para tareas con misma prioridad y tiempo
            priority_entry = (
                task.priority.value,
                task.execution_time.timestamp() if task.execution_time else float('inf'),
                time.time(),
                task.task_id
            )
            
            # Añadir a la cola y al mapa
            heapq.heappush(self._queue, priority_entry)
            self._task_map[task.task_id] = task
            
            logger.debug(f"Tarea añadida: {task.task_id} - Tipo: {task.task_type} - Prioridad: {task.priority.name}")
            
            # Notificar a los hilos en espera que hay una nueva tarea
            self._condition.notify_all()
            
            return task.task_id
    
    def get_next_task(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Obtiene la siguiente tarea a ejecutar basada en prioridad y tiempo.
        
        Args:
            block: Si es True, bloquea hasta que haya una tarea disponible
            timeout: Tiempo máximo de espera en segundos (si block es True)
            
        Returns:
            Task o None: La siguiente tarea a ejecutar o None si no hay tareas disponibles
        """
        with self._condition:
            # Esperar si la cola está vacía y block es True
            if block and not self._queue:
                self._condition.wait(timeout)
            
            # Verificar si hay tareas disponibles
            if not self._queue:
                return None
            
            # Obtener la tarea con mayor prioridad
            now = time.time()
            next_priority, next_exec_time, _, next_id = self._queue[0]
            
            # Si la tarea aún no está lista para ejecutar, devolver None
            if next_exec_time > now:
                return None
            
            # Eliminar la tarea de la cola
            heapq.heappop(self._queue)
            
            # Obtener la tarea del mapa y actualizar su estado
            task = self._task_map[next_id]
            task.status = TaskStatus.PENDING
            
            logger.debug(f"Obtenida tarea: {task.task_id} - Tipo: {task.task_type}")
            
            return task
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        """
        Actualiza los atributos de una tarea existente.
        
        Args:
            task_id: ID de la tarea a actualizar
            **kwargs: Atributos a actualizar
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        with self._lock:
            if task_id not in self._task_map:
                logger.warning(f"Intento de actualizar tarea inexistente: {task_id}")
                return False
            
            task = self._task_map[task_id]
            
            # Actualizar atributos
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            logger.debug(f"Tarea actualizada: {task_id}")
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea programada.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si la cancelación fue exitosa, False en caso contrario
        """
        with self._lock:
            if task_id not in self._task_map:
                logger.warning(f"Intento de cancelar tarea inexistente: {task_id}")
                return False
            
            # Marcar la tarea como cancelada
            task = self._task_map[task_id]
            task.status = TaskStatus.CANCELLED
            
            # No eliminamos de la cola principal por eficiencia
            # Se filtrará cuando se intente obtener
            
            logger.info(f"Tarea cancelada: {task_id}")
            return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Obtiene una tarea por su ID.
        
        Args:
            task_id: ID de la tarea a obtener
            
        Returns:
            Task o None: La tarea solicitada o None si no existe
        """
        with self._lock:
            return self._task_map.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """
        Obtiene todas las tareas en la cola.
        
        Returns:
            List[Task]: Lista de todas las tareas
        """
        with self._lock:
            return list(self._task_map.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Obtiene todas las tareas con un estado específico.
        
        Args:
            status: Estado de las tareas a obtener
            
        Returns:
            List[Task]: Lista de tareas con el estado especificado
        """
        with self._lock:
            return [task for task in self._task_map.values() if task.status == status]
    
    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        """
        Obtiene todas las tareas de un tipo específico.
        
        Args:
            task_type: Tipo de las tareas a obtener
            
        Returns:
            List[Task]: Lista de tareas del tipo especificado
        """
        with self._lock:
            return [task for task in self._task_map.values() if task.task_type == task_type]
    
    def clear(self) -> None:
        """
        Elimina todas las tareas de la cola.
        """
        with self._lock:
            self._queue = []
            self._task_map = {}
            logger.info("Cola de tareas limpiada")
    
    def size(self) -> int:
        """
        Obtiene el número de tareas en la cola.
        
        Returns:
            int: Número de tareas
        """
        with self._lock:
            return len(self._task_map)
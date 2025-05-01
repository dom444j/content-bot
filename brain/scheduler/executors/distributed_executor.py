"""
Ejecutor distribuido para tareas en nodos remotos.

Este módulo implementa un ejecutor que distribuye tareas a nodos remotos
a través de un sistema de mensajería, permitiendo la ejecución en un
entorno distribuido.
"""

import logging
import threading
import time
import json
import uuid
import asyncio
import aiohttp
import backoff
from typing import Dict, Any, Optional, List, Callable, Union

from ..core.task_model import Task, TaskStatus
from .base_executor import BaseExecutor

logger = logging.getLogger('Scheduler.Executor.Distributed')

class DistributedExecutor(BaseExecutor):
    """
    Ejecutor que distribuye tareas a nodos remotos.
    
    Ideal para:
    - Tareas que requieren recursos no disponibles localmente
    - Distribución de carga en un clúster
    - Procesamiento de alto rendimiento en múltiples máquinas
    
    Ventajas:
    - Escalabilidad horizontal
    - Distribución de carga
    - Tolerancia a fallos
    
    Limitaciones:
    - Mayor latencia
    - Complejidad de configuración
    - Dependencia de infraestructura externa
    """
    
    def __init__(self, task_queue, config=None):
        """
        Inicializa el ejecutor distribuido.
        
        Args:
            task_queue: Cola de tareas compartida
            config: Configuración específica del ejecutor
        """
        super().__init__(task_queue, config)
        
        # Configuración de nodos
        self.nodes = self.config.get('nodes', [])
        self.max_workers = self.config.get('max_workers', 20)
        self.timeout = self.config.get('timeout', 300)  # segundos
        self.retry_delay = self.config.get('retry_delay', 5)  # segundos
        self.max_retries = self.config.get('max_retries', 3)
        
        # Estado de nodos
        self.node_status = {}  # node_url -> status
        self.node_status_lock = threading.RLock()
        
        # Tareas en ejecución
        self.running_tasks = {}  # task_id -> node_url
        self.running_tasks_lock = threading.RLock()
        
        # Cliente HTTP
        self.session = None
        self.event_loop = None
        self.event_loop_thread = None
        
        # Verificar configuración
        if not self.nodes:
            logger.warning("No se han configurado nodos para el ejecutor distribuido")
        
        logger.info(f"DistributedExecutor inicializado con {len(self.nodes)} nodos")
    
    def start(self):
        """
        Inicia el ejecutor distribuido.
        """
        if self.running:
            logger.warning("DistributedExecutor ya está en ejecución")
            return
        
        # Iniciar loop de eventos en un hilo separado
        self.event_loop = asyncio.new_event_loop()
        self.event_loop_thread = threading.Thread(
            target=self._run_event_loop,
            name="DistributedExecutorEventLoop",
            daemon=True
        )
        self.event_loop_thread.start()
        
        # Crear sesión HTTP
        future = asyncio.run_coroutine_threadsafe(self._create_session(), self.event_loop)
        self.session = future.result()
        
        # Iniciar el ejecutor base
        super().start()
        
        # Iniciar monitoreo de nodos
        self._start_node_monitoring()
        
        logger.info(f"DistributedExecutor iniciado con {len(self.nodes)} nodos")
    
    def _run_event_loop(self):
        """
        Ejecuta el loop de eventos en un hilo separado.
        """
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()
    
    async def _create_session(self):
        """
        Crea una sesión HTTP asíncrona.
        
        Returns:
            aiohttp.ClientSession: Sesión HTTP
        """
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={"Content-Type": "application/json"}
        )
    
    def _start_node_monitoring(self):
        """
        Inicia el monitoreo periódico de nodos.
        """
        # Programar primera verificación
        asyncio.run_coroutine_threadsafe(self._check_nodes_health(), self.event_loop)
        
        # Programar verificaciones periódicas
        def schedule_check():
            if self.running:
                asyncio.run_coroutine_threadsafe(self._check_nodes_health(), self.event_loop)
                # Programar próxima verificación en 60 segundos
                threading.Timer(60, schedule_check).start()
        
        # Iniciar programación
        threading.Timer(60, schedule_check).start()
    
    async def _check_nodes_health(self):
        """
        Verifica el estado de salud de todos los nodos.
        """
        logger.debug("Verificando estado de nodos...")
        
        for node_url in self.nodes:
            try:
                health_url = f"{node_url}/health"
                async with self.session.get(health_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        with self.node_status_lock:
                            self.node_status[node_url] = {
                                'status': 'online',
                                'last_check': time.time(),
                                'load': data.get('load', 0),
                                'available_workers': data.get('available_workers', 1)
                            }
                    else:
                        logger.warning(f"Nodo {node_url} respondió con estado {response.status}")
                        with self.node_status_lock:
                            self.node_status[node_url] = {
                                'status': 'error',
                                'last_check': time.time(),
                                'error': f"HTTP {response.status}"
                            }
            except Exception as e:
                logger.warning(f"Error al verificar nodo {node_url}: {str(e)}")
                with self.node_status_lock:
                    self.node_status[node_url] = {
                        'status': 'offline',
                        'last_check': time.time(),
                        'error': str(e)
                    }
        
        # Registrar estado general
        online_nodes = sum(1 for status in self.node_status.values() if status.get('status') == 'online')
        logger.debug(f"Estado de nodos: {online_nodes}/{len(self.nodes)} online")
    
    def shutdown(self, wait=True):
        """
        Detiene el ejecutor distribuido.
        
        Args:
            wait: Si es True, espera a que todas las tareas terminen
        """
        if not self.running:
            return
        
        logger.info("Deteniendo DistributedExecutor...")
        
        # Detener el ejecutor base
        super().shutdown(wait=False)  # No esperar aquí, lo haremos después
        
        # Cerrar sesión HTTP
        if self.session:
            future = asyncio.run_coroutine_threadsafe(self.session.close(), self.event_loop)
            if wait:
                future.result()  # Esperar a que se cierre la sesión
            self.session = None
        
        # Detener loop de eventos
        if self.event_loop and self.event_loop.is_running():
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            
            if self.event_loop_thread and self.event_loop_thread.is_alive() and wait:
                self.event_loop_thread.join(timeout=10)  # Esperar máximo 10 segundos
            
            self.event_loop = None
            self.event_loop_thread = None
        
        logger.info("DistributedExecutor detenido correctamente")
    
    def _get_executor_type(self) -> str:
        """
        Obtiene el tipo de ejecutor.
        
        Returns:
            str: 'distributed'
        """
        return 'distributed'
    
    def _execute_task(self, task: Task) -> None:
        """
        Ejecuta una tarea en un nodo remoto.
        
        Args:
            task: Tarea a ejecutar
        """
        logger.debug(f"Programando tarea {task.task_id} ({task.task_type}) para ejecución distribuida")
        
        # Actualizar estado y registrar en tareas activas
        task.status = TaskStatus.QUEUED
        
        with self.active_tasks_lock:
            self.active_tasks[task.task_id] = {
                'task': task,
                'status': TaskStatus.QUEUED,
                'queued_at': time.time()
            }
        
        # Buscar handler para el tipo de tarea
        handler = self.task_handlers.get(task.task_type)
        
        if handler is None:
            error_msg = f"No hay handler registrado para el tipo de tarea: {task.task_type}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Programar ejecución asíncrona
        asyncio.run_coroutine_threadsafe(
            self._execute_task_async(task),
            self.event_loop
        )
    
    async def _execute_task_async(self, task: Task) -> None:
        """
        Ejecuta una tarea de forma asíncrona en un nodo remoto.
        
        Args:
            task: Tarea a ejecutar
        """
        # Seleccionar nodo para la tarea
        node_url = await self._select_node_for_task(task)
        
        if not node_url:
            error_msg = "No hay nodos disponibles para ejecutar la tarea"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Actualizar estado a RUNNING
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        with self.active_tasks_lock:
            if task.task_id in self.active_tasks:
                self.active_tasks[task.task_id]['status'] = TaskStatus.RUNNING
                self.active_tasks[task.task_id]['start_time'] = task.started_at
        
        self._update_task_status(task.task_id, TaskStatus.RUNNING)
        
        # Registrar tarea en ejecución
        with self.running_tasks_lock:
            self.running_tasks[task.task_id] = node_url
        
        logger.info(f"Enviando tarea {task.task_id} ({task.task_type}) al nodo {node_url}")
        
        # Preparar datos de la tarea para envío
        task_data = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'data': task.data,
            'priority': task.priority.value if hasattr(task.priority, 'value') else task.priority,
            'created_at': task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
            'timeout': self.timeout
        }
        
        # Enviar tarea al nodo remoto
        retry_count = 0
        max_retries = self.max_retries
        
        while retry_count <= max_retries:
            try:
                # Enviar tarea al nodo
                async with self.session.post(
                    f"{node_url}/tasks",
                    json=task_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200 or response.status == 201:
                        # Tarea aceptada por el nodo
                        response_data = await response.json()
                        remote_task_id = response_data.get('remote_task_id')
                        
                        logger.info(f"Tarea {task.task_id} aceptada por nodo {node_url} (ID remoto: {remote_task_id})")
                        
                        # Iniciar polling para verificar estado
                        asyncio.create_task(
                            self._poll_task_status(task.task_id, node_url, remote_task_id)
                        )
                        
                        return
                    else:
                        # Error al enviar tarea
                        error_text = await response.text()
                        error_msg = f"Error al enviar tarea a nodo {node_url}: HTTP {response.status} - {error_text}"
                        logger.warning(error_msg)
                        
                        # Si es un error 5xx, reintentar con otro nodo
                        if 500 <= response.status < 600:
                            break  # Salir del bucle para probar otro nodo
                        else:
                            # Para otros errores, fallar directamente
                            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
                            
                            # Limpiar tarea de running_tasks
                            with self.running_tasks_lock:
                                if task.task_id in self.running_tasks:
                                    del self.running_tasks[task.task_id]
                            
                            return
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al enviar tarea {task.task_id} a nodo {node_url}")
                # Continuar con reintento
            
            except Exception as e:
                error_msg = f"Error al enviar tarea {task.task_id} a nodo {node_url}: {str(e)}"
                logger.warning(error_msg)
                # Continuar con reintento
            
            # Incrementar contador de reintentos
            retry_count += 1
            
            if retry_count <= max_retries:
                # Calcular tiempo de espera con backoff exponencial
                wait_time = self.retry_delay * (2 ** (retry_count - 1))
                # Añadir jitter (±25%)
                wait_time = wait_time * random.uniform(0.75, 1.25)
                
                logger.info(f"Reintentando envío de tarea {task.task_id} en {wait_time:.2f}s (intento {retry_count}/{max_retries})")
                
                # Esperar antes de reintentar
                await asyncio.sleep(wait_time)
                
                # Seleccionar un nuevo nodo para el reintento
                new_node_url = await self._select_node_for_task(task, exclude=[node_url])
                
                if new_node_url:
                    node_url = new_node_url
                    
                    # Actualizar nodo en running_tasks
                    with self.running_tasks_lock:
                        if task.task_id in self.running_tasks:
                            self.running_tasks[task.task_id] = node_url
                else:
                    logger.warning(f"No hay nodos alternativos disponibles para reintento de tarea {task.task_id}")
            else:
                # Máximo de reintentos alcanzado
                error_msg = f"Máximo de reintentos alcanzado para tarea {task.task_id}"
                logger.error(error_msg)
                self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
                
                # Limpiar tarea de running_tasks
                with self.running_tasks_lock:
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
    
    async def _select_node_for_task(self, task: Task, exclude: List[str] = None) -> Optional[str]:
        """
        Selecciona el mejor nodo para ejecutar una tarea.
        
        Args:
            task: Tarea a ejecutar
            exclude: Lista de nodos a excluir
            
        Returns:
            Optional[str]: URL del nodo seleccionado o None si no hay nodos disponibles
        """
        if exclude is None:
            exclude = []
        
        # Obtener nodos disponibles
        available_nodes = []
        
        with self.node_status_lock:
            for node_url, status in self.node_status.items():
                # Verificar si el nodo está online y no está en la lista de exclusión
                if (status.get('status') == 'online' and 
                    node_url not in exclude and
                    time.time() - status.get('last_check', 0) < 300):  # Nodo verificado en los últimos 5 minutos
                    
                    # Añadir a lista de nodos disponibles con su carga
                    available_nodes.append({
                        'url': node_url,
                        'load': status.get('load', 0),
                        'available_workers': status.get('available_workers', 1)
                    })
        
        if not available_nodes:
            # Si no hay nodos disponibles, intentar verificar nodos
            await self._check_nodes_health()
            
            # Volver a buscar nodos disponibles
            with self.node_status_lock:
                for node_url, status in self.node_status.items():
                    if (status.get('status') == 'online' and 
                        node_url not in exclude and
                        time.time() - status.get('last_check', 0) < 300):
                        
                        available_nodes.append({
                            'url': node_url,
                            'load': status.get('load', 0),
                            'available_workers': status.get('available_workers', 1)
                        })
        
        if not available_nodes:
            logger.warning("No hay nodos disponibles para ejecutar tareas")
            return None
        
        # Ordenar nodos por carga (menor a mayor) y disponibilidad de workers (mayor a menor)
        available_nodes.sort(key=lambda n: (n['load'], -n['available_workers']))
        
        # Seleccionar el mejor nodo (menor carga, más workers disponibles)
        selected_node = available_nodes[0]['url']
        
        logger.debug(f"Nodo seleccionado para tarea {task.task_id}: {selected_node} (carga: {available_nodes[0]['load']}, workers: {available_nodes[0]['available_workers']})")
        
        return selected_node
    
    async def _poll_task_status(self, task_id: str, node_url: str, remote_task_id: str) -> None:
        """
        Consulta periódicamente el estado de una tarea en un nodo remoto.
        
        Args:
            task_id: ID de la tarea local
            node_url: URL del nodo
            remote_task_id: ID de la tarea en el nodo remoto
        """
        poll_interval = 5  # segundos iniciales entre consultas
        max_poll_interval = 60  # máximo intervalo entre consultas
        
        while True:
            try:
                # Verificar si la tarea sigue en ejecución
                with self.running_tasks_lock:
                    if task_id not in self.running_tasks:
                        logger.info(f"Tarea {task_id} ya no está en ejecución, deteniendo polling")
                        return
                
                # Consultar estado de la tarea
                async with self.session.get(
                    f"{node_url}/tasks/{remote_task_id}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get('status')
                        
                        if status in ['completed', 'failed', 'cancelled']:
                            # Tarea terminada
                            if status == 'completed':
                                result = data.get('result')
                                logger.info(f"Tarea {task_id} completada en nodo {node_url}")
                                self._update_task_status(task_id, TaskStatus.COMPLETED, result=result)
                            
                            elif status == 'failed':
                                error = data.get('error', 'Error desconocido en nodo remoto')
                                logger.error(f"Tarea {task_id} falló en nodo {node_url}: {error}")
                                self._update_task_status(task_id, TaskStatus.FAILED, error=error)
                            
                            elif status == 'cancelled':
                                logger.info(f"Tarea {task_id} cancelada en nodo {node_url}")
                                self._update_task_status(task_id, TaskStatus.CANCELLED)
                            
                            # Limpiar tarea de running_tasks
                            with self.running_tasks_lock:
                                if task_id in self.running_tasks:
                                    del self.running_tasks[task_id]
                            
                            return
                        
                        elif status == 'running':
                            # Tarea en ejecución, continuar polling
                            progress = data.get('progress')
                            if progress:
                                logger.debug(f"Tarea {task_id} en ejecución en nodo {node_url} (progreso: {progress}%)")
                        
                        else:
                            # Estado desconocido, continuar polling
                            logger.warning(f"Estado desconocido para tarea {task_id} en nodo {node_url}: {status}")
                    
                    elif response.status == 404:
                        # Tarea no encontrada en el nodo
                        error_msg = f"Tarea {remote_task_id} no encontrada en nodo {node_url}"
                        logger.error(error_msg)
                        self._update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                        
                        # Limpiar tarea de running_tasks
                        with self.running_tasks_lock:
                            if task_id in self.running_tasks:
                                del self.running_tasks[task_id]
                        
                        return
                    
                    else:
                        # Error al consultar estado
                        error_text = await response.text()
                        logger.warning(f"Error al consultar estado de tarea {task_id} en nodo {node_url}: HTTP {response.status} - {error_text}")
            
            except Exception as e:
                logger.warning(f"Error durante polling de tarea {task_id} en nodo {node_url}: {str(e)}")
            
            # Incrementar intervalo de polling (hasta el máximo)
            poll_interval = min(poll_interval * 1.5, max_poll_interval)
            
            # Esperar hasta la próxima consulta
            await asyncio.sleep(poll_interval)
    
    async def _cancel_remote_task(self, task_id: str, node_url: str) -> bool:
        """
        Cancela una tarea en un nodo remoto.
        
        Args:
            task_id: ID de la tarea
            node_url: URL del nodo
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        try:
            # Obtener ID remoto de la tarea
            remote_task_id = None
            
            # Consultar ID remoto
            async with self.session.get(
                f"{node_url}/tasks",
                params={'local_task_id': task_id},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    tasks = data.get('tasks', [])
                    
                    if tasks:
                        remote_task_id = tasks[0].get('remote_task_id')
            
            if not remote_task_id:
                logger.warning(f"No se encontró ID remoto para tarea {task_id} en nodo {node_url}")
                return False
            
            # Cancelar tarea remota
            async with self.session.delete(
                f"{node_url}/tasks/{remote_task_id}",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    logger.info(f"Tarea {task_id} cancelada exitosamente en nodo {node_url}")
                    return True
                else:
                    error_text = await response.text()
                    logger.warning(f"Error al cancelar tarea {task_id} en nodo {node_url}: HTTP {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Error al cancelar tarea {task_id} en nodo {node_url}: {str(e)}")
            return False
    
    def _cancel_task_impl(self, task_id: str) -> bool:
        """
        Implementación específica de cancelación de tarea.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        with self.running_tasks_lock:
            if task_id not in self.running_tasks:
                return False
            
            node_url = self.running_tasks[task_id]
        
        # Programar cancelación asíncrona
        future = asyncio.run_coroutine_threadsafe(
            self._cancel_remote_task(task_id, node_url),
            self.event_loop
        )
        
        try:
            # Esperar resultado con timeout
            cancelled = future.result(timeout=30)
            return cancelled
        except Exception as e:
            logger.error(f"Error al cancelar tarea {task_id}: {str(e)}")
            return False
    
    def get_active_tasks(self) -> List[str]:
        """
        Obtiene las tareas actualmente en ejecución o en cola.
        
        Returns:
            List[str]: Lista de IDs de tareas activas
        """
        with self.active_tasks_lock, self.running_tasks_lock:
            # Combinar tareas activas y en ejecución
            active_tasks = list(self.active_tasks.keys())
            return active_tasks
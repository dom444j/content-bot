"""
Scheduler - Planificador de tareas y publicaciones

Este módulo gestiona la programación temporal del sistema:
- Planificación de creación de contenido
- Programación de publicaciones en horarios óptimos
- Gestión de tareas recurrentes (análisis, engagement)
- Coordinación de múltiples canales y plataformas
"""

import os
import sys
import json
import logging
import time
import datetime
import heapq
import threading
import random
import uuid
import backoff
import functools
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import schedule
from enum import Enum, auto

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'scheduler.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Scheduler')

# Definir enumeraciones para estados y prioridades de tareas
class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    PAUSED = auto()
    CANCELLED = auto()

class TaskPriority(Enum):
    CRITICAL = 1    # Tareas críticas (shadowban, problemas de monetización)
    HIGH = 2        # Tareas de alta prioridad (publicaciones programadas)
    MEDIUM = 3      # Tareas de prioridad media (creación de contenido)
    LOW = 4         # Tareas de baja prioridad (análisis, optimización)
    BACKGROUND = 5  # Tareas en segundo plano (limpieza, mantenimiento)

# Decorador para reintentos con backoff exponencial
def retry_with_backoff(max_tries=3, backoff_factor=2, jitter=None):
    """
    Decorador para reintentar funciones con backoff exponencial
    
    Args:
        max_tries: Número máximo de intentos
        backoff_factor: Factor de backoff exponencial
        jitter: Jitter aleatorio (None, 'full', 'equal')
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_tries:
                        logger.error(f"Máximo de reintentos alcanzado para {func.__name__}: {str(e)}")
                        raise
                    
                    # Calcular tiempo de espera con backoff exponencial
                    wait_time = backoff_factor ** retry_count
                    
                    # Aplicar jitter si está configurado
                    if jitter == 'full':
                        wait_time = random.uniform(0, wait_time)
                    elif jitter == 'equal':
                        wait_time = wait_time / 2 + random.uniform(0, wait_time / 2)
                    
                    logger.warning(f"Reintentando {func.__name__} en {wait_time:.2f}s (intento {retry_count}/{max_tries})")
                    time.sleep(wait_time)
        return wrapper
    return decorator

class Scheduler:
    """
    Planificador de tareas y publicaciones que gestiona la programación
    temporal del sistema de monetización.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Scheduler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el planificador si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Scheduler...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Cola de tareas programadas (priority queue)
        self.task_queue = []
        
        # Tareas recurrentes
        self.recurring_tasks = {}
        
        # Tareas en ejecución
        self.running_tasks = {}
        
        # Tareas fallidas pendientes de reintento
        self.retry_queue = []
        
        # Tareas pausadas
        self.paused_tasks = {}
        
        # Historial de tareas
        self.task_history = []
        
        # Configuración de horarios
        self.content_creation_hours = self.strategy.get('global_settings', {}).get('content_creation_hours', [0, 23])
        self.publishing_hours = self.strategy.get('global_settings', {}).get('publishing_hours', [17, 22])
        self.engagement_hours = self.strategy.get('global_settings', {}).get('engagement_hours', [8, 23])
        self.analysis_hours = self.strategy.get('global_settings', {}).get('analysis_hours', [0, 5])
        
        # Configuración de reintentos
        self.retry_config = self.strategy.get('task_management', {}).get('retry_config', {
            'max_retries': 3,
            'backoff_factor': 2,
            'jitter': 'equal',
            'retry_delay_base': 60  # segundos
        })
        
        # Límites de tareas concurrentes por tipo
        self.concurrency_limits = self.strategy.get('task_management', {}).get('concurrency_limits', {
            'creation': 3,
            'publishing': 5,
            'analysis': 2,
            'engagement': 3,
            'default': 10
        })
        
        # Contadores de tareas concurrentes
        self.concurrency_counters = {task_type: 0 for task_type in self.concurrency_limits}
        
        # Bloqueos para operaciones thread-safe
        self.queue_lock = threading.RLock()
        self.counter_lock = threading.RLock()
        
        # Iniciar thread de planificación
        self.scheduler_thread = None
        self.retry_thread = None
        self.stop_event = threading.Event()
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'avg_completion_time': 0,
            'total_completion_time': 0
        }
        
        # Cargar estado persistente si existe
        self._load_persistent_state()
        
        self._initialized = True
        logger.info("Scheduler inicializado correctamente")
    
    @retry_with_backoff(max_tries=3, backoff_factor=2, jitter='equal')
    def _load_strategy(self) -> Dict:
        """Carga la configuración de estrategia desde el archivo JSON"""
        try:
            if os.path.exists(self.strategy_file):
                with open(self.strategy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estrategia no encontrado: {self.strategy_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar estrategia: {str(e)}")
            raise
    
    def _load_persistent_state(self):
        """Carga el estado persistente del planificador desde la base de conocimiento"""
        try:
            state = self.kb.get_from_mongodb('scheduler_state', {'type': 'state'})
            if state:
                # Restaurar tareas pendientes
                for task_data in state.get('pending_tasks', []):
                    # Convertir string de fecha a objeto datetime
                    execution_time = datetime.datetime.fromisoformat(task_data.get('scheduled_time'))
                    task_id = task_data.get('id')
                    
                    # Añadir a la cola con prioridad
                    with self.queue_lock:
                        heapq.heappush(self.task_queue, (
                            execution_time, 
                            self._priority_to_int(task_data.get('priority', TaskPriority.MEDIUM.value)),
                            task_id, 
                            task_data
                        ))
                
                # Restaurar tareas pausadas
                self.paused_tasks = {task['id']: task for task in state.get('paused_tasks', [])}
                
                # Restaurar métricas
                self.performance_metrics = state.get('metrics', self.performance_metrics)
                
                logger.info(f"Estado persistente cargado: {len(self.task_queue)} tareas pendientes, {len(self.paused_tasks)} pausadas")
            else:
                logger.info("No se encontró estado persistente previo")
        except Exception as e:
            logger.error(f"Error al cargar estado persistente: {str(e)}")
    
    def _save_persistent_state(self):
        """Guarda el estado actual del planificador en la base de conocimiento"""
        try:
            # Extraer tareas pendientes (sin modificar la cola original)
            pending_tasks = []
            with self.queue_lock:
                # Crear una copia de la cola para no modificar la original
                queue_copy = self.task_queue.copy()
                
                # Extraer todas las tareas
                while queue_copy:
                    _, _, _, task = heapq.heappop(queue_copy)
                    pending_tasks.append(task)
            
            # Preparar estado para guardar
            state = {
                'type': 'state',
                'pending_tasks': pending_tasks,
                'paused_tasks': list(self.paused_tasks.values()),
                'metrics': self.performance_metrics,
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            # Guardar en la base de conocimiento
            self.kb.save_to_mongodb('scheduler_state', state, {'type': 'state'})
            logger.info("Estado del planificador guardado correctamente")
        except Exception as e:
            logger.error(f"Error al guardar estado persistente: {str(e)}")
    
    def start(self):
        """Inicia el planificador en un thread separado"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("El planificador ya está en ejecución")
            return False
        
        self.stop_event.clear()
        
        # Iniciar thread principal de planificación
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Iniciar thread de procesamiento de reintentos
        self.retry_thread = threading.Thread(target=self._retry_loop)
        self.retry_thread.daemon = True
        self.retry_thread.start()
        
        # Configurar tareas recurrentes
        self._setup_recurring_tasks()
        
        # Configurar guardado periódico del estado
        schedule.every(15).minutes.do(self._save_persistent_state)
        
        logger.info("Planificador iniciado")
        return True
    
    def stop(self):
        """Detiene el planificador"""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("El planificador no está en ejecución")
            return False
        
        # Señalizar parada
        self.stop_event.set()
        
        # Esperar a que los threads terminen
        self.scheduler_thread.join(timeout=5)
        if self.retry_thread and self.retry_thread.is_alive():
            self.retry_thread.join(timeout=5)
        
        # Limpiar tareas recurrentes
        schedule.clear()
        
        # Guardar estado antes de terminar
        self._save_persistent_state()
        
        logger.info("Planificador detenido")
        return True
    
    def _scheduler_loop(self):
        """Loop principal del planificador"""
        while not self.stop_event.is_set():
            try:
                # Ejecutar tareas programadas
                self._process_task_queue()
                
                # Ejecutar tareas recurrentes
                schedule.run_pending()
                
                # Verificar límites de concurrencia y liberar recursos si es necesario
                self._check_running_tasks()
                
                # Dormir brevemente
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error en el loop principal del planificador: {str(e)}")
                time.sleep(5)  # Esperar un poco más en caso de error
    
    def _retry_loop(self):
        """Loop para procesar la cola de reintentos"""
        while not self.stop_event.is_set():
            try:
                # Procesar cola de reintentos
                self._process_retry_queue()
                
                # Dormir brevemente
                time.sleep(5)  # Verificar reintentos cada 5 segundos
            except Exception as e:
                logger.error(f"Error en el loop de reintentos: {str(e)}")
                time.sleep(10)  # Esperar un poco más en caso de error
    
    def _setup_recurring_tasks(self):
        """Configura tareas recurrentes basadas en la estrategia"""
        # Tarea diaria de análisis (a las 3 AM)
        schedule.every().day.at("03:00").do(self._create_task, 
            task_type="analysis", 
            task_data={"type": "daily_analysis"},
            priority=TaskPriority.MEDIUM
        )
        
        # Tarea diaria de detección de tendencias (cada 6 horas)
        for hour in [6, 12, 18, 0]:
            schedule.every().day.at(f"{hour:02d}:00").do(self._create_task,
                task_type="trends",
                task_data={"type": "trend_detection"},
                priority=TaskPriority.HIGH
            )
        
        # Tarea de engagement (cada 2 horas durante horas de engagement)
        for hour in range(self.engagement_hours[0], self.engagement_hours[1] + 1, 2):
            schedule.every().day.at(f"{hour:02d}:00").do(self._create_task,
                task_type="engagement",
                task_data={"type": "comment_response"},
                priority=TaskPriority.MEDIUM
            )
        
        # Tareas de creación de contenido (basadas en nichos activos)
        active_niches = [niche for niche, config in self.strategy.get('niches', {}).items() 
                         if config.get('active', False)]
        
        for niche in active_niches:
            # Determinar frecuencia de publicación
            platforms = self.strategy.get('niches', {}).get(niche, {}).get('platforms', [])
            
            for platform in platforms:
                frequency = self.strategy.get('niches', {}).get(niche, {}).get('posting_frequency', {}).get(platform, 0)
                
                if frequency > 0:
                    # Convertir frecuencia a intervalo en días
                    interval = max(1, int(1 / frequency))
                    
                    # Programar tarea de creación
                    creation_hour = random.randint(self.content_creation_hours[0], self.content_creation_hours[1])
                    
                    if interval == 1:
                        # Diario
                        schedule.every().day.at(f"{creation_hour:02d}:00").do(self._create_task,
                            task_type="creation",
                            task_data={"niche": niche, "platform": platform},
                            priority=TaskPriority.MEDIUM
                        )
                    else:
                        # Cada X días
                        schedule.every(interval).days.at(f"{creation_hour:02d}:00").do(self._create_task,
                            task_type="creation",
                            task_data={"niche": niche, "platform": platform},
                            priority=TaskPriority.MEDIUM
                        )
        
        # Tareas de verificación de shadowban (cada 12 horas)
        schedule.every(12).hours.do(self._create_task,
            task_type="compliance",
            task_data={"type": "shadowban_check"},
            priority=TaskPriority.HIGH
        )
        
        # Tarea de optimización de CTAs (semanal)
        schedule.every().monday.at("04:00").do(self._create_task,
            task_type="optimization",
            task_data={"type": "cta_optimization"},
            priority=TaskPriority.LOW
        )
        
        # Tarea de redistribución de tráfico (semanal)
        schedule.every().sunday.at("05:00").do(self._create_task,
            task_type="optimization",
            task_data={"type": "traffic_redistribution"},
            priority=TaskPriority.MEDIUM
        )
        
        # Tarea de limpieza de caché (diaria)
        schedule.every().day.at("02:00").do(self._create_task,
            task_type="maintenance",
            task_data={"type": "cache_cleanup"},
            priority=TaskPriority.BACKGROUND
        )
        
        logger.info(f"Configuradas {len(schedule.jobs)} tareas recurrentes")
    
    def _process_task_queue(self):
        """Procesa la cola de tareas programadas"""
        now = datetime.datetime.now()
        
        # Procesar tareas cuyo tiempo de ejecución ha llegado
        with self.queue_lock:
            while self.task_queue and (
                self.task_queue[0][0] <= now  # Tiempo de ejecución ha llegado
            ):
                # Extraer tarea de la cola
                execution_time, priority, task_id, task = heapq.heappop(self.task_queue)
                
                # Verificar si la tarea ya está en ejecución
                if task_id in self.running_tasks:
                    logger.warning(f"Tarea {task_id} ya en ejecución, omitiendo")
                    continue
                
                # Verificar límites de concurrencia
                task_type = task.get('type', 'default')
                if not self._can_execute_task(task_type):
                    logger.info(f"Límite de concurrencia alcanzado para {task_type}, reponiendo tarea {task_id}")
                    # Reponer la tarea en la cola con un pequeño retraso
                    new_execution_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
                    heapq.heappush(self.task_queue, (new_execution_time, priority, task_id, task))
                    continue
                
                # Ejecutar tarea en un thread separado
                self._execute_task(task_id, task)
    
    def _can_execute_task(self, task_type: str) -> bool:
        """Verifica si se puede ejecutar una tarea según límites de concurrencia"""
        with self.counter_lock:
            limit = self.concurrency_limits.get(task_type, self.concurrency_limits.get('default', 10))
            current = self.concurrency_counters.get(task_type, 0)
            
            if current >= limit:
                return False
            
            # Incrementar contador
            self.concurrency_counters[task_type] = current + 1
            return True
    
    def _release_concurrency_slot(self, task_type: str):
        """Libera un slot de concurrencia para un tipo de tarea"""
        with self.counter_lock:
            current = self.concurrency_counters.get(task_type, 0)
            self.concurrency_counters[task_type] = max(0, current - 1)
    
    def _check_running_tasks(self):
        """Verifica tareas en ejecución y libera recursos si es necesario"""
        # Copiar las claves para evitar modificar durante la iteración
        task_ids = list(self.running_tasks.keys())
        
        for task_id in task_ids:
            task_info = self.running_tasks.get(task_id)
            if not task_info:
                continue
                
            # Verificar si el thread sigue vivo
            thread = task_info.get('thread')
            if thread and not thread.is_alive():
                # El thread ha terminado pero no se ha actualizado el estado
                # Esto puede ocurrir si hubo un error no controlado
                logger.warning(f"Tarea {task_id} terminó sin actualizar estado, limpiando")
                
                task = task_info.get('task', {})
                task_type = task.get('type', 'default')
                
                # Liberar slot de concurrencia
                self._release_concurrency_slot(task_type)
                
                # Eliminar de tareas en ejecución
                del self.running_tasks[task_id]
                
                # Registrar como fallida
                self._task_completed(task_id, task, success=False, 
                                    error="Thread terminó sin actualizar estado")
    
    def _execute_task(self, task_id: str, task: Dict):
        """Ejecuta una tarea en un thread separado"""
        task_thread = threading.Thread(
            target=self._task_worker,
            args=(task_id, task)
        )
        task_thread.daemon = True
        task_thread.start()
        
        # Registrar tarea en ejecución
        self.running_tasks[task_id] = {
            'task': task,
            'thread': task_thread,
            'start_time': datetime.datetime.now()
        }
        
        logger.info(f"Iniciada tarea {task_id}: {task['type']}")
    
    def _task_worker(self, task_id: str, task: Dict):
        """Worker que ejecuta una tarea"""
        try:
            # Actualizar estado de la tarea
            task['status'] = TaskStatus.RUNNING.name
            
            # Aquí se implementaría la lógica para ejecutar diferentes tipos de tareas
            task_type = task['type']
            task_data = task['data']
            
            logger.info(f"Ejecutando tarea {task_id} de tipo {task_type}: {task_data}")
            
            # Simular tiempo de ejecución
            time.sleep(2)
            
            # Simular éxito/fallo aleatorio para pruebas
            if random.random() < 0.9:  # 90% de éxito
                # Registrar finalización exitosa
                self._task_completed(task_id, task, success=True)
            else:
                # Simular error
                raise Exception("Error simulado para pruebas")
            
        except Exception as e:
            logger.error(f"Error al ejecutar tarea {task_id}: {str(e)}")
            # Registrar error y programar reintento si es necesario
            self._handle_task_failure(task_id, task, str(e))
        finally:
            # Liberar slot de concurrencia
            self._release_concurrency_slot(task.get('type', 'default'))
            
            # Eliminar de tareas en ejecución si aún está ahí
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _handle_task_failure(self, task_id: str, task: Dict, error: str):
        """Maneja el fallo de una tarea y programa reintento si es necesario"""
        # Obtener número de reintentos actuales
        retries = task.get('retries', 0)
        max_retries = self.retry_config.get('max_retries', 3)
        
        if retries < max_retries:
            # Incrementar contador de reintentos
            task['retries'] = retries + 1
            task['status'] = TaskStatus.RETRYING.name
            task['last_error'] = error
            
            # Calcular tiempo de espera con backoff exponencial
            backoff_factor = self.retry_config.get('backoff_factor', 2)
            retry_delay_base = self.retry_config.get('retry_delay_base', 60)
            
            delay = retry_delay_base * (backoff_factor ** retries)
            
            # Aplicar jitter si está configurado
            jitter = self.retry_config.get('jitter')
            if jitter == 'full':
                delay = random.uniform(0, delay)
            elif jitter == 'equal':
                delay = delay / 2 + random.uniform(0, delay / 2)
            
            # Calcular tiempo de reintento
            retry_time = datetime.datetime.now() + datetime.timedelta(seconds=delay)
            
            # Añadir a la cola de reintentos
            with self.queue_lock:
                self.retry_queue.append((retry_time, task_id, task))
            
            logger.info(f"Tarea {task_id} fallida, programado reintento {retries+1}/{max_retries} para {retry_time.isoformat()}")
            
            # Actualizar métricas
            self.performance_metrics['tasks_retried'] += 1
        else:
            # Máximo de reintentos alcanzado, marcar como fallida definitivamente
            self._task_completed(task_id, task, success=False, error=error)
    
    def _process_retry_queue(self):
        """Procesa la cola de reintentos"""
        now = datetime.datetime.now()
        
        # Ordenar cola de reintentos por tiempo
        with self.queue_lock:
            self.retry_queue.sort(key=lambda x: x[0])
            
            # Procesar tareas cuyo tiempo de reintento ha llegado
            while self.retry_queue and self.retry_queue[0][0] <= now:
                retry_time, task_id, task = self.retry_queue.pop(0)
                
                # Verificar si la tarea ya está en ejecución (por si acaso)
                if task_id in self.running_tasks:
                    logger.warning(f"Tarea {task_id} ya en ejecución, omitiendo reintento")
                    continue
                
                # Verificar límites de concurrencia
                task_type = task.get('type', 'default')
                if not self._can_execute_task(task_type):
                    logger.info(f"Límite de concurrencia alcanzado para {task_type}, reponiendo reintento {task_id}")
                    # Reponer en la cola con un pequeño retraso
                    new_retry_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
                    self.retry_queue.append((new_retry_time, task_id, task))
                    continue
                
                # Ejecutar reintento
                logger.info(f"Ejecutando reintento {task.get('retries')}/{self.retry_config.get('max_retries')} para tarea {task_id}")
                self._execute_task(task_id, task)
    
    def _task_completed(self, task_id: str, task: Dict, success: bool, error: str = None):
        """Registra la finalización de una tarea"""
        end_time = datetime.datetime.now()
        
        # Obtener información de inicio
        start_time = self.running_tasks.get(task_id, {}).get('start_time', end_time)
        
        # Actualizar estado de la tarea
        task['status'] = TaskStatus.COMPLETED.name if success else TaskStatus.FAILED.name
        
        if error:
            task['error'] = error
        
        # Calcular duración
        duration = (end_time - start_time).total_seconds()
        
        # Registrar en historial
        task_record = {
            'id': task_id,
            'type': task.get('type'),
            'data': task.get('data'),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': duration,
            'success': success,
            'retries': task.get('retries', 0)
        }
        
        if error:
            task_record['error'] = error
        
        self.task_history.append(task_record)
        
        # Limitar tamaño del historial
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
        
        # Actualizar métricas
        if success:
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['total_completion_time'] += duration
            self.performance_metrics['avg_completion_time'] = (
                self.performance_metrics['total_completion_time'] / 
                self.performance_metrics['tasks_completed']
            )
        else:
            self.performance_metrics['tasks_failed'] += 1
        
        logger.info(f"Tarea {task_id} completada. Éxito: {success}, Duración: {duration:.2f}s")
        
        # Guardar estado periódicamente (cada 10 tareas completadas)
        if (self.performance_metrics['tasks_completed'] + 
            self.performance_metrics['tasks_failed']) % 10 == 0:
            self._save_persistent_state()
    
    def _priority_to_int(self, priority: Union[int, str, TaskPriority]) -> int:
        """Convierte diferentes formatos de prioridad a entero para la cola"""
        if isinstance(priority, int):
            return priority
        elif isinstance(priority, str):
            try:
                return TaskPriority[priority.upper()].value
            except (KeyError, AttributeError):
                return TaskPriority.MEDIUM.value
        elif isinstance(priority, TaskPriority):
            return priority.value
        else:
            return TaskPriority.MEDIUM.value
    
        def schedule_task(self, task_type: str, task_data: Dict, execution_time: datetime.datetime = None, 
                     priority: Union[int, str, TaskPriority] = TaskPriority.MEDIUM, task_id: str = None) -> str:
        """
        Programa una tarea para ejecución futura
        
        Args:
            task_type: Tipo de tarea (creation, publishing, analysis, etc.)
            task_data: Datos específicos de la tarea
            execution_time: Tiempo de ejecución (si es None, se ejecuta inmediatamente)
            priority: Prioridad (TaskPriority enum, string o int)
            task_id: ID de tarea (si es None, se genera automáticamente)
            
        Returns:
            ID de la tarea programada
        """
        # Generar ID único si no se proporciona
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Establecer tiempo de ejecución si no se proporciona
        if execution_time is None:
            execution_time = datetime.datetime.now()
        
        # Ajustar prioridad según el contexto del sistema de monetización
        adjusted_priority = self._adjust_priority_by_context(task_type, task_data, priority)
        
        # Crear objeto de tarea
        task = {
            'id': task_id,
            'type': task_type,
            'data': task_data,
            'status': TaskStatus.PENDING.name,
            'priority': adjusted_priority.name if isinstance(adjusted_priority, TaskPriority) else adjusted_priority,
            'scheduled_time': execution_time.isoformat(),
            'created_at': datetime.datetime.now().isoformat(),
            'retries': 0,
            'dependencies': task_data.get('dependencies', []),
            'timeout': task_data.get('timeout', 3600),  # Tiempo máximo de ejecución en segundos (1 hora por defecto)
            'tags': task_data.get('tags', []),
            'metadata': {
                'source': task_data.get('source', 'manual'),
                'user': task_data.get('user', 'system'),
                'campaign': task_data.get('campaign', None),
                'niche': task_data.get('niche', None),
                'platform': task_data.get('platform', None)
            }
        }
        
        # Verificar dependencias
        if not self._check_dependencies(task):
            logger.warning(f"Tarea {task_id} tiene dependencias no resueltas, marcando como pendiente de dependencias")
            task['status'] = 'DEPENDENCY_PENDING'
            # Guardar en tareas pausadas hasta que se resuelvan las dependencias
            self.paused_tasks[task_id] = task
            return task_id
        
        # Añadir a la cola con prioridad
        priority_value = self._priority_to_int(adjusted_priority)
        
        with self.queue_lock:
            heapq.heappush(self.task_queue, (
                execution_time,
                priority_value,
                task_id,
                task
            ))
        
        # Actualizar métricas
        self.performance_metrics['tasks_scheduled'] += 1
        
        logger.info(f"Tarea {task_id} programada para {execution_time.isoformat()} con prioridad {adjusted_priority}")
        
        return task_id
    
    def _adjust_priority_by_context(self, task_type: str, task_data: Dict, 
                                   base_priority: Union[int, str, TaskPriority]) -> TaskPriority:
        """
        Ajusta la prioridad de una tarea según el contexto del sistema de monetización
        
        Args:
            task_type: Tipo de tarea
            task_data: Datos de la tarea
            base_priority: Prioridad base
            
        Returns:
            Prioridad ajustada
        """
        # Convertir a objeto TaskPriority
        if isinstance(base_priority, int):
            try:
                base_priority = TaskPriority(base_priority)
            except ValueError:
                base_priority = TaskPriority.MEDIUM
        elif isinstance(base_priority, str):
            try:
                base_priority = TaskPriority[base_priority.upper()]
            except KeyError:
                base_priority = TaskPriority.MEDIUM
        
        # Reglas de ajuste de prioridad basadas en el contexto
        
        # 1. Tareas relacionadas con shadowban o problemas de monetización son CRÍTICAS
        if task_type == 'compliance' and task_data.get('type') in ['shadowban_check', 'monetization_issue']:
            return TaskPriority.CRITICAL
        
        # 2. Tareas de publicación programadas con fecha cercana son ALTAS
        if task_type == 'publishing':
            # Si la publicación está programada para menos de 1 hora, es CRÍTICA
            scheduled_time = task_data.get('scheduled_time')
            if scheduled_time:
                try:
                    scheduled_dt = datetime.datetime.fromisoformat(scheduled_time)
                    time_diff = (scheduled_dt - datetime.datetime.now()).total_seconds()
                    if time_diff < 3600:  # Menos de 1 hora
                        return TaskPriority.CRITICAL
                    elif time_diff < 7200:  # Menos de 2 horas
                        return TaskPriority.HIGH
                except (ValueError, TypeError):
                    pass
        
        # 3. Tareas de tendencias virales son ALTAS
        if task_type == 'trends' and task_data.get('viral_score', 0) > 0.7:
            return TaskPriority.HIGH
        
        # 4. Tareas de nichos con alto ROI tienen mayor prioridad
        niche = task_data.get('niche')
        if niche:
            niche_roi = self.strategy.get('niches', {}).get(niche, {}).get('roi', 0)
            if niche_roi > 0.5:  # ROI > 50%
                # Aumentar prioridad en 1 nivel (sin superar CRITICAL)
                priority_value = min(base_priority.value - 1, TaskPriority.CRITICAL.value)
                return TaskPriority(priority_value)
        
        # 5. Tareas de plataformas con mayor rendimiento tienen mayor prioridad
        platform = task_data.get('platform')
        if platform:
            platform_performance = self.strategy.get('platforms', {}).get(platform, {}).get('performance_score', 0)
            if platform_performance > 0.7:  # Rendimiento > 70%
                # Aumentar prioridad en 1 nivel (sin superar CRITICAL)
                priority_value = min(base_priority.value - 1, TaskPriority.CRITICAL.value)
                return TaskPriority(priority_value)
        
        # 6. Tareas de engagement en contenido viral son ALTAS
        if task_type == 'engagement' and task_data.get('content_viral_score', 0) > 0.8:
            return TaskPriority.HIGH
        
        # 7. Tareas de optimización de CTAs son MEDIAS por defecto
        if task_type == 'optimization' and task_data.get('type') == 'cta_optimization':
            return TaskPriority.MEDIUM
        
        # 8. Tareas de mantenimiento son BACKGROUND por defecto
        if task_type == 'maintenance':
            return TaskPriority.BACKGROUND
        
        # Devolver prioridad base si no hay ajustes
        return base_priority
    
    def _check_dependencies(self, task: Dict) -> bool:
        """
        Verifica si todas las dependencias de una tarea están resueltas
        
        Args:
            task: Tarea a verificar
            
        Returns:
            True si todas las dependencias están resueltas, False en caso contrario
        """
        dependencies = task.get('dependencies', [])
        if not dependencies:
            return True
        
        # Verificar cada dependencia
        for dep_id in dependencies:
            # Buscar en historial de tareas completadas
            found_completed = False
            for hist_task in self.task_history:
                if hist_task['id'] == dep_id and hist_task.get('success', False):
                    found_completed = True
                    break
            
            if not found_completed:
                return False
        
        return True
    
    def _create_task(self, task_type: str, task_data: Dict, priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """
        Crea y programa una tarea (usado por tareas recurrentes)
        
        Args:
            task_type: Tipo de tarea
            task_data: Datos de la tarea
            priority: Prioridad
            
        Returns:
            ID de la tarea programada
        """
        return self.schedule_task(task_type, task_data, priority=priority)
    
    def get_next_task(self) -> Optional[Dict]:
        """
        Obtiene la siguiente tarea de mayor prioridad sin extraerla de la cola
        
        Returns:
            Tarea o None si no hay tareas pendientes
        """
        with self.queue_lock:
            if not self.task_queue:
                return None
            
            # Obtener la tarea de mayor prioridad (sin extraerla)
            _, _, _, task = self.task_queue[0]
            return task
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """
        Busca una tarea por su ID en todas las colas
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Tarea o None si no se encuentra
        """
        # Buscar en tareas en ejecución
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]['task']
        
        # Buscar en tareas pausadas
        if task_id in self.paused_tasks:
            return self.paused_tasks[task_id]
        
        # Buscar en cola de tareas
        with self.queue_lock:
            for _, _, tid, task in self.task_queue:
                if tid == task_id:
                    return task
        
        # Buscar en cola de reintentos
        with self.queue_lock:
            for _, tid, task in self.retry_queue:
                if tid == task_id:
                    return task
        
        # Buscar en historial
        for task in self.task_history:
            if task['id'] == task_id:
                return task
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea programada
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            True si se canceló correctamente, False en caso contrario
        """
        # Buscar en tareas en ejecución
        if task_id in self.running_tasks:
            logger.warning(f"No se puede cancelar tarea {task_id} porque ya está en ejecución")
            return False
        
        # Buscar en tareas pausadas
        if task_id in self.paused_tasks:
            task = self.paused_tasks[task_id]
            task['status'] = TaskStatus.CANCELLED.name
            del self.paused_tasks[task_id]
            self.task_history.append(task)
            logger.info(f"Tarea pausada {task_id} cancelada")
            return True
        
        # Buscar en cola de tareas
        with self.queue_lock:
            new_queue = []
            found = False
            
            for item in self.task_queue:
                _, _, tid, task = item
                if tid == task_id:
                    found = True
                    task['status'] = TaskStatus.CANCELLED.name
                    self.task_history.append(task)
                else:
                    new_queue.append(item)
            
            if found:
                # Reconstruir la cola sin la tarea cancelada
                self.task_queue = new_queue
                heapq.heapify(self.task_queue)
                logger.info(f"Tarea programada {task_id} cancelada")
                return True
        
        # Buscar en cola de reintentos
        with self.queue_lock:
            new_retry_queue = []
            found = False
            
            for item in self.retry_queue:
                _, tid, task = item
                if tid == task_id:
                    found = True
                    task['status'] = TaskStatus.CANCELLED.name
                    self.task_history.append(task)
                else:
                    new_retry_queue.append(item)
            
            if found:
                self.retry_queue = new_retry_queue
                logger.info(f"Tarea de reintento {task_id} cancelada")
                return True
        
        logger.warning(f"No se encontró tarea {task_id} para cancelar")
        return False
    
    def pause_task(self, task_id: str) -> bool:
        """
        Pausa una tarea programada
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            True si se pausó correctamente, False en caso contrario
        """
        # Buscar en tareas en ejecución
        if task_id in self.running_tasks:
            logger.warning(f"No se puede pausar tarea {task_id} porque ya está en ejecución")
            return False
        
        # Buscar en cola de tareas
        with self.queue_lock:
            new_queue = []
            found = False
            
            for item in self.task_queue:
                _, _, tid, task = item
                if tid == task_id:
                    found = True
                    task['status'] = TaskStatus.PAUSED.name
                    self.paused_tasks[task_id] = task
                else:
                    new_queue.append(item)
            
            if found:
                # Reconstruir la cola sin la tarea pausada
                self.task_queue = new_queue
                heapq.heapify(self.task_queue)
                logger.info(f"Tarea programada {task_id} pausada")
                return True
        
        # Buscar en cola de reintentos
        with self.queue_lock:
            new_retry_queue = []
            found = False
            
            for item in self.retry_queue:
                _, tid, task = item
                if tid == task_id:
                    found = True
                    task['status'] = TaskStatus.PAUSED.name
                    self.paused_tasks[task_id] = task
                else:
                    new_retry_queue.append(item)
            
            if found:
                self.retry_queue = new_retry_queue
                logger.info(f"Tarea de reintento {task_id} pausada")
                return True
        
        logger.warning(f"No se encontró tarea {task_id} para pausar")
        return False
    
    def resume_task(self, task_id: str, execution_time: datetime.datetime = None) -> bool:
        """
        Reanuda una tarea pausada
        
        Args:
            task_id: ID de la tarea
            execution_time: Nuevo tiempo de ejecución (opcional)
            
        Returns:
            True si se reanudó correctamente, False en caso contrario
        """
        if task_id not in self.paused_tasks:
            logger.warning(f"No se encontró tarea pausada {task_id}")
            return False
        
        # Obtener tarea pausada
        task = self.paused_tasks[task_id]
        
        # Actualizar estado
        task['status'] = TaskStatus.PENDING.name
        
        # Actualizar tiempo de ejecución si se proporciona
        if execution_time:
            task['scheduled_time'] = execution_time.isoformat()
        else:
            # Si no se proporciona, usar el tiempo original o ahora
            try:
                execution_time = datetime.datetime.fromisoformat(task['scheduled_time'])
            except (ValueError, KeyError):
                execution_time = datetime.datetime.now()
        
        # Añadir a la cola con prioridad
        priority = self._priority_to_int(task.get('priority', TaskPriority.MEDIUM.value))
        
        with self.queue_lock:
            heapq.heappush(self.task_queue, (
                execution_time,
                priority,
                task_id,
                task
            ))
        
        # Eliminar de tareas pausadas
        del self.paused_tasks[task_id]
        
        logger.info(f"Tarea {task_id} reanudada para {execution_time.isoformat()}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Obtiene el estado actual de una tarea
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Estado de la tarea o None si no se encuentra
        """
        task = self.get_task_by_id(task_id)
        if task:
            return task.get('status')
        return None
    
    def get_pending_tasks(self, task_type: str = None, limit: int = 100) -> List[Dict]:
        """
        Obtiene las tareas pendientes, opcionalmente filtradas por tipo
        
        Args:
            task_type: Tipo de tarea para filtrar (opcional)
            limit: Número máximo de tareas a devolver
            
        Returns:
            Lista de tareas pendientes
        """
        pending_tasks = []
        
        # Copiar cola de tareas para no modificarla
        with self.queue_lock:
            queue_copy = self.task_queue.copy()
        
        # Extraer tareas
        while queue_copy and len(pending_tasks) < limit:
            _, _, _, task = heapq.heappop(queue_copy)
            if task_type is None or task.get('type') == task_type:
                pending_tasks.append(task)
        
        return pending_tasks
    
    def get_running_tasks(self, task_type: str = None) -> List[Dict]:
        """
        Obtiene las tareas en ejecución, opcionalmente filtradas por tipo
        
        Args:
            task_type: Tipo de tarea para filtrar (opcional)
            
        Returns:
            Lista de tareas en ejecución
        """
        running_tasks = []
        
        for task_id, task_info in self.running_tasks.items():
            task = task_info.get('task', {})
            if task_type is None or task.get('type') == task_type:
                # Añadir información de tiempo de ejecución
                task_with_runtime = task.copy()
                start_time = task_info.get('start_time')
                if start_time:
                    runtime = (datetime.datetime.now() - start_time).total_seconds()
                    task_with_runtime['runtime'] = runtime
                
                running_tasks.append(task_with_runtime)
        
        return running_tasks
    
    def get_task_history(self, task_type: str = None, success: bool = None, 
                        limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Obtiene el historial de tareas, con opciones de filtrado y paginación
        
        Args:
            task_type: Tipo de tarea para filtrar (opcional)
            success: Filtrar por éxito/fallo (opcional)
            limit: Número máximo de tareas a devolver
            offset: Desplazamiento para paginación
            
        Returns:
            Lista de tareas del historial
        """
        # Filtrar historial
        filtered_history = self.task_history
        
        if task_type is not None:
            filtered_history = [t for t in filtered_history if t.get('type') == task_type]
        
        if success is not None:
            filtered_history = [t for t in filtered_history if t.get('success') == success]
        
        # Aplicar paginación
        paginated_history = filtered_history[offset:offset+limit]
        
        return paginated_history
    
    def get_performance_metrics(self) -> Dict:
        """
        Obtiene las métricas de rendimiento del planificador
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        # Añadir métricas adicionales
        metrics = self.performance_metrics.copy()
        
        # Añadir contadores actuales
        metrics['pending_tasks'] = len(self.task_queue)
        metrics['running_tasks'] = len(self.running_tasks)
        metrics['paused_tasks'] = len(self.paused_tasks)
        metrics['retry_queue'] = len(self.retry_queue)
        
        # Calcular tasa de éxito
        total_tasks = metrics['tasks_completed'] + metrics['tasks_failed']
        if total_tasks > 0:
            metrics['success_rate'] = metrics['tasks_completed'] / total_tasks
        else:
            metrics['success_rate'] = 0
        
        # Añadir timestamp
        metrics['timestamp'] = datetime.datetime.now().isoformat()
        
        return metrics
    
    def bulk_schedule_tasks(self, tasks: List[Dict]) -> List[str]:
        """
        Programa múltiples tareas en lote
        
        Args:
            tasks: Lista de diccionarios con información de tareas
                Cada diccionario debe tener: type, data, execution_time (opcional), priority (opcional)
                
        Returns:
            Lista de IDs de tareas programadas
        """
        task_ids = []
        
        for task_info in tasks:
            task_type = task_info.get('type')
            task_data = task_info.get('data', {})
            execution_time = task_info.get('execution_time')
            priority = task_info.get('priority', TaskPriority.MEDIUM)
            
            if execution_time and isinstance(execution_time, str):
                try:
                    execution_time = datetime.datetime.fromisoformat(execution_time)
                except ValueError:
                    execution_time = None
            
            task_id = self.schedule_task(
                task_type=task_type,
                task_data=task_data,
                execution_time=execution_time,
                priority=priority
            )
            
            task_ids.append(task_id)
        
        logger.info(f"Programadas {len(task_ids)} tareas en lote")
        return task_ids
    
    def reschedule_task(self, task_id: str, execution_time: datetime.datetime, 
                       priority: Union[int, str, TaskPriority] = None) -> bool:
        """
        Reprograma una tarea existente
        
        Args:
            task_id: ID de la tarea
            execution_time: Nuevo tiempo de ejecución
            priority: Nueva prioridad (opcional)
            
        Returns:
            True si se reprogramó correctamente, False en caso contrario
        """
        # Cancelar tarea existente
        if not self.cancel_task(task_id):
            # Si no se pudo cancelar, intentar pausarla
            if not self.pause_task(task_id):
                logger.warning(f"No se pudo encontrar tarea {task_id} para reprogramar")
                return False
        
        # Obtener tarea original
        task = self.get_task_by_id(task_id)
        if not task:
            logger.error(f"Tarea {task_id} no encontrada después de cancelar/pausar")
            return False
        
        # Actualizar estado
        task['status'] = TaskStatus.PENDING.name
        
        # Actualizar prioridad si se proporciona
        if priority is not None:
            task['priority'] = priority.name if isinstance(priority, TaskPriority) else priority
        
        # Programar con nuevo tiempo
        self.schedule_task(
            task_type=task.get('type'),
            task_data=task.get('data', {}),
            execution_time=execution_time,
            priority=task.get('priority'),
            task_id=task_id
        )
        
        logger.info(f"Tarea {task_id} reprogramada para {execution_time.isoformat()}")
        return True
    
    def clear_completed_history(self, older_than_days: int = 30) -> int:
        """
        Limpia el historial de tareas completadas más antiguas que el número de días especificado
        
        Args:
            older_than_days: Número de días
            
        Returns:
            Número de tareas eliminadas
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        
        original_count = len(self.task_history)
        
        # Filtrar tareas más recientes que la fecha de corte
        self.task_history = [
            task for task in self.task_history
            if (
                # Mantener tareas sin fecha de finalización
                'end_time' not in task or
                # O tareas más recientes que la fecha de corte
                datetime.datetime.fromisoformat(task['end_time']) > cutoff_date
            )
        ]
        
        removed_count = original_count - len(self.task_history)
        
        if removed_count > 0:
            logger.info(f"Eliminadas {removed_count} tareas antiguas del historial")
        
        return removed_count
    
    def get_task_dependencies(self, task_id: str) -> Dict:
        """
        Obtiene información sobre las dependencias de una tarea
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Diccionario con información de dependencias
        """
        task = self.get_task_by_id(task_id)
        if not task:
            return {'error': 'Tarea no encontrada'}
        
        dependencies = task.get('dependencies', [])
        if not dependencies:
            return {'dependencies': [], 'all_resolved': True}
        
        # Verificar estado de cada dependencia
        dependency_status = []
        all_resolved = True
        
        for dep_id in dependencies:
            dep_task = self.get_task_by_id(dep_id)
            
            if not dep_task:
                status = 'NOT_FOUND'
                all_resolved = False
            elif dep_task.get('status') == TaskStatus.COMPLETED.name and dep_task.get('success', False):
                status = 'RESOLVED'
            else:
                status = dep_task.get('status', 'UNKNOWN')
                all_resolved = False
            
            dependency_status.append({
                'id': dep_id,
                'status': status
            })
        
        return {
            'dependencies': dependency_status,
            'all_resolved': all_resolved
        }
    
    def add_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """
        Añade una dependencia a una tarea existente
        
        Args:
            task_id: ID de la tarea
            dependency_id: ID de la tarea dependencia
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        task = self.get_task_by_id(task_id)
        if not task:
            logger.warning(f"No se encontró tarea {task_id}")
            return False
        
        # Verificar que la dependencia existe
        dep_task = self.get_task_by_id(dependency_id)
        if not dep_task:
            logger.warning(f"No se encontró tarea dependencia {dependency_id}")
            return False
        
        # Verificar que no se crea un ciclo de dependencias
        if self._would_create_dependency_cycle(task_id, dependency_id):
            logger.warning(f"No se puede añadir dependencia {dependency_id} a {task_id}: crearía un ciclo")
            return False
        
        # Añadir dependencia
        if 'dependencies' not in task:
            task['dependencies'] = []
        
        if dependency_id not in task['dependencies']:
            task['dependencies'].append(dependency_id)
            logger.info(f"Añadida dependencia {dependency_id} a tarea {task_id}")
            
            # Si la tarea está pendiente, verificar si debe pausarse
            if task.get('status') == TaskStatus.PENDING.name:
                if not self._check_dependencies(task):
                    # Pausar tarea hasta que se resuelvan las dependencias
                    self.pause_task(task_id)
            
            return True
        else:
            logger.info(f"La dependencia {dependency_id} ya existe en la tarea {task_id}")
            return False
    
    def _would_create_dependency_cycle(self, task_id: str, dependency_id: str, visited: Set[str] = None) -> bool:
        """
        Verifica si añadir una dependencia crearía un ciclo
        
        Args:
            task_id: ID de la tarea
            dependency_id: ID de la dependencia a añadir
            visited: Conjunto de IDs de tareas visitadas (para recursión)
            
        Returns:
            True si crearía un ciclo, False en caso contrario
        """
        # Si la dependencia es la misma tarea, es un ciclo
        if task_id == dependency_id:
            return True
        
        # Inicializar conjunto de visitados
        if visited is None:
            visited = set()
        
        # Marcar tarea actual como visitada
        visited.add(dependency_id)
        
        # Obtener dependencias de la dependencia
        dep_task = self.get_task_by_id(dependency_id)
        if not dep_task:
            return False
        
        # Verificar cada dependencia de la dependencia
        for dep_dep_id in dep_task.get('dependencies', []):
            # Si ya visitamos esta tarea, es un ciclo
            if dep_dep_id in visited:
                return True
            
            # Verificar recursivamente
            if dep_dep_id == task_id or self._would_create_dependency_cycle(task_id, dep_dep_id, visited.copy()):
                return True
        
        return False
    
    def check_dependencies_resolved(self) -> List[str]:
        """
        Verifica tareas pausadas por dependencias y las reanuda si las dependencias están resueltas
        
        Returns:
            Lista de IDs de tareas reanudadas
        """
        resumed_tasks = []
        
        # Copiar claves para evitar modificar durante la iteración
        paused_task_ids = list(self.paused_tasks.keys())
        
        for task_id in paused_task_ids:
            task = self.paused_tasks.get(task_id)
            if not task:
                continue
            
            # Verificar si la tarea está pausada por dependencias
            if task.get('status') == 'DEPENDENCY_PENDING':
                                # Verificar si las dependencias están resueltas
                if self._check_dependencies(task):
                    # Reanudar tarea
                    task['status'] = TaskStatus.PENDING.name
                    
                    # Obtener tiempo de ejecución
                    try:
                        execution_time = datetime.datetime.fromisoformat(task['scheduled_time'])
                    except (ValueError, KeyError):
                        execution_time = datetime.datetime.now()
                    
                    # Añadir a la cola con prioridad
                    priority = self._priority_to_int(task.get('priority', TaskPriority.MEDIUM.value))
                    
                    with self.queue_lock:
                        heapq.heappush(self.task_queue, (
                            execution_time,
                            priority,
                            task_id,
                            task
                        ))
                    
                    # Eliminar de tareas pausadas
                    del self.paused_tasks[task_id]
                    
                    logger.info(f"Tarea {task_id} reanudada automáticamente al resolverse sus dependencias")
                    resumed_tasks.append(task_id)
        
        return resumed_tasks
    
    def handle_failed_task(self, task_id: str, error: str, retry: bool = True, 
                          max_retries: int = 3, backoff_factor: int = 2) -> bool:
        """
        Gestiona una tarea fallida, opcionalmente programando un reintento
        
        Args:
            task_id: ID de la tarea
            error: Mensaje de error
            retry: Si se debe reintentar la tarea
            max_retries: Número máximo de reintentos
            backoff_factor: Factor de backoff exponencial para reintentos
            
        Returns:
            True si se gestionó correctamente, False en caso contrario
        """
        # Obtener tarea
        task = self.get_task_by_id(task_id)
        if not task:
            logger.error(f"No se encontró tarea {task_id} para gestionar fallo")
            return False
        
        # Incrementar contador de reintentos
        current_retries = task.get('retries', 0)
        task['retries'] = current_retries + 1
        
        # Registrar error
        if 'errors' not in task:
            task['errors'] = []
        
        task['errors'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'message': error,
            'retry_count': current_retries
        })
        
        # Verificar si se debe reintentar
        if retry and current_retries < max_retries:
            # Calcular tiempo de espera con backoff exponencial
            wait_seconds = backoff_factor ** current_retries
            
            # Añadir jitter aleatorio (±20%)
            jitter = random.uniform(0.8, 1.2)
            wait_seconds = int(wait_seconds * jitter)
            
            # Calcular tiempo de reintento
            retry_time = datetime.datetime.now() + datetime.timedelta(seconds=wait_seconds)
            
            # Actualizar estado
            task['status'] = TaskStatus.RETRYING.name
            task['retry_time'] = retry_time.isoformat()
            
            # Añadir a cola de reintentos
            with self.queue_lock:
                heapq.heappush(self.retry_queue, (
                    retry_time,
                    task_id,
                    task
                ))
            
            logger.info(f"Tarea {task_id} programada para reintento en {wait_seconds}s (intento {current_retries + 1}/{max_retries})")
            return True
        else:
            # Marcar como fallida definitivamente
            task['status'] = TaskStatus.FAILED.name
            
            # Registrar en historial
            self.task_history.append(task)
            
            # Actualizar métricas
            self.performance_metrics['tasks_failed'] += 1
            
            logger.warning(f"Tarea {task_id} marcada como fallida definitivamente después de {current_retries} reintentos")
            
            # Verificar si hay tareas dependientes que deben cancelarse
            self._handle_dependent_tasks(task_id)
            
            return False
    
    def _handle_dependent_tasks(self, failed_task_id: str) -> None:
        """
        Gestiona las tareas que dependen de una tarea fallida
        
        Args:
            failed_task_id: ID de la tarea fallida
        """
        # Buscar tareas dependientes en todas las colas
        dependent_tasks = []
        
        # Buscar en tareas pausadas
        for task_id, task in self.paused_tasks.items():
            if failed_task_id in task.get('dependencies', []):
                dependent_tasks.append((task_id, 'paused'))
        
        # Buscar en cola de tareas
        with self.queue_lock:
            for _, _, tid, task in self.task_queue:
                if failed_task_id in task.get('dependencies', []):
                    dependent_tasks.append((tid, 'queued'))
        
        # Buscar en cola de reintentos
        with self.queue_lock:
            for _, tid, task in self.retry_queue:
                if failed_task_id in task.get('dependencies', []):
                    dependent_tasks.append((tid, 'retry'))
        
        # Gestionar cada tarea dependiente
        for task_id, queue_type in dependent_tasks:
            # Obtener tarea
            task = self.get_task_by_id(task_id)
            if not task:
                continue
            
            # Actualizar estado
            task['status'] = TaskStatus.FAILED.name
            task['dependency_failure'] = {
                'dependency_id': failed_task_id,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Eliminar de la cola correspondiente
            if queue_type == 'paused':
                if task_id in self.paused_tasks:
                    del self.paused_tasks[task_id]
            elif queue_type == 'queued':
                self.cancel_task(task_id)
            elif queue_type == 'retry':
                # Eliminar de cola de reintentos
                with self.queue_lock:
                    self.retry_queue = [(t, tid, task) for t, tid, task in self.retry_queue if tid != task_id]
                    heapq.heapify(self.retry_queue)
            
            # Registrar en historial
            self.task_history.append(task)
            
            # Actualizar métricas
            self.performance_metrics['tasks_failed'] += 1
            
            logger.warning(f"Tarea {task_id} marcada como fallida debido a dependencia fallida {failed_task_id}")
            
            # Recursivamente gestionar tareas que dependen de esta
            self._handle_dependent_tasks(task_id)
    
    def process_retry_queue(self) -> int:
        """
        Procesa la cola de reintentos, moviendo tareas listas a la cola principal
        
        Returns:
            Número de tareas movidas
        """
        if not self.retry_queue:
            return 0
        
        now = datetime.datetime.now()
        moved_count = 0
        
        # Procesar cola de reintentos
        with self.queue_lock:
            new_retry_queue = []
            
            while self.retry_queue:
                retry_time, task_id, task = heapq.heappop(self.retry_queue)
                
                # Verificar si es hora de reintentar
                if retry_time <= now:
                    # Mover a cola principal
                    task['status'] = TaskStatus.PENDING.name
                    
                    # Añadir a cola principal
                    heapq.heappush(self.task_queue, (
                        now,  # Ejecutar inmediatamente
                        self._priority_to_int(task.get('priority', TaskPriority.MEDIUM.value)),
                        task_id,
                        task
                    ))
                    
                    moved_count += 1
                    logger.info(f"Tarea {task_id} movida de cola de reintentos a cola principal")
                else:
                    # Mantener en cola de reintentos
                    new_retry_queue.append((retry_time, task_id, task))
            
            # Restaurar cola de reintentos
            self.retry_queue = new_retry_queue
            heapq.heapify(self.retry_queue)
        
        return moved_count
    
    def get_retry_tasks(self) -> List[Dict]:
        """
        Obtiene las tareas en cola de reintentos
        
        Returns:
            Lista de tareas en cola de reintentos
        """
        retry_tasks = []
        
        with self.queue_lock:
            for retry_time, task_id, task in self.retry_queue:
                task_copy = task.copy()
                task_copy['retry_time'] = retry_time.isoformat()
                retry_tasks.append(task_copy)
        
        return retry_tasks
    
    def optimize_task_priorities(self) -> int:
        """
        Optimiza las prioridades de las tareas en cola según el contexto actual
        
        Returns:
            Número de tareas cuya prioridad fue ajustada
        """
        if not self.task_queue:
            return 0
        
        adjusted_count = 0
        
        # Procesar cola de tareas
        with self.queue_lock:
            new_queue = []
            
            while self.task_queue:
                execution_time, _, task_id, task = heapq.heappop(self.task_queue)
                
                # Obtener prioridad actual
                current_priority = task.get('priority')
                
                # Recalcular prioridad según contexto actual
                adjusted_priority = self._adjust_priority_by_context(
                    task.get('type'),
                    task.get('data', {}),
                    current_priority
                )
                
                # Verificar si cambió la prioridad
                if (isinstance(adjusted_priority, TaskPriority) and 
                    (not isinstance(current_priority, TaskPriority) or 
                     adjusted_priority != current_priority)):
                    # Actualizar prioridad
                    task['priority'] = adjusted_priority.name
                    adjusted_count += 1
                    logger.info(f"Prioridad de tarea {task_id} ajustada de {current_priority} a {adjusted_priority.name}")
                
                # Añadir a nueva cola
                new_queue.append((
                    execution_time,
                    self._priority_to_int(adjusted_priority),
                    task_id,
                    task
                ))
            
            # Restaurar cola
            self.task_queue = new_queue
            heapq.heapify(self.task_queue)
        
        return adjusted_count
    
    def get_task_stats_by_type(self) -> Dict[str, Dict[str, int]]:
        """
        Obtiene estadísticas de tareas agrupadas por tipo
        
        Returns:
            Diccionario con estadísticas por tipo de tarea
        """
        stats = {}
        
        # Función auxiliar para actualizar estadísticas
        def update_stats(task_type, status):
            if task_type not in stats:
                stats[task_type] = {
                    'pending': 0,
                    'running': 0,
                    'completed': 0,
                    'failed': 0,
                    'retrying': 0,
                    'paused': 0,
                    'cancelled': 0,
                    'total': 0
                }
            
            if status in stats[task_type]:
                stats[task_type][status.lower()] += 1
            stats[task_type]['total'] += 1
        
        # Procesar cola de tareas
        with self.queue_lock:
            for _, _, _, task in self.task_queue:
                update_stats(task.get('type', 'unknown'), 'pending')
        
        # Procesar tareas en ejecución
        for task_id, task_info in self.running_tasks.items():
            task = task_info.get('task', {})
            update_stats(task.get('type', 'unknown'), 'running')
        
        # Procesar tareas pausadas
        for task_id, task in self.paused_tasks.items():
            update_stats(task.get('type', 'unknown'), 'paused')
        
        # Procesar cola de reintentos
        with self.queue_lock:
            for _, _, task in self.retry_queue:
                update_stats(task.get('type', 'unknown'), 'retrying')
        
        # Procesar historial (últimas 1000 tareas)
        for task in self.task_history:
            status = task.get('status', 'unknown').lower()
            if status == 'completed' and task.get('success', False):
                update_stats(task.get('type', 'unknown'), 'completed')
            elif status == 'failed' or (status == 'completed' and not task.get('success', True)):
                update_stats(task.get('type', 'unknown'), 'failed')
            elif status == 'cancelled':
                update_stats(task.get('type', 'unknown'), 'cancelled')
        
        return stats
    
    def get_task_execution_times(self, task_type: str = None, limit: int = 100) -> Dict[str, List[float]]:
        """
        Obtiene tiempos de ejecución de tareas completadas
        
        Args:
            task_type: Tipo de tarea para filtrar (opcional)
            limit: Número máximo de tareas a considerar
            
        Returns:
            Diccionario con tiempos de ejecución por tipo de tarea
        """
        execution_times = {}
        
        # Filtrar historial
        filtered_history = [
            task for task in self.task_history
            if (task_type is None or task.get('type') == task_type) and
               task.get('success', False) and
               'duration' in task
        ]
        
        # Limitar número de tareas
        filtered_history = filtered_history[-limit:]
        
        # Agrupar por tipo
        for task in filtered_history:
            task_type = task.get('type', 'unknown')
            
            if task_type not in execution_times:
                execution_times[task_type] = []
            
            execution_times[task_type].append(task.get('duration', 0))
        
        return execution_times
    
    def get_task_success_rates(self, task_type: str = None, limit: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Obtiene tasas de éxito de tareas
        
        Args:
            task_type: Tipo de tarea para filtrar (opcional)
            limit: Número máximo de tareas a considerar
            
        Returns:
            Diccionario con tasas de éxito por tipo de tarea
        """
        success_rates = {}
        
        # Filtrar historial
        filtered_history = [
            task for task in self.task_history
            if (task_type is None or task.get('type') == task_type) and
               task.get('status') in [TaskStatus.COMPLETED.name, TaskStatus.FAILED.name]
        ]
        
        # Limitar número de tareas
        filtered_history = filtered_history[-limit:]
        
        # Agrupar por tipo
        for task in filtered_history:
            task_type = task.get('type', 'unknown')
            
            if task_type not in success_rates:
                success_rates[task_type] = {
                    'success': 0,
                    'failure': 0,
                    'total': 0,
                    'rate': 0.0
                }
            
            if task.get('success', False):
                success_rates[task_type]['success'] += 1
            else:
                success_rates[task_type]['failure'] += 1
            
            success_rates[task_type]['total'] += 1
        
        # Calcular tasas
        for task_type, stats in success_rates.items():
            if stats['total'] > 0:
                stats['rate'] = stats['success'] / stats['total']
        
        return success_rates
    
    def get_task_priority_distribution(self) -> Dict[str, int]:
        """
        Obtiene la distribución de prioridades de tareas en cola
        
        Returns:
            Diccionario con conteo de tareas por prioridad
        """
        distribution = {
            TaskPriority.CRITICAL.name: 0,
            TaskPriority.HIGH.name: 0,
            TaskPriority.MEDIUM.name: 0,
            TaskPriority.LOW.name: 0,
            TaskPriority.BACKGROUND.name: 0,
            'unknown': 0
        }
        
        # Procesar cola de tareas
        with self.queue_lock:
            for _, _, _, task in self.task_queue:
                priority = task.get('priority')
                
                if isinstance(priority, TaskPriority):
                    priority = priority.name
                
                if priority in distribution:
                    distribution[priority] += 1
                else:
                    distribution['unknown'] += 1
        
        return distribution
    
    def get_task_execution_forecast(self, lookahead_hours: int = 24) -> Dict[str, List[Dict]]:
        """
        Genera un pronóstico de ejecución de tareas para las próximas horas
        
        Args:
            lookahead_hours: Número de horas a pronosticar
            
        Returns:
            Diccionario con pronóstico por hora
        """
        forecast = {}
        
        # Definir ventanas de tiempo
        now = datetime.datetime.now()
        end_time = now + datetime.timedelta(hours=lookahead_hours)
        
        # Crear buckets por hora
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        while current_hour < end_time:
            hour_key = current_hour.isoformat()
            forecast[hour_key] = []
            current_hour += datetime.timedelta(hours=1)
        
        # Procesar cola de tareas
        with self.queue_lock:
            for execution_time, _, _, task in self.task_queue:
                # Verificar si está dentro del período de pronóstico
                if execution_time < end_time:
                    # Encontrar bucket correspondiente
                    bucket_time = execution_time.replace(minute=0, second=0, microsecond=0)
                    bucket_key = bucket_time.isoformat()
                    
                    # Si el bucket no existe (tarea programada para antes de la hora actual),
                    # usar el bucket actual
                    if bucket_key not in forecast:
                        bucket_key = now.replace(minute=0, second=0, microsecond=0).isoformat()
                    
                    # Añadir tarea al pronóstico
                    forecast[bucket_key].append({
                        'id': task.get('id'),
                        'type': task.get('type'),
                        'priority': task.get('priority'),
                        'execution_time': execution_time.isoformat()
                    })
        
        return forecast
    
    def get_system_load_metrics(self) -> Dict[str, float]:
        """
        Obtiene métricas de carga del sistema
        
        Returns:
            Diccionario con métricas de carga
        """
        metrics = {}
        
        # Calcular tareas por minuto (últimos 60 minutos)
        now = datetime.datetime.now()
        one_hour_ago = now - datetime.timedelta(minutes=60)
        
        # Filtrar tareas completadas en la última hora
        completed_last_hour = [
            task for task in self.task_history
            if 'end_time' in task and
               datetime.datetime.fromisoformat(task['end_time']) > one_hour_ago
        ]
        
        # Calcular tasa de tareas por minuto
        metrics['tasks_per_minute'] = len(completed_last_hour) / 60
        
        # Calcular tiempo promedio de ejecución (últimos 60 minutos)
        if completed_last_hour:
            avg_duration = sum(task.get('duration', 0) for task in completed_last_hour) / len(completed_last_hour)
            metrics['avg_execution_time'] = avg_duration
        else:
            metrics['avg_execution_time'] = 0
        
        # Calcular tasa de éxito (últimos 60 minutos)
        if completed_last_hour:
            success_count = sum(1 for task in completed_last_hour if task.get('success', False))
            metrics['success_rate_1h'] = success_count / len(completed_last_hour)
        else:
            metrics['success_rate_1h'] = 0
        
        # Calcular carga actual
        metrics['queue_size'] = len(self.task_queue)
        metrics['running_tasks'] = len(self.running_tasks)
        metrics['retry_queue_size'] = len(self.retry_queue)
        
        # Estimar tiempo de espera promedio
        if metrics['queue_size'] > 0 and metrics['tasks_per_minute'] > 0:
            metrics['estimated_wait_time'] = metrics['queue_size'] / metrics['tasks_per_minute']
        else:
            metrics['estimated_wait_time'] = 0
        
        return metrics
    
    def _save_persistent_state(self) -> bool:
        """
        Guarda el estado del planificador en disco
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Crear directorio si no existe
            os.makedirs('data/scheduler', exist_ok=True)
            
            # Preparar estado para serialización
            state = {
                'performance_metrics': self.performance_metrics,
                'task_history': self.task_history[-100:],  # Guardar solo las últimas 100 tareas
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Guardar estado
            with open('data/scheduler/state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Estado del planificador guardado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al guardar estado del planificador: {str(e)}")
            return False
    
    def _load_persistent_state(self) -> bool:
        """
        Carga el estado del planificador desde disco
        
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        try:
            # Verificar si existe archivo de estado
            if not os.path.exists('data/scheduler/state.json'):
                logger.info("No se encontró archivo de estado del planificador")
                return False
            
            # Cargar estado
            with open('data/scheduler/state.json', 'r') as f:
                state = json.load(f)
            
            # Restaurar estado
            self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
            self.task_history = state.get('task_history', [])
            
            logger.info("Estado del planificador cargado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al cargar estado del planificador: {str(e)}")
            return False
    
    def shutdown(self) -> None:
        """
        Realiza tareas de limpieza antes de apagar el planificador
        """
        logger.info("Apagando planificador de tareas...")
        
        # Guardar estado
        self._save_persistent_state()
        
        # Cancelar tareas en ejecución
        for task_id in list(self.running_tasks.keys()):
            logger.warning(f"Tarea en ejecución {task_id} interrumpida por apagado")
            self.mark_task_completed(task_id, False, "Interrumpida por apagado del sistema")
        
        logger.info("Planificador de tareas apagado correctamente")
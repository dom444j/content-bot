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
from typing import Dict, List, Any, Optional, Tuple, Callable
import schedule

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
        
        # Historial de tareas
        self.task_history = []
        
        # Configuración de horarios
        self.content_creation_hours = self.strategy.get('global_settings', {}).get('content_creation_hours', [0, 23])
        self.publishing_hours = self.strategy.get('global_settings', {}).get('publishing_hours', [17, 22])
        self.engagement_hours = self.strategy.get('global_settings', {}).get('engagement_hours', [8, 23])
        self.analysis_hours = self.strategy.get('global_settings', {}).get('analysis_hours', [0, 5])
        
        # Iniciar thread de planificación
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        self._initialized = True
        logger.info("Scheduler inicializado correctamente")
    
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
            return {}
    
    def start(self):
        """Inicia el planificador en un thread separado"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("El planificador ya está en ejecución")
            return
        
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Configurar tareas recurrentes
        self._setup_recurring_tasks()
        
        logger.info("Planificador iniciado")
    
    def stop(self):
        """Detiene el planificador"""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("El planificador no está en ejecución")
            return
        
        self.stop_event.set()
        self.scheduler_thread.join(timeout=5)
        
        # Limpiar tareas recurrentes
        schedule.clear()
        
        logger.info("Planificador detenido")
    
    def _scheduler_loop(self):
        """Loop principal del planificador"""
        while not self.stop_event.is_set():
            # Ejecutar tareas programadas
            self._process_task_queue()
            
            # Ejecutar tareas recurrentes
            schedule.run_pending()
            
            # Dormir brevemente
            time.sleep(1)
    
    def _setup_recurring_tasks(self):
        """Configura tareas recurrentes basadas en la estrategia"""
        # Tarea diaria de análisis (a las 3 AM)
        schedule.every().day.at("03:00").do(self._create_task, 
            task_type="analysis", 
            task_data={"type": "daily_analysis"},
            priority=1
        )
        
        # Tarea diaria de detección de tendencias (cada 6 horas)
        for hour in [6, 12, 18, 0]:
            schedule.every().day.at(f"{hour:02d}:00").do(self._create_task,
                task_type="trends",
                task_data={"type": "trend_detection"},
                priority=2
            )
        
        # Tarea de engagement (cada 2 horas durante horas de engagement)
        for hour in range(self.engagement_hours[0], self.engagement_hours[1] + 1, 2):
            schedule.every().day.at(f"{hour:02d}:00").do(self._create_task,
                task_type="engagement",
                task_data={"type": "comment_response"},
                priority=3
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
                            priority=2
                        )
                    else:
                        # Cada X días
                        schedule.every(interval).days.at(f"{creation_hour:02d}:00").do(self._create_task,
                            task_type="creation",
                            task_data={"niche": niche, "platform": platform},
                            priority=2
                        )
        
        logger.info(f"Configuradas {len(schedule.jobs)} tareas recurrentes")
    
    def _process_task_queue(self):
        """Procesa la cola de tareas programadas"""
        now = datetime.datetime.now()
        
        # Procesar tareas cuyo tiempo de ejecución ha llegado
        while self.task_queue and self.task_queue[0][0] <= now:
            # Extraer tarea de la cola
            _, task_id, task = heapq.heappop(self.task_queue)
            
            # Verificar si la tarea ya está en ejecución
            if task_id in self.running_tasks:
                logger.warning(f"Tarea {task_id} ya en ejecución, omitiendo")
                continue
            
            # Ejecutar tarea en un thread separado
            self._execute_task(task_id, task)
    
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
            # Aquí se implementaría la lógica para ejecutar diferentes tipos de tareas
            # Por ahora, solo registramos la ejecución
            
            task_type = task['type']
            task_data = task['data']
            
            logger.info(f"Ejecutando tarea {task_id} de tipo {task_type}: {task_data}")
            
            # Simular tiempo de ejecución
            time.sleep(2)
            
            # Registrar finalización exitosa
            self._task_completed(task_id, task, success=True)
            
        except Exception as e:
            logger.error(f"Error al ejecutar tarea {task_id}: {str(e)}")
            # Registrar finalización con error
            self._task_completed(task_id, task, success=False, error=str(e))
    
    def _task_completed(self, task_id: str, task: Dict, success: bool, error: str = None):
        """Registra la finalización de una tarea"""
        end_time = datetime.datetime.now()
        
        # Obtener información de inicio
        start_time = self.running_tasks[task_id]['start_time'] if task_id in self.running_tasks else end_time
        
        # Registrar en historial
        task_record = {
            'id': task_id,
            'type': task['type'],
            'data': task['data'],
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': (end_time - start_time).total_seconds(),
            'success': success
        }
        
        if error:
            task_record['error'] = error
        
        self.task_history.append(task_record)
        
        # Limitar tamaño del historial
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
        
        # Eliminar de tareas en ejecución
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        logger.info(f"Tarea {task_id} completada. Éxito: {success}")
    
        def schedule_task(self, task_type: str, task_data: Dict, execution_time: datetime.datetime = None, 
                     priority: int = 5, task_id: str = None) -> str:
        """
        Programa una tarea para ejecución futura
        
        Args:
            task_type: Tipo de tarea (creation, publishing, analysis, etc.)
            task_data: Datos específicos de la tarea
            execution_time: Tiempo de ejecución (si es None, se ejecuta inmediatamente)
            priority: Prioridad (1-10, siendo 1 la más alta)
            task_id: ID de tarea (si es None, se genera automáticamente)
            
        Returns:
            ID de la tarea programada
        """
        # Generar ID de tarea si no se proporciona
        if task_id is None:
            task_id = f"{task_type}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Si no se especifica tiempo de ejecución, usar tiempo actual
        if execution_time is None:
            execution_time = datetime.datetime.now()
        
        # Crear objeto de tarea
        task = {
            'id': task_id,
            'type': task_type,
            'data': task_data,
            'priority': priority,
            'scheduled_time': execution_time.isoformat(),
            'created_at': datetime.datetime.now().isoformat()
        }
        
        # Añadir a la cola de prioridad
        heapq.heappush(self.task_queue, (execution_time, task_id, task))
        
        logger.info(f"Tarea {task_id} programada para {execution_time.isoformat()}")
        
        return task_id
    
    def _create_task(self, task_type: str, task_data: Dict, priority: int = 5) -> str:
        """
        Crea una tarea para ejecución inmediata (usado por tareas recurrentes)
        
        Args:
            task_type: Tipo de tarea
            task_data: Datos de la tarea
            priority: Prioridad de la tarea
            
        Returns:
            ID de la tarea creada
        """
        return self.schedule_task(task_type, task_data, datetime.datetime.now(), priority)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea programada
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            True si se canceló correctamente, False si no se encontró
        """
        # Buscar la tarea en la cola
        for i, (time, tid, task) in enumerate(self.task_queue):
            if tid == task_id:
                # Eliminar de la cola
                self.task_queue[i] = self.task_queue[-1]
                self.task_queue.pop()
                
                if i < len(self.task_queue):
                    heapq.heapify(self.task_queue)
                
                logger.info(f"Tarea {task_id} cancelada")
                return True
        
        # Verificar si está en ejecución
        if task_id in self.running_tasks:
            logger.warning(f"No se puede cancelar tarea {task_id}, ya está en ejecución")
            return False
        
        logger.warning(f"Tarea {task_id} no encontrada para cancelar")
        return False
    
    def reschedule_task(self, task_id: str, new_execution_time: datetime.datetime) -> bool:
        """
        Reprograma una tarea existente
        
        Args:
            task_id: ID de la tarea a reprogramar
            new_execution_time: Nuevo tiempo de ejecución
            
        Returns:
            True si se reprogramó correctamente, False si no se encontró
        """
        # Buscar y eliminar la tarea actual
        task_found = False
        task_data = None
        
        for i, (time, tid, task) in enumerate(self.task_queue):
            if tid == task_id:
                task_found = True
                task_data = task
                
                # Eliminar de la cola
                self.task_queue[i] = self.task_queue[-1]
                self.task_queue.pop()
                
                if i < len(self.task_queue):
                    heapq.heapify(self.task_queue)
                
                break
        
        if not task_found:
            logger.warning(f"Tarea {task_id} no encontrada para reprogramar")
            return False
        
        # Reprogramar con nuevo tiempo
        heapq.heappush(self.task_queue, (new_execution_time, task_id, task_data))
        
        logger.info(f"Tarea {task_id} reprogramada para {new_execution_time.isoformat()}")
        return True
    
    def get_pending_tasks(self, task_type: str = None) -> List[Dict]:
        """
        Obtiene lista de tareas pendientes
        
        Args:
            task_type: Filtrar por tipo de tarea (opcional)
            
        Returns:
            Lista de tareas pendientes
        """
        pending_tasks = []
        
        for time, tid, task in self.task_queue:
            if task_type is None or task['type'] == task_type:
                task_info = task.copy()
                task_info['execution_time'] = time.isoformat()
                pending_tasks.append(task_info)
        
        return pending_tasks
    
    def get_running_tasks(self) -> List[Dict]:
        """
        Obtiene lista de tareas en ejecución
        
        Returns:
            Lista de tareas en ejecución
        """
        running_tasks = []
        
        for task_id, task_info in self.running_tasks.items():
            task_data = task_info['task'].copy()
            task_data['start_time'] = task_info['start_time'].isoformat()
            task_data['running_time'] = (datetime.datetime.now() - task_info['start_time']).total_seconds()
            running_tasks.append(task_data)
        
        return running_tasks
    
    def get_task_history(self, limit: int = 100, task_type: str = None) -> List[Dict]:
        """
        Obtiene historial de tareas completadas
        
        Args:
            limit: Número máximo de tareas a devolver
            task_type: Filtrar por tipo de tarea (opcional)
            
        Returns:
            Lista de tareas completadas
        """
        if task_type is None:
            return self.task_history[-limit:]
        else:
            return [task for task in self.task_history if task['type'] == task_type][-limit:]
    
    def schedule_optimal_publishing_time(self, niche: str, platform: str, content_id: str) -> str:
        """
        Programa la publicación de contenido en un horario óptimo
        
        Args:
            niche: Nicho del contenido
            platform: Plataforma de publicación
            content_id: ID del contenido a publicar
            
        Returns:
            ID de la tarea de publicación
        """
        # Determinar horario óptimo basado en datos históricos y configuración
        now = datetime.datetime.now()
        
        # Obtener rango de horas de publicación
        start_hour, end_hour = self.publishing_hours
        
        # Si estamos fuera del rango de horas, programar para el inicio del próximo rango
        if now.hour < start_hour:
            # Programar para hoy a la hora de inicio
            optimal_time = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        elif now.hour >= end_hour:
            # Programar para mañana a la hora de inicio
            optimal_time = now.replace(hour=start_hour, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        else:
            # Estamos dentro del rango, buscar el mejor momento
            # Aquí se podría implementar lógica más avanzada basada en datos históricos
            
            # Por ahora, simplemente elegimos un momento aleatorio dentro del rango
            random_hour = random.randint(start_hour, end_hour)
            random_minute = random.randint(0, 59)
            
            optimal_time = now.replace(hour=random_hour, minute=random_minute, second=0, microsecond=0)
            
            # Si el tiempo ya pasó, programar para mañana
            if optimal_time < now:
                optimal_time += datetime.timedelta(days=1)
        
        # Programar tarea de publicación
        task_data = {
            'content_id': content_id,
            'niche': niche,
            'platform': platform
        }
        
        return self.schedule_task(
            task_type="publishing",
            task_data=task_data,
            execution_time=optimal_time,
            priority=1  # Alta prioridad para publicaciones
        )
"""
Ejecutor de tareas recurrentes basado en expresiones cron.

Este módulo implementa un ejecutor que programa tareas recurrentes
utilizando expresiones cron, ideal para tareas periódicas como
análisis, mantenimiento, o publicaciones programadas.
"""

import logging
import threading
import time
import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPoolExecutor
from pytz import timezone

from ..core.task_model import Task, TaskStatus
from .base_executor import BaseExecutor

logger = logging.getLogger('Scheduler.Executor.Cron')

class CronExecutor(BaseExecutor):
    """
    Ejecutor que programa tareas recurrentes utilizando expresiones cron.
    
    Ideal para:
    - Tareas periódicas (diarias, semanales, mensuales)
    - Publicaciones programadas en horarios específicos
    - Análisis y reportes recurrentes
    - Mantenimiento y limpieza programada
    
    Ventajas:
    - Programación precisa basada en tiempo
    - Soporte para expresiones cron estándar
    - Persistencia de programaciones
    - Manejo de zonas horarias
    
    Limitaciones:
    - No adecuado para tareas de alta frecuencia (< 1 minuto)
    - Overhead para tareas simples
    """
    
    def __init__(self, task_queue, config=None):
        """
        Inicializa el ejecutor de tareas recurrentes.
        
        Args:
            task_queue: Cola de tareas compartida
            config: Configuración específica del ejecutor
        """
        super().__init__(task_queue, config)
        
        # Configuración
        self.timezone = self.config.get('timezone', 'UTC')
        self.max_workers = self.config.get('max_workers', 10)
        self.misfire_grace_time = self.config.get('misfire_grace_time', 60)  # segundos
        
        # Inicializar scheduler de APScheduler
        self.scheduler = None
        
        # Mapeo de tareas recurrentes
        self.recurring_tasks = {}  # task_id -> job_id
        self.recurring_tasks_lock = threading.RLock()
        
        logger.info(f"CronExecutor inicializado con timezone {self.timezone}")
    
    def start(self):
        """
        Inicia el ejecutor de tareas recurrentes.
        """
        if self.running:
            logger.warning("CronExecutor ya está en ejecución")
            return
        
        # Configurar APScheduler
        jobstores = {
            'default': MemoryJobStore()
        }
        
        executors = {
            'default': APSThreadPoolExecutor(max_workers=self.max_workers)
        }
        
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': self.misfire_grace_time
        }
        
        # Inicializar scheduler
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone(self.timezone)
        )
        
        # Iniciar scheduler
        self.scheduler.start()
        
        # Iniciar el ejecutor base
        super().start()
        
        logger.info(f"CronExecutor iniciado con {self.max_workers} workers")
    
    def shutdown(self, wait=True):
        """
        Detiene el ejecutor de tareas recurrentes.
        
        Args:
            wait: Si es True, espera a que todas las tareas terminen
        """
        if not self.running:
            return
        
        logger.info("Deteniendo CronExecutor...")
        
        # Detener el ejecutor base
        super().shutdown(wait=False)  # No esperar aquí, lo haremos después
        
        # Detener APScheduler
        if self.scheduler:
            self.scheduler.shutdown(wait=wait)
            self.scheduler = None
        
        # Limpiar tareas recurrentes
        with self.recurring_tasks_lock:
            self.recurring_tasks.clear()
        
        logger.info("CronExecutor detenido correctamente")
    
    def _get_executor_type(self) -> str:
        """
        Obtiene el tipo de ejecutor.
        
        Returns:
            str: 'cron'
        """
        return 'cron'
    
    def _execute_task(self, task: Task) -> None:
        """
        Programa una tarea recurrente.
        
        Args:
            task: Tarea a programar
        """
        logger.debug(f"Programando tarea recurrente {task.task_id} ({task.task_type})")
        
        # Verificar que la tarea tenga configuración de programación
        schedule_config = task.data.get('schedule')
        
        if not schedule_config:
            error_msg = f"La tarea {task.task_id} no tiene configuración de programación"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Buscar handler para el tipo de tarea
        handler = self.task_handlers.get(task.task_type)
        
        if handler is None:
            error_msg = f"No hay handler registrado para el tipo de tarea: {task.task_type}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Crear trigger según el tipo de programación
        trigger = self._create_trigger(schedule_config)
        
        if trigger is None:
            error_msg = f"Configuración de programación inválida para tarea {task.task_id}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Crear función de callback para la tarea
        def task_callback():
            # Crear copia de la tarea para cada ejecución
            execution_task = Task(
                task_id=f"{task.task_id}_{int(time.time())}",
                task_type=task.task_type,
                data=task.data.get('task_data', {}),
                priority=task.priority,
                status=TaskStatus.CREATED
            )
            
            logger.info(f"Ejecutando tarea recurrente {task.task_id} (ejecución {execution_task.task_id})")
            
            try:
                # Ejecutar handler directamente
                result = handler(execution_task)
                
                # Actualizar estado de la ejecución
                execution_task.status = TaskStatus.COMPLETED
                execution_task.completed_at = time.time()
                execution_task.result = result
                
                # Registrar ejecución exitosa
                self._record_execution(task.task_id, execution_task, success=True)
                
                logger.info(f"Tarea recurrente {task.task_id} ejecutada exitosamente")
                
            except Exception as e:
                # Actualizar estado de la ejecución
                execution_task.status = TaskStatus.FAILED
                execution_task.error = str(e)
                
                # Registrar ejecución fallida
                self._record_execution(task.task_id, execution_task, success=False, error=str(e))
                
                logger.error(f"Error en ejecución de tarea recurrente {task.task_id}: {str(e)}", exc_info=True)
        
        # Programar tarea en APScheduler
        job = self.scheduler.add_job(
            task_callback,
            trigger=trigger,
            id=f"job_{task.task_id}",
            name=f"Task {task.task_id} ({task.task_type})",
            replace_existing=True
        )
        
        # Registrar tarea recurrente
        with self.recurring_tasks_lock:
            self.recurring_tasks[task.task_id] = job.id
        
        # Actualizar estado de la tarea
        task.status = TaskStatus.SCHEDULED
        
        with self.active_tasks_lock:
            self.active_tasks[task.task_id] = {
                'task': task,
                'status': TaskStatus.SCHEDULED,
                'scheduled_at': time.time(),
                'next_run': job.next_run_time.timestamp() if job.next_run_time else None
            }
        
        self._update_task_status(task.task_id, TaskStatus.SCHEDULED)
        
        logger.info(f"Tarea recurrente {task.task_id} programada correctamente (próxima ejecución: {job.next_run_time})")
    
    def _create_trigger(self, schedule_config):
        """
        Crea un trigger de APScheduler según la configuración.
        
        Args:
            schedule_config: Configuración de programación
            
        Returns:
            Trigger de APScheduler o None si la configuración es inválida
        """
        schedule_type = schedule_config.get('type')
        
        if schedule_type == 'cron':
            # Expresión cron
            try:
                return CronTrigger(
                    year=schedule_config.get('year', '*'),
                    month=schedule_config.get('month', '*'),
                    day=schedule_config.get('day', '*'),
                    week=schedule_config.get('week', '*'),
                    day_of_week=schedule_config.get('day_of_week', '*'),
                    hour=schedule_config.get('hour', '*'),
                    minute=schedule_config.get('minute', '*'),
                    second=schedule_config.get('second', '0'),
                    timezone=timezone(schedule_config.get('timezone', self.timezone))
                )
            except Exception as e:
                logger.error(f"Error al crear trigger cron: {str(e)}")
                return None
        
        elif schedule_type == 'interval':
            # Intervalo
            try:
                return IntervalTrigger(
                    weeks=schedule_config.get('weeks', 0),
                    days=schedule_config.get('days', 0),
                    hours=schedule_config.get('hours', 0),
                    minutes=schedule_config.get('minutes', 0),
                    seconds=schedule_config.get('seconds', 0),
                    start_date=schedule_config.get('start_date'),
                    end_date=schedule_config.get('end_date'),
                    timezone=timezone(schedule_config.get('timezone', self.timezone))
                )
            except Exception as e:
                logger.error(f"Error al crear trigger de intervalo: {str(e)}")
                return None
        
        elif schedule_type == 'date':
            # Fecha específica
            try:
                return DateTrigger(
                    run_date=schedule_config.get('run_date'),
                    timezone=timezone(schedule_config.get('timezone', self.timezone))
                )
            except Exception as e:
                logger.error(f"Error al crear trigger de fecha: {str(e)}")
                return None
        
        else:
            # Tipo de programación desconocido
            logger.error(f"Tipo de programación desconocido: {schedule_type}")
            return None
    
    def _record_execution(self, task_id: str, execution_task: Task, success: bool, error: str = None) -> None:
        """
        Registra la ejecución de una tarea recurrente.
        
        Args:
            task_id: ID de la tarea recurrente
            execution_task: Tarea de ejecución específica
            success: Si la ejecución fue exitosa
            error: Mensaje de error (si hubo)
        """
        # Obtener historial de ejecuciones
        with self.active_tasks_lock:
            if task_id not in self.active_tasks:
                logger.warning(f"No se encontró tarea recurrente {task_id} para registrar ejecución")
                return
            
            task_info = self.active_tasks[task_id]
            
            # Inicializar historial si no existe
            if 'executions' not in task_info:
                task_info['executions'] = []
            
            # Registrar ejecución
            execution_record = {
                'execution_id': execution_task.task_id,
                'timestamp': time.time(),
                'success': success,
                'duration': (execution_task.completed_at or time.time()) - execution_task.started_at if execution_task.started_at else 0,
                'error': error
            }
            
            # Añadir al historial (limitando a las últimas 100 ejecuciones)
            task_info['executions'].append(execution_record)
            if len(task_info['executions']) > 100:
                task_info['executions'] = task_info['executions'][-100:]
            
            # Actualizar estadísticas
            if 'stats' not in task_info:
                task_info['stats'] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'last_execution': None,
                    'average_duration': 0
                }
            
            stats = task_info['stats']
            stats['total_executions'] += 1
            
            if success:
                stats['successful_executions'] += 1
            else:
                stats['failed_executions'] += 1
            
            stats['last_execution'] = time.time()
            
            # Actualizar duración promedio
            if 'duration' in execution_record and execution_record['duration'] > 0:
                if stats['average_duration'] == 0:
                    stats['average_duration'] = execution_record['duration']
                else:
                    # Media móvil ponderada (90% histórico, 10% actual)
                    stats['average_duration'] = (stats['average_duration'] * 0.9) + (execution_record['duration'] * 0.1)
    
    def _cancel_task_impl(self, task_id: str) -> bool:
        """
        Implementación específica de cancelación de tarea.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        with self.recurring_tasks_lock:
            if task_id not in self.recurring_tasks:
                logger.warning(f"No se encontró tarea recurrente {task_id} para cancelar")
                return False
            
            job_id = self.recurring_tasks[task_id]
            
            try:
                # Eliminar job de APScheduler
                self.scheduler.remove_job(job_id)
                
                # Eliminar de tareas recurrentes
                del self.recurring_tasks[task_id]
                
                # Actualizar estado
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                
                logger.info(f"Tarea recurrente {task_id} cancelada correctamente")
                return True
            
            except Exception as e:
                logger.error(f"Error al cancelar tarea recurrente {task_id}: {str(e)}")
                return False
    
    def get_active_tasks(self) -> List[str]:
        """
        Obtiene las tareas actualmente programadas.
        
        Returns:
            List[str]: Lista de IDs de tareas activas
        """
        with self.active_tasks_lock:
            return list(self.active_tasks.keys())
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene detalles de una tarea recurrente.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Optional[Dict]: Detalles de la tarea o None si no existe
        """
        with self.active_tasks_lock, self.recurring_tasks_lock:
            if task_id not in self.active_tasks:
                return None
            
            task_info = self.active_tasks[task_id]
            
            # Obtener información del job
            job_id = self.recurring_tasks.get(task_id)
            job = None
            
            if job_id:
                job = self.scheduler.get_job(job_id)
            
            # Construir respuesta
            result = {
                'task_id': task_id,
                'status': task_info.get('status', TaskStatus.UNKNOWN).name,
                'scheduled_at': task_info.get('scheduled_at'),
                'next_run': job.next_run_time.timestamp() if job and job.next_run_time else None,
                'task_type': task_info['task'].task_type if 'task' in task_info else None,
                'stats': task_info.get('stats', {})
            }
            
            return result
    
    def get_upcoming_executions(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Obtiene las próximas ejecuciones programadas.
        
        Args:
            hours_ahead: Horas hacia adelante para buscar
            
        Returns:
            List[Dict]: Lista de próximas ejecuciones
        """
        if not self.scheduler:
            return []
        
        # Obtener todos los jobs
        jobs = self.scheduler.get_jobs()
        
        # Filtrar por tiempo
        now = datetime.datetime.now(self.scheduler.timezone)
        end_time = now + datetime.timedelta(hours=hours_ahead)
        
        upcoming = []
        
        for job in jobs:
            # Verificar si tiene próxima ejecución
            if not job.next_run_time:
                continue
            
            # Verificar si está dentro del rango
            if job.next_run_time > end_time:
                continue
            
            # Extraer task_id del job_id
            task_id = job.id.replace('job_', '') if job.id.startswith('job_') else job.id
            
            # Obtener detalles de la tarea
            task_info = None
            with self.active_tasks_lock:
                if task_id in self.active_tasks:
                    task_info = self.active_tasks[task_id]
            
            # Añadir a la lista
            upcoming.append({
                'task_id': task_id,
                'next_run_time': job.next_run_time.timestamp(),
                'task_type': task_info['task'].task_type if task_info and 'task' in task_info else None,
                'priority': task_info['task'].priority.name if task_info and 'task' in task_info else None
            })
        
        # Ordenar por tiempo de ejecución
        upcoming.sort(key=lambda x: x['next_run_time'])
        
        return upcoming
    
    def pause_task(self, task_id: str) -> bool:
        """
        Pausa una tarea recurrente temporalmente.
        
        Args:
            task_id: ID de la tarea a pausar
            
        Returns:
            bool: True si se pausó correctamente, False en caso contrario
        """
        with self.recurring_tasks_lock:
            if task_id not in self.recurring_tasks:
                logger.warning(f"No se encontró tarea recurrente {task_id} para pausar")
                return False
            
            job_id = self.recurring_tasks[task_id]
            
            try:
                # Pausar job en APScheduler
                self.scheduler.pause_job(job_id)
                
                # Actualizar estado
                self._update_task_status(task_id, TaskStatus.PAUSED)
                
                logger.info(f"Tarea recurrente {task_id} pausada correctamente")
                return True
            
            except Exception as e:
                logger.error(f"Error al pausar tarea recurrente {task_id}: {str(e)}")
                return False
    
    def resume_task(self, task_id: str) -> bool:
        """
        Reanuda una tarea recurrente pausada.
        
        Args:
            task_id: ID de la tarea a reanudar
            
        Returns:
            bool: True si se reanudó correctamente, False en caso contrario
        """
        with self.recurring_tasks_lock:
            if task_id not in self.recurring_tasks:
                logger.warning(f"No se encontró tarea recurrente {task_id} para reanudar")
                return False
            
            job_id = self.recurring_tasks[task_id]
            
            try:
                # Reanudar job en APScheduler
                self.scheduler.resume_job(job_id)
                
                # Actualizar estado
                self._update_task_status(task_id, TaskStatus.SCHEDULED)
                
                logger.info(f"Tarea recurrente {task_id} reanudada correctamente")
                return True
            
            except Exception as e:
                logger.error(f"Error al reanudar tarea recurrente {task_id}: {str(e)}")
                return False
    
    def modify_schedule(self, task_id: str, new_schedule: Dict[str, Any]) -> bool:
        """
        Modifica la programación de una tarea recurrente.
        
        Args:
            task_id: ID de la tarea a modificar
            new_schedule: Nueva configuración de programación
            
        Returns:
            bool: True si se modificó correctamente, False en caso contrario
        """
        with self.recurring_tasks_lock, self.active_tasks_lock:
            if task_id not in self.recurring_tasks or task_id not in self.active_tasks:
                logger.warning(f"No se encontró tarea recurrente {task_id} para modificar")
                return False
            
            job_id = self.recurring_tasks[task_id]
            task_info = self.active_tasks[task_id]
            
            # Crear nuevo trigger
            trigger = self._create_trigger(new_schedule)
            
            if trigger is None:
                logger.error(f"Configuración de programación inválida para tarea {task_id}")
                return False
            
            try:
                # Modificar job en APScheduler
                self.scheduler.reschedule_job(
                    job_id,
                    trigger=trigger
                )
                
                # Actualizar configuración en la tarea
                if 'task' in task_info:
                    task_info['task'].data['schedule'] = new_schedule
                
                # Actualizar próxima ejecución
                job = self.scheduler.get_job(job_id)
                if job and job.next_run_time:
                    task_info['next_run'] = job.next_run_time.timestamp()
                
                logger.info(f"Programación de tarea {task_id} modificada correctamente")
                return True
            
            except Exception as e:
                logger.error(f"Error al modificar programación de tarea {task_id}: {str(e)}")
                return False
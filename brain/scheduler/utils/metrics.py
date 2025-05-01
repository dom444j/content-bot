"""
Módulo de métricas para el sistema de planificación.

Este módulo proporciona funcionalidades para recolectar, agregar y reportar
métricas relacionadas con el rendimiento del planificador de tareas.
"""

import logging
import time
import threading
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict, deque

from ..core.task_model import Task, TaskStatus

# Configurar logging
logger = logging.getLogger('Scheduler.Utils.Metrics')

class MetricsCollector:
    """
    Recolector de métricas para el sistema de planificación.
    
    Esta clase se encarga de recolectar, agregar y reportar métricas
    sobre el rendimiento del planificador, incluyendo tiempos de ejecución,
    tasas de éxito/fallo, y estadísticas de colas.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el recolector de métricas.
        
        Args:
            config: Configuración opcional para el recolector
        """
        self.config = config or {}
        
        # Intervalo de agregación de métricas (segundos)
        self.aggregation_interval = self.config.get('aggregation_interval', 60)
        
        # Período de retención de métricas (horas)
        self.retention_period = self.config.get('retention_period', 24)
        
        # Directorio para almacenar métricas
        self.metrics_dir = self.config.get('metrics_dir', os.path.join('logs', 'metrics', 'scheduler'))
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Métricas en memoria
        self._metrics = {
            'task_execution': defaultdict(list),  # Tiempos de ejecución por tipo de tarea
            'task_status': defaultdict(lambda: defaultdict(int)),  # Contadores de estado por tipo
            'queue_stats': [],  # Estadísticas de cola (longitud, tiempo de espera)
            'executor_stats': defaultdict(list),  # Estadísticas por ejecutor
            'error_counts': defaultdict(int),  # Conteo de errores por tipo
            'throughput': [],  # Tareas procesadas por intervalo
        }
        
        # Métricas agregadas
        self._aggregated_metrics = {
            'hourly': defaultdict(list),
            'daily': defaultdict(list),
        }
        
        # Historial reciente para cálculos en tiempo real
        self._recent_execution_times = defaultdict(lambda: deque(maxlen=100))
        self._recent_queue_times = deque(maxlen=100)
        self._recent_throughput = deque(maxlen=10)
        
        # Lock para operaciones thread-safe
        self._lock = threading.RLock()
        
        # Iniciar hilo de agregación si está configurado
        self._stop_aggregation = threading.Event()
        if self.config.get('auto_aggregate', True):
            self._aggregation_thread = threading.Thread(
                target=self._aggregation_loop,
                daemon=True,
                name="MetricsAggregationThread"
            )
            self._aggregation_thread.start()
            logger.debug("Hilo de agregación de métricas iniciado")
        
        logger.info("MetricsCollector inicializado")
    
    def record_task_execution(self, task: Task, execution_time: float) -> None:
        """
        Registra el tiempo de ejecución de una tarea.
        
        Args:
            task: Tarea ejecutada
            execution_time: Tiempo de ejecución en segundos
        """
        with self._lock:
            task_type = task.task_type
            timestamp = datetime.now()
            
            # Registrar en métricas generales
            self._metrics['task_execution'][task_type].append({
                'task_id': task.task_id,
                'execution_time': execution_time,
                'timestamp': timestamp.isoformat(),
                'status': task.status.value if isinstance(task.status, Enum) else task.status,
            })
            
            # Actualizar historial reciente
            self._recent_execution_times[task_type].append(execution_time)
            
            # Actualizar contadores de estado
            status_key = task.status.value if isinstance(task.status, Enum) else task.status
            self._metrics['task_status'][task_type][status_key] += 1
            
            logger.debug(f"Registrado tiempo de ejecución para tarea {task.task_id}: {execution_time:.4f}s")
    
    def record_queue_stats(self, queue_length: int, avg_wait_time: float) -> None:
        """
        Registra estadísticas de la cola de tareas.
        
        Args:
            queue_length: Longitud actual de la cola
            avg_wait_time: Tiempo promedio de espera en cola (segundos)
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Registrar en métricas generales
            self._metrics['queue_stats'].append({
                'queue_length': queue_length,
                'avg_wait_time': avg_wait_time,
                'timestamp': timestamp.isoformat(),
            })
            
            # Actualizar historial reciente
            self._recent_queue_times.append(avg_wait_time)
            
            logger.debug(f"Registradas estadísticas de cola: longitud={queue_length}, tiempo_espera={avg_wait_time:.4f}s")
    
    def record_executor_stats(self, executor_type: str, active_workers: int, 
                             pending_tasks: int, completed_tasks: int) -> None:
        """
        Registra estadísticas de un ejecutor.
        
        Args:
            executor_type: Tipo de ejecutor (thread, process, etc.)
            active_workers: Número de workers activos
            pending_tasks: Número de tareas pendientes
            completed_tasks: Número de tareas completadas desde el último registro
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Registrar en métricas generales
            self._metrics['executor_stats'][executor_type].append({
                'active_workers': active_workers,
                'pending_tasks': pending_tasks,
                'completed_tasks': completed_tasks,
                'timestamp': timestamp.isoformat(),
            })
            
            logger.debug(f"Registradas estadísticas de ejecutor {executor_type}: "
                       f"workers={active_workers}, pendientes={pending_tasks}, completadas={completed_tasks}")
    
    def record_error(self, error_type: str, task_id: Optional[str] = None) -> None:
        """
        Registra un error ocurrido durante la ejecución.
        
        Args:
            error_type: Tipo de error
            task_id: ID de la tarea relacionada (opcional)
        """
        with self._lock:
            # Incrementar contador de errores
            self._metrics['error_counts'][error_type] += 1
            
            # Si hay tarea relacionada, registrar en detalle
            if task_id:
                if 'error_details' not in self._metrics:
                    self._metrics['error_details'] = []
                
                self._metrics['error_details'].append({
                    'error_type': error_type,
                    'task_id': task_id,
                    'timestamp': datetime.now().isoformat(),
                })
            
            logger.debug(f"Registrado error de tipo '{error_type}'" + 
                       (f" para tarea {task_id}" if task_id else ""))
    
    def record_throughput(self, tasks_processed: int, time_period: float) -> None:
        """
        Registra el rendimiento del sistema (tareas procesadas por unidad de tiempo).
        
        Args:
            tasks_processed: Número de tareas procesadas
            time_period: Período de tiempo en segundos
        """
        with self._lock:
            timestamp = datetime.now()
            throughput = tasks_processed / time_period if time_period > 0 else 0
            
            # Registrar en métricas generales
            self._metrics['throughput'].append({
                'tasks_processed': tasks_processed,
                'time_period': time_period,
                'throughput': throughput,
                'timestamp': timestamp.isoformat(),
            })
            
            # Actualizar historial reciente
            self._recent_throughput.append(throughput)
            
            logger.debug(f"Registrado throughput: {throughput:.2f} tareas/segundo "
                       f"({tasks_processed} tareas en {time_period:.2f}s)")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas actuales calculadas a partir del historial reciente.
        
        Returns:
            Diccionario con métricas actuales
        """
        with self._lock:
            current_metrics = {
                'execution_times': {},
                'queue': {},
                'throughput': {},
                'error_rates': {},
                'task_status_counts': {},
                'timestamp': datetime.now().isoformat(),
            }
            
            # Calcular tiempos de ejecución por tipo de tarea
            for task_type, times in self._recent_execution_times.items():
                if times:
                    current_metrics['execution_times'][task_type] = {
                        'avg': statistics.mean(times),
                        'min': min(times),
                        'max': max(times),
                        'p95': self._percentile(times, 95),
                        'count': len(times),
                    }
            
            # Calcular estadísticas de cola
            if self._recent_queue_times:
                current_metrics['queue'] = {
                    'avg_wait_time': statistics.mean(self._recent_queue_times),
                    'max_wait_time': max(self._recent_queue_times),
                    'p95_wait_time': self._percentile(self._recent_queue_times, 95),
                }
            
            # Calcular throughput
            if self._recent_throughput:
                current_metrics['throughput'] = {
                    'current': self._recent_throughput[-1] if self._recent_throughput else 0,
                    'avg': statistics.mean(self._recent_throughput) if self._recent_throughput else 0,
                }
            
            # Calcular tasas de error
            total_tasks = sum(sum(counts.values()) for counts in self._metrics['task_status'].values())
            total_errors = sum(self._metrics['error_counts'].values())
            
            if total_tasks > 0:
                current_metrics['error_rates'] = {
                    'overall': total_errors / total_tasks,
                    'by_type': {error_type: count / total_tasks 
                               for error_type, count in self._metrics['error_counts'].items()}
                }
            
            # Conteos de estado por tipo de tarea
            current_metrics['task_status_counts'] = dict(self._metrics['task_status'])
            
            return current_metrics
    
    def aggregate_metrics(self, force_persist: bool = False) -> Dict[str, Any]:
        """
        Agrega métricas y opcionalmente las persiste.
        
        Args:
            force_persist: Si es True, fuerza la persistencia de métricas
            
        Returns:
            Diccionario con métricas agregadas
        """
        with self._lock:
            now = datetime.now()
            current_hour = now.replace(minute=0, second=0, microsecond=0)
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Agregar métricas por hora
            hourly_metrics = self._calculate_aggregated_metrics(
                timeframe_start=current_hour,
                timeframe_end=now
            )
            self._aggregated_metrics['hourly'].append({
                'timestamp': current_hour.isoformat(),
                'metrics': hourly_metrics
            })
            
            # Agregar métricas por día
            daily_metrics = self._calculate_aggregated_metrics(
                timeframe_start=current_day,
                timeframe_end=now
            )
            self._aggregated_metrics['daily'].append({
                'timestamp': current_day.isoformat(),
                'metrics': daily_metrics
            })
            
            # Limpiar métricas antiguas
            self._clean_old_metrics()
            
            # Persistir si es necesario
            if force_persist or self.config.get('auto_persist', True):
                self._persist_metrics()
            
            logger.debug("Métricas agregadas")
            
            return {
                'hourly': hourly_metrics,
                'daily': daily_metrics
            }
    
    def _calculate_aggregated_metrics(self, timeframe_start: datetime, 
                                     timeframe_end: datetime) -> Dict[str, Any]:
        """
        Calcula métricas agregadas para un período de tiempo específico.
        
        Args:
            timeframe_start: Inicio del período
            timeframe_end: Fin del período
            
        Returns:
            Diccionario con métricas agregadas
        """
        # Filtrar métricas dentro del período
        def is_in_timeframe(timestamp_str):
            timestamp = datetime.fromisoformat(timestamp_str)
            return timeframe_start <= timestamp <= timeframe_end
        
        # Métricas agregadas
        aggregated = {
            'task_execution': {},
            'queue_stats': {},
            'error_counts': dict(self._metrics['error_counts']),
            'throughput': {},
            'task_status_counts': {},
        }
        
        # Agregar tiempos de ejecución por tipo de tarea
        for task_type, executions in self._metrics['task_execution'].items():
            # Filtrar por período
            period_executions = [
                e['execution_time'] for e in executions 
                if is_in_timeframe(e['timestamp'])
            ]
            
            if period_executions:
                aggregated['task_execution'][task_type] = {
                    'count': len(period_executions),
                    'avg': statistics.mean(period_executions),
                    'min': min(period_executions),
                    'max': max(period_executions),
                    'p95': self._percentile(period_executions, 95),
                }
        
        # Agregar estadísticas de cola
        queue_stats = [
            s for s in self._metrics['queue_stats']
            if is_in_timeframe(s['timestamp'])
        ]
        
        if queue_stats:
            queue_lengths = [s['queue_length'] for s in queue_stats]
            wait_times = [s['avg_wait_time'] for s in queue_stats]
            
            aggregated['queue_stats'] = {
                'avg_length': statistics.mean(queue_lengths),
                'max_length': max(queue_lengths),
                'avg_wait_time': statistics.mean(wait_times),
                'max_wait_time': max(wait_times),
                'p95_wait_time': self._percentile(wait_times, 95),
            }
        
        # Agregar throughput
        throughput_metrics = [
            t for t in self._metrics['throughput']
            if is_in_timeframe(t['timestamp'])
        ]
        
        if throughput_metrics:
            throughput_values = [t['throughput'] for t in throughput_metrics]
            tasks_processed = sum(t['tasks_processed'] for t in throughput_metrics)
            
            aggregated['throughput'] = {
                'avg': statistics.mean(throughput_values),
                'max': max(throughput_values),
                'total_tasks': tasks_processed,
            }
        
        # Agregar conteos de estado
        aggregated['task_status_counts'] = dict(self._metrics['task_status'])
        
        return aggregated
    
    def _clean_old_metrics(self) -> None:
        """
        Elimina métricas antiguas según el período de retención.
        """
        now = datetime.now()
        retention_limit = now - timedelta(hours=self.retention_period)
        
        # Limpiar métricas de ejecución
        for task_type in self._metrics['task_execution']:
            self._metrics['task_execution'][task_type] = [
                e for e in self._metrics['task_execution'][task_type]
                if datetime.fromisoformat(e['timestamp']) > retention_limit
            ]
        
        # Limpiar estadísticas de cola
        self._metrics['queue_stats'] = [
            s for s in self._metrics['queue_stats']
            if datetime.fromisoformat(s['timestamp']) > retention_limit
        ]
        
        # Limpiar estadísticas de ejecutor
        for executor_type in self._metrics['executor_stats']:
            self._metrics['executor_stats'][executor_type] = [
                s for s in self._metrics['executor_stats'][executor_type]
                if datetime.fromisoformat(s['timestamp']) > retention_limit
            ]
        
        # Limpiar throughput
        self._metrics['throughput'] = [
            t for t in self._metrics['throughput']
            if datetime.fromisoformat(t['timestamp']) > retention_limit
        ]
        
        # Limpiar métricas agregadas
        for period in ['hourly', 'daily']:
            self._aggregated_metrics[period] = [
                m for m in self._aggregated_metrics[period]
                if datetime.fromisoformat(m['timestamp']) > retention_limit
            ]
        
        logger.debug(f"Limpiadas métricas anteriores a {retention_limit.isoformat()}")
    
    def _persist_metrics(self) -> None:
        """
        Persiste las métricas agregadas en archivos.
        """
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        
        # Guardar métricas agregadas
        for period in ['hourly', 'daily']:
            if self._aggregated_metrics[period]:
                filename = f"scheduler_metrics_{period}_{date_str}.json"
                filepath = os.path.join(self.metrics_dir, filename)
                
                try:
                    with open(filepath, 'w') as f:
                        json.dump(self._aggregated_metrics[period], f, indent=2)
                    
                    logger.debug(f"Métricas {period} guardadas en {filepath}")
                except Exception as e:
                    logger.error(f"Error al guardar métricas {period}: {str(e)}")
        
        # Guardar métricas actuales
        try:
            current_metrics = self.get_current_metrics()
            filename = f"scheduler_metrics_current_{date_str}.json"
            filepath = os.path.join(self.metrics_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            
            logger.debug(f"Métricas actuales guardadas en {filepath}")
        except Exception as e:
            logger.error(f"Error al guardar métricas actuales: {str(e)}")
    
    def _aggregation_loop(self) -> None:
        """
        Bucle de agregación periódica de métricas.
        """
        while not self._stop_aggregation.is_set():
            try:
                # Dormir hasta el próximo intervalo
                time.sleep(self.aggregation_interval)
                
                # Agregar métricas
                self.aggregate_metrics()
            except Exception as e:
                logger.error(f"Error en bucle de agregación de métricas: {str(e)}")
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """
        Calcula un percentil específico de una lista de valores.
        
        Args:
            data: Lista de valores numéricos
            percentile: Percentil a calcular (0-100)
            
        Returns:
            Valor del percentil
        """
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_data):
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        else:
            return sorted_data[f]
    
    def stop(self) -> None:
        """
        Detiene el recolector de métricas y persiste los datos.
        """
        if hasattr(self, '_aggregation_thread') and self._aggregation_thread.is_alive():
            self._stop_aggregation.set()
            self._aggregation_thread.join(timeout=5.0)
            logger.debug("Hilo de agregación de métricas detenido")
        
        # Persistir métricas finales
        self.aggregate_metrics(force_persist=True)
        logger.info("MetricsCollector detenido y métricas finales persistidas")


class MetricsReporter:
    """
    Generador de informes de métricas para el sistema de planificación.
    
    Esta clase se encarga de generar informes y visualizaciones
    basados en las métricas recolectadas.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, config: Dict[str, Any] = None):
        """
        Inicializa el generador de informes.
        
        Args:
            metrics_collector: Recolector de métricas
            config: Configuración opcional
        """
        self.metrics_collector = metrics_collector
        self.config = config or {}
        
        # Directorio para informes
        self.reports_dir = self.config.get('reports_dir', os.path.join('logs', 'reports', 'scheduler'))
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info("MetricsReporter inicializado")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Genera un informe resumido de las métricas actuales.
        
        Returns:
            Diccionario con el informe resumido
        """
        current_metrics = self.metrics_collector.get_current_metrics()
        
        # Crear informe resumido
        summary = {
            'timestamp': current_metrics['timestamp'],
            'performance': {
                'avg_execution_time': self._calculate_avg_execution_time(current_metrics),
                'throughput': current_metrics.get('throughput', {}).get('avg', 0),
                'queue_wait_time': current_metrics.get('queue', {}).get('avg_wait_time', 0),
            },
            'reliability': {
                'error_rate': current_metrics.get('error_rates', {}).get('overall', 0),
                'top_errors': self._get_top_errors(current_metrics, limit=3),
            },
            'task_distribution': self._summarize_task_distribution(current_metrics),
        }
        
        return summary
    
    def generate_detailed_report(self, timeframe: str = 'daily') -> Dict[str, Any]:
        """
        Genera un informe detallado para un período específico.
        
        Args:
            timeframe: Período de tiempo ('hourly' o 'daily')
            
        Returns:
            Diccionario con el informe detallado
        """
        # Forzar agregación para tener datos actualizados
        aggregated = self.metrics_collector.aggregate_metrics()
        
        # Obtener métricas del período solicitado
        period_metrics = aggregated.get(timeframe, {})
        
        # Crear informe detallado
        detailed_report = {
            'timeframe': timeframe,
            'generated_at': datetime.now().isoformat(),
            'performance_trends': self._calculate_performance_trends(period_metrics),
            'task_type_analysis': self._analyze_task_types(period_metrics),
            'error_analysis': self._analyze_errors(period_metrics),
            'resource_utilization': self._analyze_resource_utilization(period_metrics),
        }
        
        return detailed_report
    
    def save_report(self, report: Dict[str, Any], report_type: str) -> str:
        """
        Guarda un informe en un archivo.
        
        Args:
            report: Informe a guardar
            report_type: Tipo de informe ('summary', 'detailed', etc.)
            
        Returns:
            Ruta del archivo guardado
        """
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H%M%S')
        
        filename = f"scheduler_{report_type}_report_{date_str}_{time_str}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.debug(f"Informe {report_type} guardado en {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error al guardar informe {report_type}: {str(e)}")
            return ""
    
    def _calculate_avg_execution_time(self, metrics: Dict[str, Any]) -> float:
        """
        Calcula el tiempo medio de ejecución de todas las tareas.
        
        Args:
            metrics: Métricas actuales
            
        Returns:
            Tiempo medio de ejecución en segundos
        """
        execution_times = metrics.get('execution_times', {})
        if not execution_times:
            return 0.0
        
        total_time = 0.0
        total_count = 0
        
        for task_type, stats in execution_times.items():
            avg_time = stats.get('avg', 0)
            count = stats.get('count', 0)
            
            total_time += avg_time * count
            total_count += count
        
        return total_time / total_count if total_count > 0 else 0.0
    
    def _get_top_errors(self, metrics: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Obtiene los errores más frecuentes.
        
        Args:
            metrics: Métricas actuales
            limit: Número máximo de errores a devolver
            
        Returns:
            Lista de errores más frecuentes
        """
        error_rates = metrics.get('error_rates', {}).get('by_type', {})
        
        # Ordenar errores por tasa
        sorted_errors = sorted(
            [{'type': t, 'rate': r} for t, r in error_rates.items()],
            key=lambda x: x['rate'],
            reverse=True
        )
        
        return sorted_errors[:limit]
    
    def _summarize_task_distribution(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resume la distribución de tareas por tipo y estado.
        
        Args:
            metrics: Métricas actuales
            
        Returns:
            Resumen de distribución de tareas
        """
        task_status_counts = metrics.get('task_status_counts', {})
        
        # Calcular totales por tipo de tarea
        task_type_totals = {}
        for task_type, status_counts in task_status_counts.items():
            task_type_totals[task_type] = sum(status_counts.values())
        
        # Calcular distribución por estado
        status_distribution = defaultdict(int)
        for task_type, status_counts in task_status_counts.items():
            for status, count in status_counts.items():
                status_distribution[status] += count
        
        return {
            'by_type': task_type_totals,
            'by_status': dict(status_distribution),
        }
    
    def _calculate_performance_trends(self, period_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula tendencias de rendimiento a lo largo del tiempo.
        
        Args:
            period_metrics: Métricas del período
            
        Returns:
            Tendencias de rendimiento
        """
        # Implementación simplificada - en una versión real se analizarían
        # las tendencias a lo largo del tiempo
        return {
            'execution_time_trend': 'stable',
            'throughput_trend': 'increasing',
            'error_rate_trend': 'decreasing',
        }
    
    def _analyze_task_types(self, period_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza el rendimiento por tipo de tarea.
        
        Args:
            period_metrics: Métricas del período
            
        Returns:
            Análisis por tipo de tarea
        """
        # Implementación simplificada
        return {
            'fastest_task_type': 'content_analysis',
            'slowest_task_type': 'video_generation',
            'most_frequent_task_type': 'social_media_post',
        }
    
    def _analyze_errors(self, period_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza patrones de errores.
        
        Args:
            period_metrics: Métricas del período
            
        Returns:
            Análisis de errores
        """
        # Implementación simplificada
        return {
            'most_common_errors': ['api_timeout', 'resource_not_found'],
            'error_patterns': {
                'time_of_day': 'higher_at_peak_hours',
                'task_correlation': 'higher_for_external_api_calls',
            },
        }
    
    def _analyze_resource_utilization(self, period_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza la utilización de recursos.
        
        Args:
            period_metrics: Métricas del período
            
        Returns:
            Análisis de utilización de recursos
        """
        # Implementación simplificada
        return {
            'worker_utilization': 0.75,  # 75% de utilización
            'queue_saturation': 0.30,    # 30% de saturación
            'bottlenecks': ['external_api_rate_limits'],
        }
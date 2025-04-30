"""
Sistema de monitoreo y métricas para el Orchestrator
"""
import time
import threading
import json
import os
import datetime
import logging
from collections import deque

# Intentar importar prometheus_client si está disponible
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configuración
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Recolector de métricas para el Orchestrator"""
    
    def __init__(self):
        self.metrics = {
            "tasks": {
                "created": 0,
                "completed": 0,
                "failed": 0,
                "by_type": {}
            },
            "publications": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "by_platform": {}
            },
            "content": {
                "generated": 0,
                "published": 0,
                "monetized": 0
            },
            "shadowbans": {
                "detected": 0,
                "recovered": 0,
                "by_platform": {}
            },
            "performance": {
                "response_times": {},
                "memory_usage": [],
                "cpu_usage": []
            },
            "monetization": {
                "revenue": 0,
                "by_channel": {},
                "by_platform": {}
            }
        }
        
        self.performance_history = {
            "response_times": deque(maxlen=1000),
            "memory_usage": deque(maxlen=100),
            "cpu_usage": deque(maxlen=100)
        }
        
        # Inicializar métricas de Prometheus si está disponible
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
            
        # Iniciar hilo de monitoreo de recursos
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()
    
    def _setup_prometheus_metrics(self):
        """Configura métricas de Prometheus"""
        # Contadores
        self.task_counter = Counter(
            'orchestrator_tasks_total', 
            'Número total de tareas procesadas',
            ['type', 'status']
        )
        
        self.publication_counter = Counter(
            'orchestrator_publications_total',
            'Número total de publicaciones',
            ['platform', 'status']
        )
        
        self.shadowban_counter = Counter(
            'orchestrator_shadowbans_total',
            'Número total de shadowbans detectados',
            ['platform']
        )
        
        # Gauges
        self.active_tasks_gauge = Gauge(
            'orchestrator_active_tasks',
            'Número de tareas activas',
            ['type']
        )
        
        self.memory_usage_gauge = Gauge(
            'orchestrator_memory_usage_bytes',
            'Uso de memoria en bytes'
        )
        
        self.cpu_usage_gauge = Gauge(
            'orchestrator_cpu_usage_percent',
            'Uso de CPU en porcentaje'
        )
        
        # Histogramas
        self.response_time_histogram = Histogram(
            'orchestrator_response_time_seconds',
            'Tiempo de respuesta en segundos',
            ['function']
        )
        
        # Iniciar servidor de métricas
        prometheus_client.start_http_server(9090)
        logger.info("Servidor de métricas Prometheus iniciado en puerto 9090")
    
    def _monitor_resources(self):
        """Monitorea recursos del sistema"""
        import psutil
        
        while self.monitoring_active:
            try:
                # Obtener uso de memoria
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_usage = memory_info.rss
                
                # Obtener uso de CPU
                cpu_percent = process.cpu_percent(interval=1.0)
                
                # Guardar métricas
                timestamp = datetime.datetime.now().isoformat()
                
                self.performance_history["memory_usage"].append({
                    "timestamp": timestamp,
                    "value": memory_usage
                })
                
                self.performance_history["cpu_usage"].append({
                    "timestamp": timestamp,
                    "value": cpu_percent
                })
                
                # Actualizar métricas de Prometheus si está disponible
                if PROMETHEUS_AVAILABLE:
                    self.memory_usage_gauge.set(memory_usage)
                    self.cpu_usage_gauge.set(cpu_percent)
                
                # Guardar métricas en archivo cada 60 segundos
                if int(time.time()) % 60 == 0:
                    self._save_metrics()
                
                time.sleep(10)  # Monitorear cada 10 segundos
            except Exception as e:
                logger.error(f"Error en monitoreo de recursos: {str(e)}")
                time.sleep(30)  # Esperar más tiempo en caso de error
    
    def record_task(self, task_type, status):
        """Registra una tarea"""
        self.metrics["tasks"][status] += 1
        
        if task_type not in self.metrics["tasks"]["by_type"]:
            self.metrics["tasks"]["by_type"][task_type] = {
                "created": 0,
                "completed": 0,
                "failed": 0
            }
        
        self.metrics["tasks"]["by_type"][task_type][status] += 1
        
        # Actualizar métricas de Prometheus si está disponible
        if PROMETHEUS_AVAILABLE:
            self.task_counter.labels(type=task_type, status=status).inc()
    
    def record_publication(self, platform, status):
        """Registra una publicación"""
        self.metrics["publications"][status] += 1
        
        if platform not in self.metrics["publications"]["by_platform"]:
            self.metrics["publications"]["by_platform"][platform] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0
            }
        
        self.metrics["publications"]["by_platform"][platform][status] += 1
        
        # Actualizar métricas de Prometheus si está disponible
        if PROMETHEUS_AVAILABLE:
            self.publication_counter.labels(platform=platform, status=status).inc()
    
    def record_shadowban(self, platform):
        """Registra un shadowban"""
        self.metrics["shadowbans"]["detected"] += 1
        
        if platform not in self.metrics["shadowbans"]["by_platform"]:
            self.metrics["shadowbans"]["by_platform"][platform] = {
                "detected": 0,
                "recovered": 0
            }
        
        self.metrics["shadowbans"]["by_platform"][platform]["detected"] += 1
        
        # Actualizar métricas de Prometheus si está disponible
        if PROMETHEUS_AVAILABLE:
            self.shadowban_counter.labels(platform=platform).inc()
    
    def record_response_time(self, function_name, duration_ms):
        """Registra tiempo de respuesta"""
        if function_name not in self.metrics["performance"]["response_times"]:
            self.metrics["performance"]["response_times"][function_name] = {
                "count": 0,
                "total_ms": 0,
                "min_ms": float('inf'),
                "max_ms": 0
            }
        
        stats = self.metrics["performance"]["response_times"][function_name]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["min_ms"] = min(stats["min_ms"], duration_ms)
        stats["max_ms"] = max(stats["max_ms"], duration_ms)
        
        self.performance_history["response_times"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "function": function_name,
            "duration_ms": duration_ms
        })
        
        # Actualizar métricas de Prometheus si está disponible
        if PROMETHEUS_AVAILABLE:
            self.response_time_histogram.labels(function=function_name).observe(duration_ms / 1000.0)
    
    def record_monetization(self, channel_id, platform, amount):
        """Registra ingresos de monetización"""
        self.metrics["monetization"]["revenue"] += amount
        
        if channel_id not in self.metrics["monetization"]["by_channel"]:
            self.metrics["monetization"]["by_channel"][channel_id] = 0
        
        if platform not in self.metrics["monetization"]["by_platform"]:
            self.metrics["monetization"]["by_platform"][platform] = 0
        
        self.metrics["monetization"]["by_channel"][channel_id] += amount
        self.metrics["monetization"]["by_platform"][platform] += amount
    
    def _save_metrics(self):
        """Guarda métricas en archivo"""
        try:
            metrics_file = os.path.join(METRICS_DIR, "orchestrator_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
            
            # Guardar historial de rendimiento
            performance_file = os.path.join(METRICS_DIR, "performance_history.json")
            with open(performance_file, "w") as f:
                json.dump({
                    "response_times": list(self.performance_history["response_times"]),
                    "memory_usage": list(self.performance_history["memory_usage"]),
                    "cpu_usage": list(self.performance_history["cpu_usage"])
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar métricas: {str(e)}")
    
    def stop(self):
        """Detiene el monitoreo"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self._save_metrics()

# Instancia global
metrics_collector = MetricsCollector()

# Decorador para monitorear rendimiento
def monitor_performance(func):
    """Decorador para monitorear el rendimiento de funciones"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            metrics_collector.record_response_time(func.__name__, duration_ms)
    return wrapper
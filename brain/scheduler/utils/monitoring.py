"""
Módulo de monitoreo y alertas para el sistema de planificación.

Este módulo proporciona funcionalidades para:
- Monitorear el estado de salud del sistema de planificación
- Generar alertas cuando se detectan problemas
- Recopilar métricas de rendimiento en tiempo real
- Integrarse con sistemas externos de monitoreo
"""

import logging
import time
import threading
import json
import os
import datetime
from typing import Dict, List, Any, Optional, Callable, Union
import psutil
import requests
from enum import Enum

# Configurar logger
logger = logging.getLogger("scheduler.monitoring")

class AlertLevel(Enum):
    """Niveles de alerta para el sistema de monitoreo."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Canales disponibles para envío de alertas."""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"

class HealthStatus(Enum):
    """Estados de salud del sistema."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class MonitoringSystem:
    """
    Sistema de monitoreo para el planificador de tareas.
    
    Proporciona funcionalidades para:
    - Monitorear recursos del sistema
    - Verificar el estado de salud de componentes
    - Generar y enviar alertas
    - Recopilar métricas de rendimiento
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de monitoreo.
        
        Args:
            config: Configuración del sistema de monitoreo
        """
        self.config = config or {}
        self.alert_handlers = {
            AlertChannel.LOG: self._send_log_alert,
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SMS: self._send_sms_alert
        }
        
        # Configuración de alertas
        self.alert_channels = self.config.get('alert_channels', [AlertChannel.LOG])
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': 80,
            'memory_percent': 80,
            'disk_percent': 90,
            'task_queue_size': 1000,
            'failed_tasks_threshold': 10,
            'task_latency_ms': 5000
        })
        
        # Estado interno
        self.health_checks = {}
        self.metrics = {}
        self.last_alert_times = {}
        
        # Intervalo de monitoreo en segundos
        self.monitoring_interval = self.config.get('monitoring_interval', 60)
        
        # Iniciar monitoreo en segundo plano si está configurado
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        if self.config.get('auto_start', False):
            self.start_monitoring()
    
    def start_monitoring(self):
        """Inicia el monitoreo en segundo plano."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("El monitoreo ya está en ejecución")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Monitoreo iniciado en segundo plano")
    
    def stop_monitoring(self):
        """Detiene el monitoreo en segundo plano."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("El monitoreo no está en ejecución")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5)
        logger.info("Monitoreo detenido")
    
    def _monitoring_loop(self):
        """Bucle principal de monitoreo."""
        logger.info("Iniciando bucle de monitoreo")
        
        while not self.stop_monitoring.is_set():
            try:
                # Recopilar métricas del sistema
                self.collect_system_metrics()
                
                # Ejecutar verificaciones de salud
                self.run_health_checks()
                
                # Verificar umbrales y enviar alertas si es necesario
                self.check_alert_thresholds()
                
                # Guardar métricas históricas si está configurado
                if self.config.get('store_metrics', False):
                    self.store_metrics()
                
                # Esperar hasta el próximo ciclo
                self.stop_monitoring.wait(timeout=self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error en bucle de monitoreo: {str(e)}")
                # Esperar un poco antes de reintentar en caso de error
                time.sleep(10)
    
    def collect_system_metrics(self):
        """Recopila métricas del sistema operativo y recursos."""
        try:
            # Métricas del sistema
            self.metrics['timestamp'] = datetime.datetime.now().isoformat()
            self.metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            self.metrics['memory'] = {
                'percent': psutil.virtual_memory().percent,
                'available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'total_mb': psutil.virtual_memory().total / (1024 * 1024)
            }
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            self.metrics['disk'] = {
                'percent': disk.percent,
                'free_gb': disk.free / (1024 * 1024 * 1024),
                'total_gb': disk.total / (1024 * 1024 * 1024)
            }
            
            # Métricas de red
            net_io = psutil.net_io_counters()
            self.metrics['network'] = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
            logger.debug(f"Métricas del sistema recopiladas: CPU {self.metrics['cpu_percent']}%, "
                        f"Memoria {self.metrics['memory']['percent']}%")
            
        except Exception as e:
            logger.error(f"Error al recopilar métricas del sistema: {str(e)}")
    
    def collect_scheduler_metrics(self, metrics_data: Dict[str, Any]):
        """
        Recopila métricas específicas del planificador.
        
        Args:
            metrics_data: Diccionario con métricas del planificador
        """
        try:
            # Actualizar métricas del planificador
            self.metrics['scheduler'] = metrics_data
            
            # Registrar algunas métricas clave
            queue_size = metrics_data.get('queue_size', 0)
            pending_tasks = metrics_data.get('pending_tasks', 0)
            failed_tasks = metrics_data.get('failed_tasks', 0)
            
            logger.debug(f"Métricas del planificador recopiladas: Cola {queue_size}, "
                        f"Pendientes {pending_tasks}, Fallidas {failed_tasks}")
            
        except Exception as e:
            logger.error(f"Error al recopilar métricas del planificador: {str(e)}")
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                             description: str = ""):
        """
        Registra una función de verificación de salud.
        
        Args:
            name: Nombre único para la verificación
            check_func: Función que devuelve True si la verificación pasa
            description: Descripción de la verificación
        """
        self.health_checks[name] = {
            'func': check_func,
            'description': description,
            'last_status': None,
            'last_check_time': None
        }
        logger.debug(f"Verificación de salud registrada: {name}")
    
    def run_health_checks(self):
        """Ejecuta todas las verificaciones de salud registradas."""
        overall_status = HealthStatus.HEALTHY
        failed_checks = []
        
        for name, check_info in self.health_checks.items():
            try:
                # Ejecutar la verificación
                status = check_info['func']()
                check_info['last_status'] = status
                check_info['last_check_time'] = datetime.datetime.now().isoformat()
                
                # Si alguna verificación falla, el estado general es degradado
                if not status:
                    overall_status = HealthStatus.DEGRADED
                    failed_checks.append(name)
                    
                    # Enviar alerta para la verificación fallida
                    self.send_alert(
                        f"Verificación de salud fallida: {name}",
                        AlertLevel.WARNING,
                        {
                            'check_name': name,
                            'description': check_info['description'],
                            'time': check_info['last_check_time']
                        }
                    )
            
            except Exception as e:
                # Si hay una excepción, la verificación falla
                check_info['last_status'] = False
                check_info['last_check_time'] = datetime.datetime.now().isoformat()
                check_info['last_error'] = str(e)
                
                overall_status = HealthStatus.DEGRADED
                failed_checks.append(name)
                
                # Enviar alerta para el error
                self.send_alert(
                    f"Error en verificación de salud: {name}",
                    AlertLevel.ERROR,
                    {
                        'check_name': name,
                        'description': check_info['description'],
                        'error': str(e),
                        'time': check_info['last_check_time']
                    }
                )
        
        # Si todas las verificaciones críticas fallan, el estado es no saludable
        if failed_checks and all(name in failed_checks for name in 
                               self.config.get('critical_checks', [])):
            overall_status = HealthStatus.UNHEALTHY
            
            # Enviar alerta crítica
            self.send_alert(
                "Sistema en estado no saludable - Verificaciones críticas fallidas",
                AlertLevel.CRITICAL,
                {
                    'failed_checks': failed_checks,
                    'time': datetime.datetime.now().isoformat()
                }
            )
        
        # Actualizar métricas con el resultado de las verificaciones
        self.metrics['health_status'] = overall_status.value
        self.metrics['failed_checks'] = failed_checks
        
        return overall_status
    
    def check_alert_thresholds(self):
        """Verifica umbrales de alerta y envía notificaciones si es necesario."""
        # Verificar CPU
        cpu_percent = self.metrics.get('cpu_percent', 0)
        if cpu_percent > self.alert_thresholds.get('cpu_percent', 80):
            self.send_alert(
                f"Uso de CPU alto: {cpu_percent}%",
                AlertLevel.WARNING if cpu_percent < 95 else AlertLevel.ERROR,
                {'metric': 'cpu_percent', 'value': cpu_percent}
            )
        
        # Verificar memoria
        memory_percent = self.metrics.get('memory', {}).get('percent', 0)
        if memory_percent > self.alert_thresholds.get('memory_percent', 80):
            self.send_alert(
                f"Uso de memoria alto: {memory_percent}%",
                AlertLevel.WARNING if memory_percent < 95 else AlertLevel.ERROR,
                {'metric': 'memory_percent', 'value': memory_percent}
            )
        
        # Verificar disco
        disk_percent = self.metrics.get('disk', {}).get('percent', 0)
        if disk_percent > self.alert_thresholds.get('disk_percent', 90):
            self.send_alert(
                f"Espacio en disco bajo: {disk_percent}% usado",
                AlertLevel.WARNING if disk_percent < 95 else AlertLevel.ERROR,
                {'metric': 'disk_percent', 'value': disk_percent}
            )
        
        # Verificar métricas del planificador
        scheduler_metrics = self.metrics.get('scheduler', {})
        
        # Tamaño de cola
        queue_size = scheduler_metrics.get('queue_size', 0)
        if queue_size > self.alert_thresholds.get('task_queue_size', 1000):
            self.send_alert(
                f"Cola de tareas grande: {queue_size} tareas",
                AlertLevel.WARNING,
                {'metric': 'task_queue_size', 'value': queue_size}
            )
        
        # Tareas fallidas
        failed_tasks = scheduler_metrics.get('failed_tasks', 0)
        if failed_tasks > self.alert_thresholds.get('failed_tasks_threshold', 10):
            self.send_alert(
                f"Alto número de tareas fallidas: {failed_tasks}",
                AlertLevel.ERROR,
                {'metric': 'failed_tasks', 'value': failed_tasks}
            )
        
        # Latencia de tareas
        task_latency = scheduler_metrics.get('avg_task_latency_ms', 0)
        if task_latency > self.alert_thresholds.get('task_latency_ms', 5000):
            self.send_alert(
                f"Latencia alta en ejecución de tareas: {task_latency}ms",
                AlertLevel.WARNING,
                {'metric': 'task_latency_ms', 'value': task_latency}
            )
    
    def send_alert(self, message: str, level: AlertLevel, 
                  context: Dict[str, Any] = None):
        """
        Envía una alerta a través de los canales configurados.
        
        Args:
            message: Mensaje de la alerta
            level: Nivel de severidad
            context: Información adicional para la alerta
        """
        # Evitar alertas repetitivas en un corto período
        alert_key = f"{level.value}:{message}"
        current_time = time.time()
        
        # Verificar si ya se envió una alerta similar recientemente
        if alert_key in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_key]
            cooldown = self.config.get('alert_cooldown_seconds', {}).get(
                level.value, 300)  # 5 minutos por defecto
            
            if time_since_last < cooldown:
                logger.debug(f"Alerta {alert_key} en período de enfriamiento, "
                           f"ignorando por {cooldown - time_since_last:.1f}s más")
                return
        
        # Actualizar tiempo de última alerta
        self.last_alert_times[alert_key] = current_time
        
        # Preparar datos de la alerta
        alert_data = {
            'message': message,
            'level': level.value,
            'timestamp': datetime.datetime.now().isoformat(),
            'context': context or {}
        }
        
        # Enviar a todos los canales configurados
        for channel in self.alert_channels:
            try:
                if isinstance(channel, str):
                    channel = AlertChannel(channel)
                
                handler = self.alert_handlers.get(channel)
                if handler:
                    handler(alert_data)
                else:
                    logger.warning(f"Canal de alerta no implementado: {channel}")
            
            except Exception as e:
                logger.error(f"Error al enviar alerta a {channel}: {str(e)}")
    
    def _send_log_alert(self, alert_data: Dict[str, Any]):
        """Envía una alerta al sistema de logging."""
        level = alert_data.get('level', 'info')
        message = alert_data.get('message', '')
        context = alert_data.get('context', {})
        
        log_message = f"{message} | Contexto: {json.dumps(context)}"
        
        if level == AlertLevel.INFO.value:
            logger.info(log_message)
        elif level == AlertLevel.WARNING.value:
            logger.warning(log_message)
        elif level == AlertLevel.ERROR.value:
            logger.error(log_message)
        elif level == AlertLevel.CRITICAL.value:
            logger.critical(log_message)
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Envía una alerta por correo electrónico."""
        if not self.config.get('email'):
            logger.warning("Configuración de email no disponible para alertas")
            return
        
        # Aquí iría la implementación para enviar emails
        # Usando smtplib o alguna biblioteca de terceros
        logger.info(f"Simulando envío de alerta por email: {alert_data['message']}")
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Envía una alerta a Slack."""
        if not self.config.get('slack_webhook'):
            logger.warning("Webhook de Slack no configurado para alertas")
            return
        
        try:
            webhook_url = self.config['slack_webhook']
            
            # Formatear mensaje para Slack
            level_emoji = {
                'info': ':information_source:',
                'warning': ':warning:',
                'error': ':x:',
                'critical': ':rotating_light:'
            }
            
            emoji = level_emoji.get(alert_data['level'], ':bell:')
            
            slack_payload = {
                'text': f"{emoji} *ALERTA*: {alert_data['message']}",
                'attachments': [{
                    'color': {
                        'info': '#36a64f',
                        'warning': '#ffcc00',
                        'error': '#ff0000',
                        'critical': '#9b0000'
                    }.get(alert_data['level'], '#cccccc'),
                    'fields': [
                        {
                            'title': 'Nivel',
                            'value': alert_data['level'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert_data['timestamp'],
                            'short': True
                        }
                    ],
                    'text': f"```{json.dumps(alert_data['context'], indent=2)}```"
                }]
            }
            
            # Enviar a Slack (simulado)
            logger.info(f"Simulando envío de alerta a Slack: {alert_data['message']}")
            # En implementación real:
            # requests.post(webhook_url, json=slack_payload)
            
        except Exception as e:
            logger.error(f"Error al enviar alerta a Slack: {str(e)}")
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Envía una alerta a un webhook genérico."""
        if not self.config.get('webhook_url'):
            logger.warning("URL de webhook no configurada para alertas")
            return
        
        try:
            webhook_url = self.config['webhook_url']
            
            # En implementación real:
            # requests.post(webhook_url, json=alert_data)
            logger.info(f"Simulando envío de alerta a webhook: {alert_data['message']}")
            
        except Exception as e:
            logger.error(f"Error al enviar alerta a webhook: {str(e)}")
    
    def _send_sms_alert(self, alert_data: Dict[str, Any]):
        """Envía una alerta por SMS."""
        if not self.config.get('sms'):
            logger.warning("Configuración de SMS no disponible para alertas")
            return
        
        # Aquí iría la implementación para enviar SMS
        # Usando algún proveedor como Twilio
        logger.info(f"Simulando envío de alerta por SMS: {alert_data['message']}")
    
    def store_metrics(self):
        """Almacena métricas históricas para análisis posterior."""
        if not self.config.get('metrics_file'):
            return
        
        try:
            metrics_file = self.config['metrics_file']
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            # Añadir métricas al archivo
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(self.metrics) + '\n')
                
            logger.debug(f"Métricas almacenadas en {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error al almacenar métricas: {str(e)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de salud actual del sistema.
        
        Returns:
            Diccionario con información de estado de salud
        """
        # Ejecutar verificaciones si no se han ejecutado recientemente
        if not self.metrics.get('health_status'):
            self.run_health_checks()
        
        return {
            'status': self.metrics.get('health_status', HealthStatus.UNKNOWN.value),
            'timestamp': datetime.datetime.now().isoformat(),
            'checks': {
                name: {
                    'status': check['last_status'],
                    'last_check_time': check['last_check_time'],
                    'description': check['description'],
                    'error': check.get('last_error')
                }
                for name, check in self.health_checks.items()
            },
            'metrics': {
                'cpu_percent': self.metrics.get('cpu_percent'),
                'memory_percent': self.metrics.get('memory', {}).get('percent'),
                'disk_percent': self.metrics.get('disk', {}).get('percent'),
                'scheduler': self.metrics.get('scheduler', {})
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene todas las métricas actuales.
        
        Returns:
            Diccionario con todas las métricas recopiladas
        """
        return self.metrics


# Funciones de utilidad para verificaciones comunes

def check_api_health(url: str, timeout: int = 5) -> bool:
    """
    Verifica la salud de una API mediante una solicitud HTTP.
    
    Args:
        url: URL del endpoint de salud
        timeout: Tiempo máximo de espera en segundos
        
    Returns:
        True si la API responde correctamente, False en caso contrario
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

def check_database_connection(db_connection_func: Callable[[], bool]) -> bool:
    """
    Verifica la conexión a una base de datos.
    
    Args:
        db_connection_func: Función que intenta conectarse a la base de datos
        
    Returns:
        True si la conexión es exitosa, False en caso contrario
    """
    try:
        return db_connection_func()
    except Exception:
        return False

def check_disk_space(min_free_gb: float = 1.0) -> bool:
    """
    Verifica que haya suficiente espacio en disco.
    
    Args:
        min_free_gb: Espacio mínimo libre en GB
        
    Returns:
        True si hay suficiente espacio, False en caso contrario
    """
    try:
        free_gb = psutil.disk_usage('/').free / (1024 * 1024 * 1024)
        return free_gb >= min_free_gb
    except Exception:
        return False

def check_memory_usage(max_percent: float = 90.0) -> bool:
    """
    Verifica que el uso de memoria no sea excesivo.
    
    Args:
        max_percent: Porcentaje máximo de uso de memoria
        
    Returns:
        True si el uso está por debajo del umbral, False en caso contrario
    """
    try:
        memory_percent = psutil.virtual_memory().percent
        return memory_percent <= max_percent
    except Exception:
        return False

def check_cpu_usage(max_percent: float = 90.0) -> bool:
    """
    Verifica que el uso de CPU no sea excesivo.
    
    Args:
        max_percent: Porcentaje máximo de uso de CPU
        
    Returns:
        True si el uso está por debajo del umbral, False en caso contrario
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent <= max_percent
    except Exception:
        return False
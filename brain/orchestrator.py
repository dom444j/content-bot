"""
Orchestrator - Coordinador central del sistema de monetización (Nivel Amazónico)

Este módulo es el núcleo del sistema de monetización, coordinando subsistemas para maximizar ingresos:
- Detección de tendencias y oportunidades en tiempo real
- Creación de contenido optimizado para plataformas
- Verificación de cumplimiento normativo
- Publicación estratégica multiplataforma
- Monetización avanzada con análisis de ROI
- Análisis profundo y optimización continua

Características PRO (según orchestrator.MD):
1. Sistema de prioridades con rebalanceo dinámico y clustering de tareas
2. Persistencia robusta con recuperación de fallos y snapshots
3. Reintentos con backoff exponencial, jitter y circuit breaker
4. Gestión avanzada de shadowbans con simulaciones y planes de recuperación
5. Monitoreo en tiempo real con dashboard interactivo y alertas
6. Seguridad reforzada: encriptación, auditoría, y autenticación multifactor
7. Optimización de recursos: balanceo de carga, caché distribuido, y compresión
8. Soporte completo para OAuth 2.0, JWT, y manejo de límites de tasa
9. Logging estructurado, rotación de logs, y exportación a sistemas externos
10. Integración con APIs externas, webhooks, y soporte para escalabilidad horizontal
11. Secuencia diaria automatizada para 5 canales (MonetizationSystem.md)
12. Dashboard interactivo con métricas en tiempo real
"""

import os
import sys
import logging
import time
import json
import datetime
import threading
import queue
import random
import math
import requests
import signal
import uuid
import psutil
import hashlib
import base64
import hmac
import secrets
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
import functools
import jsonschema
from backoff import on_exception, expo
from requests.exceptions import RequestException, ConnectionError, Timeout
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from dataclasses import dataclass
from enum import Enum
import shutil
import redis
import pickle
import zlib
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import socketio
import schedule
import pandas as pd
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import yaml

# Configuración de directorios
os.makedirs('logs/activity', exist_ok=True)
os.makedirs('logs/audit', exist_ok=True)
os.makedirs('logs/monitoring', exist_ok=True)
os.makedirs('data/cache', exist_ok=True)
os.makedirs('data/snapshots', exist_ok=True)
os.makedirs('config', exist_ok=True)
os.makedirs('security', exist_ok=True)
os.makedirs('dashboards', exist_ok=True)

# Configuración de logger de actividad
activity_logger = logging.getLogger("orchestrator_activity")
activity_logger.setLevel(logging.INFO)
activity_handler = RotatingFileHandler(
    "logs/activity/orchestrator_activity.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=50
)
activity_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s - [Trace: %(pathname)s:%(lineno)d] - [Thread: %(threadName)s]',
    datefmt='%Y-%m-%d %H:%M:%S'
)
activity_handler.setFormatter(activity_formatter)
activity_logger.addHandler(activity_handler)
activity_logger.propagate = False

# Logger de auditoría
audit_logger = logging.getLogger("orchestrator_audit")
audit_logger.setLevel(logging.INFO)
audit_handler = RotatingFileHandler(
    "logs/audit/orchestrator_audit.log",
    maxBytes=5*1024*1024,  # 5MB
    backupCount=30
)
audit_handler.setFormatter(activity_formatter)
audit_logger.addHandler(audit_handler)
audit_logger.propagate = False

# Logger de monitoreo
monitoring_logger = logging.getLogger("orchestrator_monitoring")
monitoring_logger.setLevel(logging.INFO)
monitoring_handler = RotatingFileHandler(
    "logs/monitoring/orchestrator_monitoring.log",
    maxBytes=5*1024*1024,  # 5MB
    backupCount=20
)
monitoring_handler.setFormatter(activity_formatter)
monitoring_logger.addHandler(monitoring_handler)
monitoring_logger.propagate = False

# Logger principal
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [Trace: %(pathname)s:%(lineno)d] - [Thread: %(threadName)s]',
    handlers=[
        RotatingFileHandler(
            'logs/orchestrator.log',
            maxBytes=20*1024*1024,
            backupCount=50
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('Orchestrator')

# Métricas de Prometheus
task_counter = Counter('orchestrator_tasks_total', 'Total tasks processed', ['type', 'status'])
task_duration = Histogram('orchestrator_task_duration_seconds', 'Task execution duration', ['type'])
resource_gauge = Gauge('orchestrator_resource_usage', 'Resource usage metrics', ['resource_type'])
publication_counter = Counter('orchestrator_publications_total', 'Total publications', ['platform', 'status'])
shadowban_counter = Counter('orchestrator_shadowbans_total', 'Total shadowbans detected', ['platform'])

# Excepciones personalizadas
class OrchestratorError(Exception):
    """Excepción base para errores del orquestador"""
    def __init__(self, message: str, code: Optional[str] = None):
        self.message = message
        self.code = code or "ORC_GENERAL"
        super().__init__(self.message)

class ConfigurationError(OrchestratorError):
    """Error en la configuración del sistema"""
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")

class PlatformError(OrchestratorError):
    """Error en operaciones con plataformas"""
    def __init__(self, message: str):
        super().__init__(message, "PLATFORM_ERROR")

class ShadowbanError(OrchestratorError):
    """Error relacionado con shadowbans"""
    def __init__(self, message: str):
        super().__init__(message, "SHADOWBAN_ERROR")

class ResourceLimitError(OrchestratorError):
    """Error por límites de recursos"""
    def __init__(self, message: str):
        super().__init__(message, "RESOURCE_LIMIT")

class SecurityError(OrchestratorError):
    """Error de seguridad"""
    def __init__(self, message: str):
        super().__init__(message, "SECURITY_ERROR")

class ComplianceError(OrchestratorError):
    """Error de cumplimiento normativo"""
    def __init__(self, message: str):
        super().__init__(message, "COMPLIANCE_ERROR")

# Enums para estados
class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    MAINTENANCE = 4

class ChannelStatus(Enum):
    ACTIVE = "active"
    SHADOWBANNED = "shadowbanned"
    RESTRICTED = "restricted"
    PAUSED = "paused"
    RECOVERING = "recovering"
    ARCHIVED = "archived"
    SUSPENDED = "suspended"

class TaskType(Enum):
    CONTENT_CREATION = "content_creation"
    CHANNEL_OPTIMIZATION = "channel_optimization"
    SHADOWBAN_RECOVERY = "shadowban_recovery"
    ANALYSIS = "analysis"
    MONETIZATION = "monetization"
    PUBLICATION = "publication"
    AUDIT = "audit"
    MAINTENANCE = "maintenance"

class TaskStatus(Enum):
    INITIATED = "initiated"
    PROCESSING = "processing"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

# Decoradores avanzados
def retry_with_backoff(max_tries=5, backoff_factor=2, exceptions=(Exception,), circuit_breaker_threshold=10):
    """Decorador para reintentos con backoff exponencial, jitter y circuit breaker"""
    failure_count = 0

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count
            if failure_count >= circuit_breaker_threshold:
                raise OrchestratorError(f"Circuit breaker abierto para {func.__name__}")
                
            attempt = 0
            while attempt < max_tries:
                try:
                    result = func(*args, **kwargs)
                    failure_count = 0  # Reset circuit breaker
                    return result
                except exceptions as e:
                    failure_count += 1
                    attempt += 1
                    if attempt == max_tries:
                        logger.error(f"Fallo en {func.__name__} tras {max_tries} intentos: {str(e)}")
                        raise
                    wait_time = backoff_factor ** attempt
                    jitter = random.uniform(0, 0.1 * wait_time)
                    wait_time += jitter
                    logger.warning(f"Reintento {attempt}/{max_tries} en {func.__name__}: {str(e)}. Esperando {wait_time:.2f}s")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

def audit_access(func):
    """Decorador para auditoría de operaciones sensibles"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        operation = f"{func.__name__} called with args: {args}, kwargs: {kwargs}"
        audit_logger.info(f"Acceso: {operation}")
        return func(*args, **kwargs)
    return wrapper

def require_authentication(func):
    """Decorador para autenticación multifactor"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._verify_authentication():
            raise SecurityError("Autenticación multifactor requerida")
        return func(self, *args, **kwargs)
    return wrapper

def monitor_performance(func):
    """Decorador para monitorear rendimiento de funciones"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            task_duration.labels(type=func.__name__).observe(duration)
            monitoring_logger.info(f"Ejecución de {func.__name__}: {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            task_duration.labels(type=func.__name__).observe(duration)
            monitoring_logger.error(f"Error en {func.__name__}: {str(e)}, duración: {duration:.2f}s")
            raise
    return wrapper

@dataclass
class Task:
    id: str
    type: TaskType
    channel_id: str
    priority: Priority
    status: TaskStatus
    created_at: str
    steps: Dict[str, Dict]
    metadata: Optional[Dict] = None
    retries: int = 0
    error: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: List[str] = None
    cluster_id: Optional[str] = None
    execution_history: List[Dict] = None

@dataclass
class Channel:
    id: str
    name: str
    niche: str
    platforms: List[str]
    character: str
    created_at: str
    status: ChannelStatus
    stats: Dict[str, Any]
    oauth_tokens: Dict[str, Dict]
    performance_metrics: Dict[str, Any]

class SecurityManager:
    """Gestor de seguridad para encriptación y autenticación"""
    def __init__(self):
        self.key_file = 'security/encryption_key.key'
        self.encryption_key = self._load_or_generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.jwt_secret = secrets.token_hex(32)
        self.mfa_secret = secrets.token_hex(16)
        self.audit_retention_days = 90

    def _load_or_generate_key(self) -> bytes:
        """Carga o genera clave de encriptación"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        audit_logger.info("Clave de encriptación generada")
        return key

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encripta datos sensibles"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return self.fernet.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Desencripta datos"""
        try:
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            raise SecurityError(f"Error al desencriptar datos: {str(e)}")

    def generate_jwt(self, payload: Dict) -> str:
        """Genera token JWT"""
        payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=3600)
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_jwt(self, token: str) -> Dict:
        """Verifica token JWT"""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Token JWT inválido: {str(e)}")

    def generate_mfa_code(self) -> str:
        """Genera código MFA"""
        return secrets.token_hex(4)

    def verify_mfa_code(self, code: str) -> bool:
        """Verifica código MFA"""
        return hmac.compare_digest(code.encode(), self.mfa_secret.encode())

    def rotate_keys(self) -> None:
        """Rota claves de encriptación"""
        old_key = self.encryption_key
        new_key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(new_key)
        self.encryption_key = new_key
        self.fernet = Fernet(new_key)
        audit_logger.info("Claves de encriptación rotadas")
        # Reencriptar datos sensibles
        self._reencrypt_sensitive_data(old_key, new_key)

    def _reencrypt_sensitive_data(self, old_key: bytes, new_key: bytes) -> None:
        """Reencripta datos sensibles tras rotación de claves"""
        old_fernet = Fernet(old_key)
        new_fernet = Fernet(new_key)
        config_path = os.path.join('config', 'platforms.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for platform, platform_config in config.items():
            for field in ['access_token', 'refresh_token', 'client_key', 'client_secret']:
                if platform_config.get(field):
                    try:
                        decrypted = old_fernet.decrypt(base64.b64decode(platform_config[field])).decode('utf-8')
                        reencrypted = base64.b64encode(new_fernet.encrypt(decrypted.encode('utf-8'))).decode('utf-8')
                        platform_config[field] = reencrypted
                    except Exception as e:
                        logger.error(f"Error al reencriptar {field} para {platform}: {str(e)}")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        audit_logger.info("Datos sensibles reencriptados tras rotación de claves")

class DashboardManager:
    """Gestor de dashboard interactivo para monitoreo en tiempo real"""
    def __init__(self):
        self.sio = socketio.Client()
        self.connected = False
        self.dashboard_data = {
            'system_metrics': {},
            'channel_metrics': {},
            'task_metrics': {},
            'publication_metrics': {},
            'shadowban_status': {}
        }
        self._init_dashboard_server()

    def _init_dashboard_server(self):
        """Inicializa servidor Socket.IO para dashboard"""
        try:
            self.sio.connect('http://localhost:5000')
            self.connected = True
            logger.info("Conectado al servidor de dashboard Socket.IO")
        except Exception as e:
            logger.error(f"Error al conectar con servidor de dashboard: {str(e)}")
            self.connected = False

    def update_dashboard(self, data: Dict):
        """Actualiza datos del dashboard"""
        if not self.connected:
            self._init_dashboard_server()
        if self.connected:
            try:
                self.dashboard_data.update(data)
                self.sio.emit('update_dashboard', self.dashboard_data)
                monitoring_logger.info("Dashboard actualizado con nuevos datos")
            except Exception as e:
                logger.error(f"Error al actualizar dashboard: {str(e)}")
                self.connected = False

    def generate_html_dashboard(self, output_path: str = 'dashboards/index.html'):
        """Genera dashboard HTML estático"""
        try:
            dashboard_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'system_metrics': self.dashboard_data['system_metrics'],
                'channel_metrics': self.dashboard_data['channel_metrics'],
                'task_metrics': self.dashboard_data['task_metrics']
            }
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Orchestrator Dashboard</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto p-4">
                    <h1 class="text-3xl font-bold mb-4">Orchestrator Dashboard</h1>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white p-4 rounded shadow">
                            <h2 class="text-xl font-semibold">System Metrics</h2>
                            <canvas id="systemChart"></canvas>
                        </div>
                        <div class="bg-white p-4 rounded shadow">
                            <h2 class="text-xl font-semibold">Channel Metrics</h2>
                            <canvas id="channelChart"></canvas>
                        </div>
                    </div>
                </div>
                <script>
                    const systemCtx = document.getElementById('systemChart').getContext('2d');
                    new Chart(systemCtx, {{
                        type: 'line',
                        data: {{
                            labels: ['CPU', 'Memory', 'Disk'],
                            datasets: [{{
                                label: 'Usage %',
                                data: [{json.dumps(dashboard_data['system_metrics'].get('cpu_usage', 0))},
                                       {json.dumps(dashboard_data['system_metrics'].get('memory_usage', 0))},
                                       {json.dumps(dashboard_data['system_metrics'].get('disk_usage', 0))}],
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }}]
                        }}
                    }});
                    const channelCtx = document.getElementById('channelChart').getContext('2d');
                    new Chart(channelCtx, {{
                        type: 'bar',
                        data: {{
                            labels: Object.keys({json.dumps(dashboard_data['channel_metrics'])}),
                            datasets: [{{
                                label: 'Views',
                                data: Object.values({json.dumps(dashboard_data['channel_metrics'])}).map(c => c.views || 0),
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }}]
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            monitoring_logger.info(f"Dashboard HTML generado en {output_path}")
        except Exception as e:
            logger.error(f"Error al generar dashboard HTML: {str(e)}")

class Orchestrator:
    """Clase principal que orquesta el sistema de monetización"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Orchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Inicializando Orchestrator Amazónico...")
        self.active = False
        self.security_manager = SecurityManager()
        self.dashboard_manager = DashboardManager()
        self.channels: Dict[str, Channel] = {}
        self.current_tasks: Dict[str, Task] = {}
        self.task_history: List[Task] = []
        self.task_queue = queue.PriorityQueue()
        self.paused_tasks: Dict[str, Task] = {}
        self.shadowban_status: Dict[str, Dict] = {}
        self.channel_status: Dict[str, ChannelStatus] = {}
        self.shadowban_history: Dict[str, List] = {}
        self.recovery_plans: Dict[str, Dict] = {}
        self.task_clusters: Dict[str, List[str]] = defaultdict(list)
        self.resource_metrics: Dict[str, Any] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'active_threads': 0,
            'network_io': 0
        }
        self.rate_limits: Dict[str, Dict] = defaultdict(lambda: {
            'limit': 100,
            'remaining': 100,
            'reset_time': time.time()
        })

        # Configuración
        self.max_retries = 5
        self.base_retry_delay = 2
        self.task_check_interval = 5
        self.max_concurrent_tasks = 20
        self.oauth_platforms = ['youtube', 'tiktok', 'twitch']
        self.config = self._load_config()
        self.snapshot_interval = 3600  # 1 hora
        self.compression_level = 6
        self.daily_sequence_interval = 86400  # 24 horas
        self.max_channels = 5  # Según MonetizationSystem.md

        # Inicializar caché Redis
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.cache_ttl = 7200  # 2 horas

        # Inicializar pool de hilos
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)

        # Métricas de publicación
        self.publication_metrics = {
            'total_attempts': 0,
            'successful_publishes': 0,
            'failed_publishes': 0,
            'retries': 0,
            'by_platform': defaultdict(lambda: {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'retries': 0,
                'rate_limit_hits': 0
            })
        }

        # Métricas avanzadas
        self.performance_metrics = {
            'task_completion_rate': 0.0,
            'average_task_duration': 0.0,
            'channel_roi': {},
            'platform_engagement': defaultdict(dict)
        }

        # Componentes y subsistemas
        self.components: Dict[str, Any] = {
            'decision_engine': None,
            'scheduler': None,
            'notifier': None,
            'knowledge_base': None,
            'analytics_engine': None,
            'shadowban_detector': None,
            'load_balancer': None,
            'webhook_manager': None
        }
        self.subsystems: Dict[str, Dict] = {
            'trends': {'trend_radar': None, 'trend_predictor': None, 'trend_aggregator': None},
            'creation': {'script_factory': None, 'character_engine': None, 'video_composer': None, 'audio_synthesizer': None},
            'compliance': {'content_auditor': None, 'platform_compliance': None},
            'publication': {'youtube_adapter': None, 'tiktok_adapter': None, 'instagram_adapter': None, 'twitch_adapter': None},
            'monetization': {'revenue_optimizer': None, 'affiliate_manager': None, 'ad_optimizer': None},
            'optimization': {'channel_optimizer': None, 'content_optimizer': None},
            'analysis': {'analytics_engine': None, 'competitor_analyzer': None, 'audience_analyzer': None}
        }

        # Base de datos SQLite
        self._init_db()

        # Manejo de señales
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Hilos de mantenimiento
        self.maintenance_tasks: Dict[str, threading.Thread] = {}

        # Scheduler para secuencia diaria
        self.scheduler = schedule.Scheduler()

        # Iniciar servidor Prometheus
        start_http_server(8000)
        logger.info("Servidor Prometheus iniciado en puerto 8000")

        self._initialized = True
        logger.info("Orchestrator Amazónico inicializado")
        activity_logger.info("Inicialización completada")

    def _init_db(self):
        """Inicializa la base de datos SQLite"""
        try:
            with sqlite3.connect('data/orchestrator.db') as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        type TEXT,
                        channel_id TEXT,
                        priority INTEGER,
                        status TEXT,
                        created_at TEXT,
                        steps TEXT,
                        metadata TEXT,
                        retries INTEGER,
                        error TEXT,
                        completed_at TEXT,
                        dependencies TEXT,
                        cluster_id TEXT,
                        execution_history TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS channels (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        niche TEXT,
                        platforms TEXT,
                        character TEXT,
                        created_at TEXT,
                        status TEXT,
                        stats TEXT,
                        oauth_tokens TEXT,
                        performance_metrics TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        metrics TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS snapshots (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        state TEXT,
                        compressed BOOLEAN
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        operation TEXT,
                        user TEXT,
                        details TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        channel_id TEXT,
                        platform TEXT,
                        metrics TEXT
                    )
                """)
                conn.commit()
            logger.info("Base de datos SQLite inicializada")
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {str(e)}")
            raise ConfigurationError(f"Error de base de datos: {str(e)}")

    @audit_access
    def _load_config(self) -> Dict:
        """Carga y valida la configuración del sistema"""
        try:
            config_path = os.path.join('config', 'system_config.yaml')
            if not os.path.exists(config_path):
                self._create_default_config()

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            config_schema = {
                "type": "object",
                "required": ["system", "platforms", "monetization", "creation", "security", "channels"],
                "properties": {
                    "system": {
                        "type": "object",
                        "required": ["max_retries", "retry_delay_base", "task_check_interval", "max_concurrent_tasks", "snapshot_interval"],
                        "properties": {
                            "max_retries": {"type": "integer", "minimum": 1},
                            "retry_delay_base": {"type": "number", "minimum": 0.1},
                            "task_check_interval": {"type": "number", "minimum": 0.1},
                            "max_concurrent_tasks": {"type": "integer", "minimum": 1},
                            "snapshot_interval": {"type": "integer", "minimum": 60}
                        }
                    },
                    "platforms": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {
                            "type": "object",
                            "required": ["enabled", "rate_limit"],
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "api_key": {"type": "string"},
                                "client_id": {"type": "string"},
                                "client_secret": {"type": "string"},
                                "redirect_uri": {"type": "string"},
                                "rate_limit": {"type": "integer", "minimum": 1}
                            }
                        }
                    },
                    "monetization": {
                        "type": "object",
                        "required": ["strategies", "roi_threshold"],
                        "properties": {
                            "strategies": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"}
                            },
                            "roi_threshold": {"type": "number", "minimum": 0}
                        }
                    },
                    "creation": {
                        "type": "object",
                        "required": ["content_types", "max_content_size"],
                        "properties": {
                            "content_types": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"}
                            },
                            "max_content_size": {"type": "integer", "minimum": 1024}
                        }
                    },
                    "security": {
                        "type": "object",
                        "required": ["mfa_enabled", "jwt_expiry"],
                        "properties": {
                            "mfa_enabled": {"type": "boolean"},
                            "jwt_expiry": {"type": "integer", "minimum": 60}
                        }
                    },
                    "channels": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "required": ["id", "name", "niche", "platforms", "character"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "niche": {"type": "string"},
                                "platforms": {"type": "array", "items": {"type": "string"}},
                                "character": {"type": "string"}
                            }
                        }
                    }
                }
            }

            jsonschema.validate(instance=config, schema=config_schema)
            logger.info("Configuración validada")

            # Cargar configuraciones adicionales
            for file in ['platforms.yaml', 'strategy.yaml', 'niches.yaml', 'character_profiles.yaml', 'security.yaml']:
                file_path = os.path.join('config', file)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config[file.replace('.yaml', '')] = yaml.safe_load(f)

            self._validate_platform_configs(config)
            self._encrypt_sensitive_config(config)
            return config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            self._create_default_config()
            return self._load_config()

    def _encrypt_sensitive_config(self, config: Dict) -> None:
        """Encripta datos sensibles en la configuración"""
        for platform, platform_config in config.get('platforms', {}).items():
            sensitive_fields = ['api_key', 'client_id', 'client_secret', 'access_token', 'refresh_token']
            for field in sensitive_fields:
                if field in platform_config and platform_config[field]:
                    encrypted = self.security_manager.encrypt_data(platform_config[field])
                    platform_config[field] = base64.b64encode(encrypted).decode('utf-8')
        audit_logger.info("Configuración sensible encriptada")

    def _decrypt_sensitive_config(self, platform: str, field: str) -> str:
        """Desencripta datos sensibles"""
        try:
            encrypted = base64.b64decode(self.config['platforms'][platform][field])
            return self.security_manager.decrypt_data(encrypted).decode('utf-8')
        except Exception as e:
            raise SecurityError(f"Error al desencriptar {field} para {platform}: {str(e)}")

    def _create_default_config(self):
        """Crea configuraciones por defecto"""
        platforms_config = {
            "youtube": {
                "client_key": "",
                "client_secret": "",
                "access_token": "",
                "refresh_token": "",
                "redirect_uri": "http://localhost:8080",
                "channel_id": "",
                "enabled": True,
                "rate_limit": 10000
            },
            "tiktok": {
                "client_key": "",
                "client_secret": "",
                "access_token": "",
                "refresh_token": "",
                "redirect_uri": "http://localhost:8080",
                "open_id": "",
                "enabled": True,
                "rate_limit": 2000
            },
            "instagram": {
                "api_key": "",
                "account_id": "",
                "enabled": True,
                "rate_limit": 5000
            },
            "twitch": {
                "client_key": "",
                "client_secret": "",
                "access_token": "",
                "refresh_token": "",
                "redirect_uri": "http://localhost:8080",
                "enabled": True,
                "rate_limit": 800
            }
        }

        strategy_config = {
            "cta_timing": {"early": [0, 3], "middle": [4, 8], "end": [-2, 0]},
            "content_types": ["educational", "entertainment", "tutorial", "live"],
            "monetization_priority": ["affiliate", "ads", "products", "sponsorships"],
            "retry_policy": {"max_retries": 5, "base_delay": 2}
        }

        niches_config = {
            "finance": {"keywords": ["crypto", "investing", "money"], "enabled": True, "weight": 0.8},
            "health": {"keywords": ["fitness", "mindfulness", "nutrition"], "enabled": True, "weight": 0.7},
            "technology": {"keywords": ["ai", "gadgets", "programming"], "enabled": True, "weight": 0.9},
            "gaming": {"keywords": ["esports", "streams", "reviews"], "enabled": True, "weight": 0.6}
        }

        characters_config = {
            "finance_expert": {
                "name": "Alex Finance",
                "personality": "professional",
                "voice_type": "male_authoritative",
                "visual_style": "business_casual",
                "cta_style": "educational"
            },
            "fitness_coach": {
                "name": "Fit Coach",
                "personality": "energetic",
                "voice_type": "female_motivational",
                "visual_style": "athletic",
                "cta_style": "motivational"
            },
            "tech_guru": {
                "name": "Tech Titan",
                "personality": "innovative",
                "voice_type": "male_tech",
                "visual_style": "modern_tech",
                "cta_style": "informative"
            }
        }

        security_config = {
            "mfa_enabled": True,
            "jwt_expiry": 3600,
            "encryption_algorithm": "AES",
            "audit_retention_days": 90
        }

        # Configuración de 5 canales según MonetizationSystem.md
        channels_config = [
            {
                "id": f"finance_{uuid.uuid4().hex}",
                "name": "Crypto Wealth Hub",
                "niche": "finance",
                "platforms": ["youtube", "tiktok", "instagram"],
                "character": "finance_expert"
            },
            {
                "id": f"health_{uuid.uuid4().hex}",
                "name": "Fit Life Journey",
                "niche": "health",
                "platforms": ["youtube", "tiktok", "instagram"],
                "character": "fitness_coach"
            },
            {
                "id": f"technology_{uuid.uuid4().hex}",
                "name": "Tech Trends Now",
                "niche": "technology",
                "platforms": ["youtube", "twitch"],
                "character": "tech_guru"
            },
            {
                "id": f"gaming_{uuid.uuid4().hex}",
                "name": "Epic Gaming Zone",
                "niche": "gaming",
                "platforms": ["twitch", "youtube"],
                "character": "tech_guru"
            },
            {
                "id": f"finance2_{uuid.uuid4().hex}",
                "name": "Money Mastery",
                "niche": "finance",
                "platforms": ["youtube", "tiktok"],
                "character": "finance_expert"
            }
        ]

        config_files = {
            'system_config.yaml': {
                "system": {
                    "max_retries": 5,
                    "retry_delay_base": 2,
                    "task_check_interval": 5,
                    "max_concurrent_tasks": 20,
                    "snapshot_interval": 3600
                },
                "platforms": platforms_config,
                "monetization": {
                    "strategies": ["ads", "affiliate", "products", "sponsorships"],
                    "roi_threshold": 0.2
                },
                "creation": {
                    "content_types": ["video", "short", "live", "stream"],
                    "max_content_size": 1024*1024*100  # 100MB
                },
                "security": security_config,
                "channels": channels_config
            },
            'platforms.yaml': platforms_config,
            'strategy.yaml': strategy_config,
            'niches.yaml': niches_config,
            'character_profiles.yaml': characters_config,
            'security.yaml': security_config
        }

        for file, content in config_files.items():
            with open(os.path.join('config', file), 'w', encoding='utf-8') as f:
                yaml.dump(content, f, allow_unicode=True)
        logger.info("Configuraciones por defecto creadas")
        activity_logger.info("Configuraciones por defecto generadas")

    @audit_access
    def _validate_platform_configs(self, config: Dict) -> None:
        """Valida configuraciones de plataformas"""
        for platform, platform_config in config.get("platforms", {}).items():
            if platform_config.get("enabled", False):
                if platform in self.oauth_platforms:
                    required_fields = ["client_id", "client_secret", "redirect_uri", "rate_limit"]
                    missing = [f for f in required_fields if not platform_config.get(f)]
                    if missing:
                        logger.warning(f"Plataforma {platform} deshabilitada: faltan {missing}")
                        platform_config["enabled"] = False
                elif not platform_config.get("api_key") or not platform_config.get("rate_limit"):
                    logger.warning(f"Plataforma {platform} deshabilitada: falta api_key o rate_limit")
                    platform_config["enabled"] = False

    def _verify_authentication(self) -> bool:
        """Verifica autenticación multifactor"""
        if not self.config['security'].get('mfa_enabled', False):
            return True
        mfa_code = self.security_manager.generate_mfa_code()
        logger.info(f"Código MFA generado: {mfa_code}")
        # En producción, enviar mfa_code al usuario (email/SMS)
        user_input = input("Ingrese código MFA: ")  # Simulación
        return self.security_manager.verify_mfa_code(user_input)

    @retry_with_backoff(max_tries=3)
    @audit_access
    @require_authentication
    def _initialize_oauth_platform(self, platform: str) -> bool:
        """Inicializa plataformas con OAuth 2.0"""
        platform_config = self.config['platforms'].get(platform, {})
        adapter = self._load_subsystem('publication', f'{platform}_adapter')
        if not adapter:
            raise PlatformError(f"Adaptador para {platform} no disponible")

        access_token = self._decrypt_sensitive_config(platform, 'access_token') if platform_config.get('access_token') else None
        refresh_token = self._decrypt_sensitive_config(platform, 'refresh_token') if platform_config.get('refresh_token') else None
        token_expiry = platform_config.get('token_expiry')

        if access_token and refresh_token and token_expiry:
            expiry_time = datetime.datetime.fromisoformat(token_expiry)
            if expiry_time > datetime.datetime.now() + datetime.timedelta(minutes=5):
                logger.info(f"Tokens válidos para {platform}")
                return True
            return self._refresh_platform_token(platform)

        client_key = self._decrypt_sensitive_config(platform, 'client_key')
        client_secret = self._decrypt_sensitive_config(platform, 'client_secret')
        redirect_uri = platform_config.get('redirect_uri')

        if not all([client_key, client_secret, redirect_uri]):
            raise ConfigurationError(f"Credenciales incompletas para {platform}")

        auth_url = adapter.get_authorization_url(client_key, redirect_uri)
        logger.info(f"URL de autorización para {platform}: {auth_url}")
        auth_code = input(f"Ingresa el código de autorización para {platform}: ")

        token_response = adapter.exchange_code_for_tokens(
            auth_code=auth_code,
            client_key=client_key,
            client_secret=client_secret,
            redirect_uri=redirect_uri
        )

        if token_response.get('status') != 'success':
            raise PlatformError(f"Error al obtener tokens para {platform}: {token_response.get('message')}")

        new_credentials = {
            'access_token': base64.b64encode(self.security_manager.encrypt_data(token_response['access_token'])).decode('utf-8'),
            'refresh_token': base64.b64encode(self.security_manager.encrypt_data(token_response['refresh_token'])).decode('utf-8'),
            'token_expiry': (datetime.datetime.now() + datetime.timedelta(seconds=token_response['expires_in'])).isoformat()
        }
        self._save_platform_credentials(platform, new_credentials)
        self.config['platforms'][platform].update(new_credentials)
        logger.info(f"Autenticación OAuth completada para {platform}")
        activity_logger.info(f"OAuth completado para {platform}")
        return True

    @retry_with_backoff(max_tries=3)
    @audit_access
    @require_authentication
    def _refresh_platform_token(self, platform: str) -> bool:
        """Refresca tokens de acceso"""
        platform_config = self.config['platforms'].get(platform, {})
        adapter = self._load_subsystem('publication', f'{platform}_adapter')
        if not adapter:
            raise PlatformError(f"Adaptador para {platform} no disponible")

        refresh_token = self._decrypt_sensitive_config(platform, 'refresh_token')
        if not refresh_token:
            raise ConfigurationError(f"No se encontró refresh_token para {platform}")

        token_response = adapter.refresh_access_token(
            refresh_token=refresh_token,
            client_key=self._decrypt_sensitive_config(platform, 'client_key'),
            client_secret=self._decrypt_sensitive_config(platform, 'client_secret')
        )

        if token_response.get('status') != 'success':
            raise PlatformError(f"Error al refrescar token para {platform}: {token_response.get('message')}")

        new_credentials = {
            'access_token': base64.b64encode(self.security_manager.encrypt_data(token_response['access_token'])).decode('utf-8'),
            'refresh_token': base64.b64encode(self.security_manager.encrypt_data(token_response.get('refresh_token', refresh_token))).decode('utf-8'),
            'token_expiry': (datetime.datetime.now() + datetime.timedelta(seconds=token_response['expires_in'])).isoformat()
        }
        self._save_platform_credentials(platform, new_credentials)
        self.config['platforms'][platform].update(new_credentials)
        logger.info(f"Token refrescado para {platform}")
        activity_logger.info(f"Token refrescado para {platform}")
        return True

    @audit_access
    def _save_platform_credentials(self, platform: str, credentials: Dict) -> None:
        """Guarda credenciales de plataforma"""
        try:
            config_file = os.path.join('config', 'platforms.yaml')
            with open(config_file, 'r', encoding='utf-8') as f:
                all_configs = yaml.safe_load(f)
            all_configs[platform].update(credentials)
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(all_configs, f, allow_unicode=True)
            audit_logger.info(f"Credenciales actualizadas para {platform}")
        except Exception as e:
            logger.error(f"Error al guardar credenciales para {platform}: {str(e)}")
            raise ConfigurationError(f"Error al guardar credenciales: {str(e)}")

    def _verify_config(self) -> bool:
        """Verifica la validez de la configuración"""
        platforms_enabled = any(p.get('enabled', False) for p in self.config.get('platforms', {}).values())
        niches_enabled = any(n.get('enabled', False) for n in self.config.get('niches', {}).values())
        security_enabled = self.config.get('security', {}).get('mfa_enabled', False)
        channels_configured = len(self.config.get('channels', [])) >= self.max_channels
        if not (platforms_enabled and niches_enabled and security_enabled and channels_configured):
            logger.error("Configuración inválida: plataformas, nichos, seguridad o canales insuficientes")
            return False
        return True

    def _load_component(self, component_name: str) -> Any:
        """Carga componentes dinámicamente con caché"""
        cache_key = f"component:{component_name}"
        if self.redis_client.exists(cache_key):
            return pickle.loads(zlib.decompress(self.redis_client.get(cache_key)))

        try:
            if component_name == 'decision_engine':
                from brain.decision_engine import DecisionEngine
                component = DecisionEngine()
            elif component_name == 'scheduler':
                from brain.scheduler import Scheduler
                component = Scheduler()
            elif component_name == 'notifier':
                from brain.notifier import Notifier
                component = Notifier()
            elif component_name == 'knowledge_base':
                from data.knowledge_base import KnowledgeBase
                component = KnowledgeBase()
            elif component_name == 'analytics_engine':
                from data.analytics_engine import AnalyticsEngine
                component = AnalyticsEngine()
            elif component_name == 'shadowban_detector':
                from compliance.shadowban_detector import ShadowbanDetector
                component = ShadowbanDetector()
            elif component_name == 'load_balancer':
                from system.load_balancer import LoadBalancer
                component = LoadBalancer()
            elif component_name == 'webhook_manager':
                from system.webhook_manager import WebhookManager
                component = WebhookManager()
            else:
                raise ConfigurationError(f"Componente desconocido: {component_name}")

            self.components[component_name] = component
            compressed = zlib.compress(pickle.dumps(component), level=self.compression_level)
            self.redis_client.setex(cache_key, self.cache_ttl, compressed)
            logger.info(f"Componente {component_name} cargado")
            return component
        except ImportError as e:
            logger.error(f"Error al cargar componente {component_name}: {str(e)}")
            return None

    def _load_subsystem(self, subsystem: str, module: str) -> Any:
        """Carga subsistemas dinámicamente con caché"""
        cache_key = f"subsystem:{subsystem}:{module}"
        if self.redis_client.exists(cache_key):
            return pickle.loads(zlib.decompress(self.redis_client.get(cache_key)))

        try:
            if subsystem == 'trends':
                if module == 'trend_radar':
                    from trends.trend_radar import TrendRadar
                    component = TrendRadar()
                elif module == 'trend_predictor':
                    from trends.trend_predictor import TrendPredictor
                    component = TrendPredictor()
                elif module == 'trend_aggregator':
                    from trends.trend_aggregator import TrendAggregator
                    component = TrendAggregator()
            elif subsystem == 'creation':
                if module == 'script_factory':
                    from creation.script_factory import ScriptFactory
                    component = ScriptFactory()
                elif module == 'character_engine':
                    from creation.character_engine import CharacterEngine
                    component = CharacterEngine()
                elif module == 'video_composer':
                    from creation.video_composer import VideoComposer
                    component = VideoComposer()
                elif module == 'audio_synthesizer':
                    from creation.audio_synthesizer import AudioSynthesizer
                    component = AudioSynthesizer()
            elif subsystem == 'compliance':
                if module == 'content_auditor':
                    from compliance.content_auditor import ContentAuditor
                    component = ContentAuditor()
                elif module == 'platform_compliance':
                    from compliance.platform_compliance import PlatformCompliance
                    component = PlatformCompliance()
            elif subsystem == 'publication':
                if module in ['youtube_adapter', 'tiktok_adapter', 'instagram_adapter', 'twitch_adapter']:
                    from platform_adapters import api_router
                    component = getattr(api_router, module)()
                    platform_name = module.replace('_adapter', '')
                    if platform_name in self.oauth_platforms:
                        self._initialize_oauth_platform(platform_name)
            elif subsystem == 'monetization':
                if module == 'revenue_optimizer':
                    from monetization.revenue_optimizer import RevenueOptimizer
                    component = RevenueOptimizer()
                elif module == 'affiliate_manager':
                    from monetization.affiliate_manager import AffiliateManager
                    component = AffiliateManager()
                elif module == 'ad_optimizer':
                    from monetization.ad_optimizer import AdOptimizer
                    component = AdOptimizer()
            elif subsystem == 'optimization':
                if module == 'channel_optimizer':
                    from optimization.channel_optimizer import ChannelOptimizer
                    component = ChannelOptimizer()
                elif module == 'content_optimizer':
                    from optimization.content_optimizer import ContentOptimizer
                    component = ContentOptimizer()
            elif subsystem == 'analysis':
                if module == 'analytics_engine':
                    from analysis.analytics_engine import AnalyticsEngine
                    component = AnalyticsEngine()
                elif module == 'competitor_analyzer':
                    from analysis.competitor_analyzer import CompetitorAnalyzer
                    component = CompetitorAnalyzer()
                elif module == 'audience_analyzer':
                    from analysis.audience_analyzer import AudienceAnalyzer
                    component = AudienceAnalyzer()
            else:
                raise ConfigurationError(f"Subsistema desconocido: {subsystem}/{module}")

            self.subsystems[subsystem][module] = component
            compressed = zlib.compress(pickle.dumps(component), level=self.compression_level)
            self.redis_client.setex(cache_key, self.cache_ttl, compressed)
            logger.info(f"Subsistema {subsystem}/{module} cargado")
            return component
        except ImportError as e:
            logger.error(f"Error al cargar subsistema {subsystem}/{module}: {str(e)}")
            return None

    @require_authentication
    def start(self) -> bool:
        """Inicia el orquestador y sus subsistemas"""
        if self.active:
            logger.warning("Orchestrator ya está activo")
            return True

        logger.info("Iniciando Orchestrator Amazónico...")
        try:
            # Cargar componentes críticos
            critical_components = [
                'decision_engine', 'scheduler', 'notifier', 'knowledge_base',
                'analytics_engine', 'shadowban_detector', 'load_balancer', 'webhook_manager'
            ]
            for comp in critical_components:
                if not self._load_component(comp):
                    raise ConfigurationError(f"Componente crítico {comp} no disponible")

            # Verificar configuración
            if not self._verify_config():
                raise ConfigurationError("Configuración inválida")

            # Inicializar canales
            self._initialize_channels()

            # Cargar tareas pendientes
            self._load_pending_tasks()

            # Iniciar tareas de mantenimiento
            self._start_maintenance_tasks()

            # Iniciar monitoreo
            self.maintenance_tasks['resource_monitor'] = threading.Thread(target=self._monitor_resources, daemon=True)
            self.maintenance_tasks['snapshot_manager'] = threading.Thread(target=self._manage_snapshots, daemon=True)
            self.maintenance_tasks['dashboard_updater'] = threading.Thread(target=self._dashboard_updater, daemon=True)
            self.maintenance_tasks['recovery_monitor'] = threading.Thread(target=self._recovery_monitor, daemon=True)
            self.maintenance_tasks['daily_sequence'] = threading.Thread(target=self._run_daily_sequence, daemon=True)
            for task in self.maintenance_tasks.values():
                task.start()

            # Iniciar bucle principal
            self.active = True
            self.maintenance_tasks['main_loop'] = threading.Thread(target=self._main_loop, daemon=True)
            self.maintenance_tasks['priority_handler'] = threading.Thread(target=self._priority_handler, daemon=True)
            self.maintenance_tasks['main_loop'].start()
            self.maintenance_tasks['priority_handler'].start()

            # Programar secuencia diaria
            self.scheduler.every().day.at("00:00").do(self._execute_daily_sequence)
            self.maintenance_tasks['scheduler'] = threading.Thread(target=self._run_scheduler, daemon=True)
            self.maintenance_tasks['scheduler'].start()

            logger.info("Orchestrator Amazónico iniciado correctamente")
            activity_logger.info("Orchestrator iniciado")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar Orchestrator: {str(e)}")
            return False

    def _start_maintenance_tasks(self):
        """Inicia tareas de mantenimiento"""
        maintenance_tasks = [
            ('cache_cleanup', self._cleanup_cache, 3600),
            ('log_rotation', self._rotate_logs, 86400),
            ('metrics_aggregation', self._aggregate_metrics, 7200),
            ('rate_limit_monitor', self._monitor_rate_limits, 300),
            ('key_rotation', self.security_manager.rotate_keys, 604800),  # Cada 7 días
            ('performance_analysis', self._run_performance_analysis, 7200)
        ]
        for task_name, task_func, interval in maintenance_tasks:
            self.maintenance_tasks[task_name] = threading.Thread(
                target=self._run_maintenance_task,
                args=(task_func, interval),
                daemon=True
            )

    def _run_maintenance_task(self, task_func: Callable, interval: int):
        """Ejecuta tarea de mantenimiento periódica"""
        while self.active:
            try:
                task_func()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error en tarea de mantenimiento {task_func.__name__}: {str(e)}")
                time.sleep(60)

    def _run_scheduler(self):
        """Ejecuta el scheduler para tareas programadas"""
        while self.active:
            try:
                self.scheduler.run_pending()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error en scheduler: {str(e)}")
                time.sleep(60)

    def stop(self) -> bool:
        """Detiene el orquestador de forma segura"""
        if not self.active:
            logger.warning("Orchestrator ya está detenido")
            return True

        logger.info("Deteniendo Orchestrator Amazónico...")
        try:
            self.active = False
            self._save_snapshot()
            for task_id, task in self.current_tasks.items():
                task.status = TaskStatus.PAUSED
                self.paused_tasks[task_id] = task
                self._persist_task(task)
                logger.info(f"Tarea {task_id} pausada")

            while not self.task_queue.empty():
                self.task_queue.get()

            self.executor.shutdown(wait=True)
            self.redis_client.close()
            self.dashboard_manager.sio.disconnect()
            for task_name, thread in self.maintenance_tasks.items():
                logger.info(f"Deteniendo tarea de mantenimiento: {task_name}")
            self.scheduler.clear()
            logger.info("Orchestrator Amazónico detenido correctamente")
            activity_logger.info("Orchestrator detenido")
            return True
        except Exception as e:
            logger.error(f"Error al detener Orchestrator: {str(e)}")
            return False

    def _handle_signal(self, signum, frame):
        """Maneja señales del sistema"""
        logger.info(f"Señal recibida: {signum}. Deteniendo Orchestrator...")
        self.stop()
        sys.exit(0)

    def _initialize_channels(self):
        """Inicializa canales desde la configuración"""
        for channel_config in self.config.get('channels', []):
            channel_id = channel_config['id']
            channel = Channel(
                id=channel_id,
                name=channel_config['name'],
                niche=channel_config['niche'],
                platforms=channel_config['platforms'],
                character=channel_config['character'],
                created_at=datetime.datetime.now().isoformat(),
                status=ChannelStatus.ACTIVE,
                stats={'videos': 0, 'subscribers': 0, 'views': 0, 'revenue': 0},
                oauth_tokens={},
                performance_metrics={}
            )
            self.channels[channel_id] = channel
            self.channel_status[channel_id] = ChannelStatus.ACTIVE
            knowledge_base = self._load_component('knowledge_base')
            if knowledge_base:
                knowledge_base.save_channel(channel.__dict__)
        logger.info(f"Inicializados {len(self.channels)} canales")

    def _create_default_channels(self):
        """Crea canales predeterminados si no existen"""
        if len(self.channels) >= self.max_channels:
            return
        for niche_name, niche_data in self.config.get('niches', {}).items():
            if niche_data.get('enabled', False) and len(self.channels) < self.max_channels:
                channel_id = f"{niche_name}_{uuid.uuid4().hex}"
                platforms = [p for p, d in self.config.get('platforms', {}).items() if d.get('enabled', False)]
                character = next((c for c in self.config.get('character_profiles', {}).keys() if niche_name in c), None)
                if not character and self.config.get('character_profiles'):
                    character = list(self.config['character_profiles'].keys())[0]

                channel = Channel(
                    id=channel_id,
                    name=f"{niche_name.capitalize()} Channel",
                    niche=niche_name,
                    platforms=platforms,
                    character=character,
                    created_at=datetime.datetime.now().isoformat(),
                    status=ChannelStatus.ACTIVE,
                    stats={'videos': 0, 'subscribers': 0, 'views': 0, 'revenue': 0},
                    oauth_tokens={},
                    performance_metrics={}
                )
                self.channels[channel_id] = channel
                self.channel_status[channel_id] = ChannelStatus.ACTIVE
                knowledge_base = self._load_component('knowledge_base')
                if knowledge_base:
                    knowledge_base.save_channel(channel.__dict__)
                logger.info(f"Canal creado: {channel.name}")

    def _load_pending_tasks(self):
        """Carga tareas pendientes desde la base de datos"""
        try:
            with sqlite3.connect('data/orchestrator.db') as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tasks WHERE status IN ('initiated', 'processing', 'paused')")
                for row in cursor.fetchall():
                    task = Task(
                        id=row[0],
                        type=TaskType(row[1]),
                        channel_id=row[2],
                        priority=Priority(row[3]),
                        status=TaskStatus(row[4]),
                        created_at=row[5],
                        steps=json.loads(row[6]),
                        metadata=json.loads(row[7]) if row[7] else None,
                        retries=row[8],
                        error=row[9],
                        completed_at=row[10],
                        dependencies=json.loads(row[11]) if row[11] else [],
                        cluster_id=row[12],
                        execution_history=json.loads(row[13]) if row[13] else []
                    )
                    self.task_queue.put((task.priority.value, task))
                    self.current_tasks[task.id] = task
                    if task.cluster_id:
                        self.task_clusters[task.cluster_id].append(task.id)
                    logger.info(f"Tarea pendiente cargada: {task.id}")
                    task_counter.labels(type=task.type.value, status=task.status.value).inc()
        except Exception as e:
            logger.error(f"Error al cargar tareas pendientes: {str(e)}")

    def _persist_task(self, task: Task):
        """Persiste una tarea en la base de datos"""
        try:
            with sqlite3.connect('data/orchestrator.db') as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO tasks (
                        id, type, channel_id, priority, status, created_at,
                        steps, metadata, retries, error, completed_at,
                        dependencies, cluster_id, execution_history
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id,
                    task.type.value,
                    task.channel_id,
                    task.priority.value,
                    task.status.value,
                    task.created_at,
                    json.dumps(task.steps),
                    json.dumps(task.metadata) if task.metadata else None,
                    task.retries,
                    task.error,
                    task.completed_at,
                    json.dumps(task.dependencies) if task.dependencies else None,
                    task.cluster_id,
                    json.dumps(task.execution_history) if task.execution_history else None
                ))
                conn.commit()
            logger.info(f"Tarea persistida: {task.id}")
            task_counter.labels(type=task.type.value, status=task.status.value).inc()
        except Exception as e:
            logger.error(f"Error al persistir tarea {task.id}: {str(e)}")

    def _save_snapshot(self):
        """Guarda un snapshot del estado del sistema"""
        try:
            state = {
                'channels': {k: v.__dict__ for k, v in self.channels.items()},
                'current_tasks': {k: v.__dict__ for k, v in self.current_tasks.items()},
                'paused_tasks': {k: v.__dict__ for k, v in self.paused_tasks.items()},
                'task_clusters': self.task_clusters,
                'shadowban_status': self.shadowban_status,
                'recovery_plans': self.recovery_plans,
                'resource_metrics': self.resource_metrics,
                'publication_metrics': self.publication_metrics,
                'performance_metrics': self.performance_metrics
            }
            compressed_state = zlib.compress(json.dumps(state).encode('utf-8'), level=self.compression_level)
            snapshot_id = f"snapshot_{uuid.uuid4().hex}"
            with sqlite3.connect('data/orchestrator.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO snapshots (id, timestamp, state, compressed) VALUES (?, ?, ?, ?)",
                    (snapshot_id, datetime.datetime.now().isoformat(), compressed_state, True)
                )
                conn.commit()
            with open(f'data/snapshots/{snapshot_id}.bin', 'wb') as f:
                f.write(compressed_state)
            logger.info(f"Snapshot guardado: {snapshot_id}")
        except Exception as e:
            logger.error(f"Error al guardar snapshot: {str(e)}")

    def _manage_snapshots(self):
        """Gestiona snapshots periódicos"""
        while self.active:
            try:
                self._save_snapshot()
                snapshots_dir = 'data/snapshots'
                max_snapshots = 100
                snapshots = sorted(os.listdir(snapshots_dir), key=lambda x: os.path.getctime(os.path.join(snapshots_dir, x)))
                while len(snapshots) > max_snapshots:
                    os.remove(os.path.join(snapshots_dir, snapshots.pop(0)))
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Error en gestión de snapshots: {str(e)}")
                time.sleep(300)

    @monitor_performance
    def add_task(self, task_data: Dict, priority: Priority, dependencies: List[str] = None, cluster_id: Optional[str] = None) -> Task:
        """Añade una tarea a la cola de prioridades"""
        task_id = task_data.get('id', f"task_{uuid.uuid4().hex}")
        task = Task(
            id=task_id,
            type=TaskType(task_data['type']),
            channel_id=task_data['channel_id'],
            priority=priority,
            status=TaskStatus.INITIATED,
            created_at=datetime.datetime.now().isoformat(),
            steps=task_data.get('steps', {}),
            metadata=task_data.get('metadata'),
            dependencies=dependencies or [],
            cluster_id=cluster_id,
            execution_history=[{'event': 'created', 'timestamp': datetime.datetime.now().isoformat()}]
        )
        self.current_tasks[task.id] = task
        self.task_queue.put((priority.value, task))
        if cluster_id:
            self.task_clusters[cluster_id].append(task.id)
        self._persist_task(task)
        logger.info(f"Tarea añadida: {task.id} con prioridad {priority}")
        activity_logger.info(f"Tarea {task.id} creada: tipo={task.type}, canal={task.channel_id}")
        task_counter.labels(type=task.type.value, status=task.status.value).inc()
        return task

    @require_authentication
    @monitor_performance
    def create_content(self, channel_id: str, content_type: str = None, topic: str = None, priority: Priority = Priority.NORMAL) -> Dict:
        """Crea contenido optimizado para un canal"""
        if not self.active:
            raise OrchestratorError("Orchestrator no está activo")
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")
        if self.channel_status.get(channel_id) in [ChannelStatus.SHADOWBANNED, ChannelStatus.RECOVERING]:
            raise ShadowbanError(f"Canal {channel_id} en estado {self.channel_status[channel_id]}")

        activity_logger.info(f"Creando contenido para canal {channel_id}: tipo={content_type}, tema={topic}")
        task_id = f"content_{channel_id}_{uuid.uuid4().hex}"
        cluster_id = f"cluster_content_{channel_id}_{uuid.uuid4().hex}"
        channel = self.channels[channel_id]

        if not content_type:
            decision_engine = self._load_component('decision_engine')
            content_type = decision_engine.decide_content_type(channel.__dict__) if decision_engine else 'video'

        # Validar tamaño de contenido
        max_size = self.config['creation']['max_content_size']
        if topic and len(topic.encode('utf-8')) > max_size:
            raise OrchestratorError(f"El tema excede el tamaño máximo permitido: {max_size} bytes")

        task_data = {
            'id': task_id,
            'type': TaskType.CONTENT_CREATION.value,
            'channel_id': channel_id,
            'metadata': {
                'content_type': content_type,
                'topic': topic,
                'compression_level': self.compression_level,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '1.0'
            },
            'steps': {
                'trend_detection': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'script_creation': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'visual_generation': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'audio_generation': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'video_production': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'compliance_check': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'publication': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'monetization': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'post_publication_analysis': {'status': 'pending', 'retries': 0, 'timestamp': None}
            }
        }

        task = self.add_task(task_data, priority, cluster_id=cluster_id)
        self._update_dashboard_metrics()
        return {'success': True, 'task_id': task_id, 'cluster_id': cluster_id, 'task': task.__dict__}

    @require_authentication
    @monitor_performance
    def create_channel(self, name: str, niche: str, platforms: List[str], character: str) -> Dict:
        """Crea un nuevo canal"""
        if len(self.channels) >= self.max_channels:
            raise OrchestratorError(f"Límite de {self.max_channels} canales alcanzado")
        if niche not in self.config.get('niches', {}):
            raise OrchestratorError(f"Nicho inválido: {niche}")
        if not all(p in self.config.get('platforms', {}) for p in platforms):
            raise OrchestratorError("Plataformas inválidas")
        if character not in self.config.get('character_profiles', {}):
            raise OrchestratorError(f"Personaje inválido: {character}")

        channel_id = f"{niche}_{uuid.uuid4().hex}"
        channel = Channel(
            id=channel_id,
            name=name,
            niche=niche,
            platforms=platforms,
            character=character,
            created_at=datetime.datetime.now().isoformat(),
            status=ChannelStatus.ACTIVE,
            stats={'videos': 0, 'subscribers': 0, 'views': 0, 'revenue': 0, 'engagement_rate': 0.0},
            oauth_tokens={},
            performance_metrics={'roi': 0.0, 'growth_rate': 0.0}
        )
        self.channels[channel_id] = channel
        self.channel_status[channel_id] = ChannelStatus.ACTIVE
        knowledge_base = self._load_component('knowledge_base')
        if knowledge_base:
            knowledge_base.save_channel(channel.__dict__)
        logger.info(f"Canal creado: {name}")
        activity_logger.info(f"Nuevo canal {channel_id}: {name}, nicho={niche}")
        self._update_dashboard_metrics()
        return {'success': True, 'channel_id': channel_id}

    @require_authentication
    @monitor_performance
    def update_channel_status(self, channel_id: str, status: ChannelStatus) -> bool:
        """Actualiza el estado de un canal"""
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")
        self.channel_status[channel_id] = status
        self.channels[channel_id].status = status
        knowledge_base = self._load_component('knowledge_base')
        if knowledge_base:
            knowledge_base.save_channel(self.channels[channel_id].__dict__)
        logger.info(f"Estado de canal {channel_id} actualizado a {status}")
        activity_logger.info(f"Canal {channel_id} cambió a estado {status}")
        self._update_dashboard_metrics()
        return True

    @require_authentication
    @monitor_performance
    def get_channel_metrics(self, channel_id: str) -> Dict:
        """Obtiene métricas de un canal"""
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")
        analytics_engine = self._load_component('analytics_engine')
        metrics = self.channels[channel_id].stats
        if analytics_engine:
            stats = analytics_engine.get_channel_stats(channel_id)
            metrics.update(stats)
            self.channels[channel_id].stats.update(stats)
        self._update_dashboard_metrics()
        return {'success': True, 'metrics': metrics}

    @require_authentication
    @monitor_performance
    def optimize_channel(self, channel_id: str, priority: Priority = Priority.HIGH) -> Dict:
        """Optimiza un canal"""
        if not self.active:
            raise OrchestratorError("Orchestrator no está activo")
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")

        task_id = f"optimize_{channel_id}_{uuid.uuid4().hex}"
        cluster_id = f"cluster_optimize_{channel_id}_{uuid.uuid4().hex}"
        task_data = {
            'id': task_id,
            'type': TaskType.CHANNEL_OPTIMIZATION.value,
            'channel_id': channel_id,
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '1.0'
            },
            'steps': {
                'performance_analysis': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'competitor_analysis': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'content_strategy_update': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'cta_optimization': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'monetization_optimization': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'audience_segmentation': {'status': 'pending', 'retries': 0, 'timestamp': None}
            }
        }

        task = self.add_task(task_data, priority, cluster_id=cluster_id)
        self._update_dashboard_metrics()
        return {'success': True, 'task_id': task_id, 'cluster_id': cluster_id, 'task': task.__dict__}

    @require_authentication
    @monitor_performance
    def analyze_performance(self, channel_id: str, platform: str = None, time_range: str = '30d') -> Dict:
        """Analiza el rendimiento de un canal"""
        if not self.active:
            raise OrchestratorError("Orchestrator no está activo")
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")
        
        task_id = f"analysis_{channel_id}_{uuid.uuid4().hex}"
        cluster_id = f"cluster_analysis_{channel_id}_{uuid.uuid4().hex}"
        task_data = {
            'id': task_id,
            'type': TaskType.ANALYSIS.value,
            'channel_id': channel_id,
            'metadata': {
                'platform': platform,
                'time_range': time_range,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '1.0'
            },
            'steps': {
                'deep_analysis': {'status': 'pending', 'retries': 0, 'timestamp': None}
            }
        }

        task = self.add_task(task_data, Priority.NORMAL, cluster_id=cluster_id)
        self._update_dashboard_metrics()
        return {'success': True, 'task_id': task_id, 'cluster_id': cluster_id, 'task': task.__dict__}

    @require_authentication
    @monitor_performance
    def publish_content(self, channel_id: str, content: Dict, platforms: List[str] = None) -> Dict:
        """Publica contenido en múltiples plataformas"""
        if not self.active:
            raise OrchestratorError("Orchestrator no está activo")
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")

        channel = self.channels[channel_id]
        platforms = platforms or channel.platforms
        invalid_platforms = [p for p in platforms if p not in channel.platforms]
        if invalid_platforms:
            raise OrchestratorError(f"Plataformas no válidas para canal {channel_id}: {invalid_platforms}")

        task_id = f"publish_{channel_id}_{uuid.uuid4().hex}"
        cluster_id = f"cluster_publish_{channel_id}_{uuid.uuid4().hex}"
        task_data = {
            'id': task_id,
            'type': TaskType.PUBLICATION.value,
            'channel_id': channel_id,
            'metadata': {
                'content': content,
                'platforms': platforms,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '1.0'
            },
            'steps': {
                'pre_publish_check': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'content_adaptation': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'platform_publication': {'status': 'pending', 'retries': 0, 'timestamp': None},
                'post_publish_verification': {'status': 'pending', 'retries': 0, 'timestamp': None}
            }
        }

        task = self.add_task(task_data, Priority.HIGH, cluster_id=cluster_id)
        activity_logger.info(f"Publicación iniciada para canal {channel_id} en {platforms}")
        publication_counter.labels(platform=','.join(platforms), status='initiated').inc()
        self._update_dashboard_metrics()
        return {'success': True, 'task_id': task_id, 'cluster_id': cluster_id, 'task': task.__dict__}

    @require_authentication
    @monitor_performance
    def _process_shadowban_recovery(self, channel_id: str, platform: str) -> Dict:
        """Procesa la recuperación de shadowban para un canal en una plataforma"""
        if channel_id not in self.channels:
            raise OrchestratorError(f"Canal no encontrado: {channel_id}")
        if platform not in self.channels[channel_id].platforms:
            raise OrchestratorError(f"Plataforma {platform} no válida para canal {channel_id}")

        recovery_plan = self.recovery_plans.get(f"{channel_id}_{platform}", {})
        if not recovery_plan:
            recovery_plan = {
                'channel_id': channel_id,
                'platform': platform,
                'steps': [
                    {'type': 'reduce_frequency', 'duration': 604800, 'status': 'pending'},  # 7 días
                    {'type': 'content_adjustment', 'duration': 604800, 'status': 'pending'},
                    {'type': 'engagement_recovery', 'duration': 1209600, 'status': 'pending'},  # 14 días
                    {'type': 'appeal_submission', 'status': 'pending'}
                ],
                'start_time': datetime.datetime.now().isoformat(),
                'progress': 0.0,
                'metrics': {'initial_engagement': 0.0, 'current_engagement': 0.0}
            }
            self.recovery_plans[f"{channel_id}_{platform}"] = recovery_plan
            shadowban_counter.labels(platform=platform).inc()

        task_id = f"shadowban_recovery_{channel_id}_{platform}_{uuid.uuid4().hex}"
        cluster_id = f"cluster_recovery_{channel_id}_{platform}_{uuid.uuid4().hex}"
        task_data = {
            'id': task_id,
            'type': TaskType.SHADOWBAN_RECOVERY.value,
            'channel_id': channel_id,
            'metadata': {
                'platform': platform,
                'recovery_plan': recovery_plan,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '1.0'
            },
            'steps': {
                step['type']: {'status': step['status'], 'retries': 0, 'timestamp': None}
                for step in recovery_plan['steps']
            }
        }

        # Evitar estancamiento con lógica de reintentos y límites temporales
        max_recovery_duration = 2592000  # 30 días
        start_time = datetime.datetime.fromisoformat(recovery_plan['start_time'])
        if (datetime.datetime.now() - start_time).total_seconds() > max_recovery_duration:
            logger.warning(f"Recuperación de shadowban para {channel_id} en {platform} excedió tiempo máximo")
            self.update_channel_status(channel_id, ChannelStatus.RESTRICTED)
            return {'success': False, 'message': 'Recuperación excedió tiempo máximo'}

        task = self.add_task(task_data, Priority.CRITICAL, cluster_id=cluster_id)
        activity_logger.info(f"Iniciada recuperación de shadowban para canal {channel_id} en {platform}")
        self._update_dashboard_metrics()
        return {'success': True, 'task_id': task_id, 'cluster_id': cluster_id, 'task': task.__dict__}

    @monitor_performance
    def _monitor_resources(self):
        """Monitorea recursos del sistema en tiempo real"""
        while self.active:
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                active_threads = threading.active_count()
                network_io = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

                self.resource_metrics.update({
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'active_threads': active_threads,
                    'network_io': network_io
                })

                resource_gauge.labels(resource_type='cpu').set(cpu_usage)
                resource_gauge.labels(resource_type='memory').set(memory_usage)
                resource_gauge.labels(resource_type='disk').set(disk_usage)
                resource_gauge.labels(resource_type='threads').set(active_threads)
                resource_gauge.labels(resource_type='network_io').set(network_io)

                self._update_dashboard_metrics()

                # Alertas de recursos
                if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                    notifier = self._load_component('notifier')
                    if notifier:
                        notifier.send_alert(
                            f"Alto uso de recursos: CPU={cpu_usage}%, Memory={memory_usage}%, Disk={disk_usage}%",
                            priority='high'
                        )
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error en monitoreo de recursos: {str(e)}")
                time.sleep(300)

    @monitor_performance
    def _optimize_resources(self):
        """Optimiza el uso de recursos del sistema"""
        try:
            cpu_usage = self.resource_metrics.get('cpu_usage', 0.0)
            memory_usage = self.resource_metrics.get('memory_usage', 0.0)
            active_threads = self.resource_metrics.get('active_threads', 0)

            # Ajustar número de hilos concurrentes
            if cpu_usage > 80 or memory_usage > 80:
                new_max_threads = max(1, self.max_concurrent_tasks - 5)
                self.executor._max_workers = new_max_threads
                self.max_concurrent_tasks = new_max_threads
                logger.info(f"Reduciendo hilos concurrentes a {new_max_threads} debido a alto uso de recursos")
            elif cpu_usage < 50 and memory_usage < 50 and active_threads < self.max_concurrent_tasks:
                new_max_threads = min(self.max_concurrent_tasks + 5, 50)
                self.executor._max_workers = new_max_threads
                self.max_concurrent_tasks = new_max_threads
                logger.info(f"Aumentando hilos concurrentes a {new_max_threads}")

            # Limpiar caché si el uso de disco es alto
            if self.resource_metrics.get('disk_usage', 0.0) > 80:
                self._cleanup_cache()
                logger.info("Caché limpiado debido a alto uso de disco")

            # Rebalancear prioridades de tareas
            high_priority_tasks = sum(1 for _, task in self.task_queue.queue if task.priority == Priority.CRITICAL)
            if high_priority_tasks > self.max_concurrent_tasks / 2:
                for task_id, task in self.current_tasks.items():
                    if task.priority == Priority.LOW or task.priority == Priority.MAINTENANCE:
                        task.status = TaskStatus.PAUSED
                        self.paused_tasks[task_id] = task
                        self.current_tasks.pop(task_id)
                        self._persist_task(task)
                        logger.info(f"Tarea {task_id} pausada para priorizar tareas críticas")

            activity_logger.info(f"Optimización de recursos completada: CPU={cpu_usage}%, Memory={memory_usage}%, Threads={active_threads}")
            self._update_dashboard_metrics()
        except Exception as e:
            logger.error(f"Error en optimización de recursos: {str(e)}")
            raise ResourceLimitError(f"Error en optimización: {str(e)}")

    @monitor_performance
    def _cleanup_cache(self):
        """Limpia caché obsoleto"""
        try:
            keys = self.redis_client.keys("component:*") + self.redis_client.keys("subsystem:*")
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Limpiados {len(keys)} elementos de caché")
            cache_dir = 'data/cache'
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)
                logger.info("Directorio de caché en disco limpiado")
            activity_logger.info("Caché limpiado exitosamente")
        except Exception as e:
            logger.error(f"Error al limpiar caché: {str(e)}")

    @monitor_performance
    def _rotate_logs(self):
        """Rota logs antiguos"""
        try:
            log_dirs = ['logs/activity', 'logs/audit', 'logs/monitoring']
            max_age_days = 30
            for log_dir in log_dirs:
                for log_file in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, log_file)
                    file_age = (time.time() - os.path.getmtime(file_path)) / (24 * 3600)
                    if file_age > max_age_days:
                        os.remove(file_path)
                        logger.info(f"Log eliminado: {file_path}")
            activity_logger.info("Rotación de logs completada")
        except Exception as e:
            logger.error(f"Error al rotar logs: {str(e)}")

    @monitor_performance
    def _aggregate_metrics(self):
        """Agrega métricas para análisis"""
        try:
            with sqlite3.connect('data/orchestrator.db') as conn:
                cursor = conn.cursor()
                metrics = {
                    'resource_metrics': self.resource_metrics,
                    'publication_metrics': self.publication_metrics,
                    'performance_metrics': self.performance_metrics
                }
                cursor.execute(
                    "INSERT INTO metrics (timestamp, metrics) VALUES (?, ?)",
                    (datetime.datetime.now().isoformat(), json.dumps(metrics))
                )
                conn.commit()
            logger.info("Métricas agregadas a la base de datos")
            activity_logger.info("Agregación de métricas completada")
        except Exception as e:
            logger.error(f"Error al agregar métricas: {str(e)}")

    @monitor_performance
    def _monitor_rate_limits(self):
        """Monitorea límites de tasa de APIs"""
        try:
            for platform, limits in self.rate_limits.items():
                if limits['remaining'] <= 0 and time.time() < limits['reset_time']:
                    logger.warning(f"Límite de tasa alcanzado para {platform}. Esperando hasta {datetime.datetime.fromtimestamp(limits['reset_time'])}")
                    self.publication_metrics['by_platform'][platform]['rate_limit_hits'] += 1
                elif time.time() >= limits['reset_time']:
                    limits['remaining'] = limits['limit']
                    limits['reset_time'] = time.time() + 3600  # Reset cada hora
                    logger.info(f"Límite de tasa reiniciado para {platform}")
            activity_logger.info("Monitoreo de límites de tasa completado")
        except Exception as e:
            logger.error(f"Error al monitorear límites de tasa: {str(e)}")

    @monitor_performance
    def _run_performance_analysis(self):
        """Ejecuta análisis de rendimiento para todos los canales"""
        try:
            analytics_engine = self._load_component('analytics_engine')
            if not analytics_engine:
                raise OrchestratorError("Motor de análisis no disponible")

            for channel_id, channel in self.channels.items():
                stats = analytics_engine.get_channel_stats(channel_id)
                self.channels[channel_id].stats.update(stats)
                roi = stats.get('revenue', 0) / max(stats.get('investment', 1), 1)
                growth_rate = stats.get('subscribers', 0) / max(stats.get('previous_subscribers', 1), 1) - 1
                self.channels[channel_id].performance_metrics.update({
                    'roi': roi,
                    'growth_rate': growth_rate
                })
                self.performance_metrics['channel_roi'][channel_id] = roi
                for platform in channel.platforms:
                    self.performance_metrics['platform_engagement'][platform][channel_id] = stats.get('engagement_rate', 0.0)
                self._persist_channel(channel)
                logger.info(f"Análisis de rendimiento completado para canal {channel_id}: ROI={roi:.2f}, Crecimiento={growth_rate:.2%}")
            activity_logger.info("Análisis de rendimiento completado para todos los canales")
            self._update_dashboard_metrics()
        except Exception as e:
            logger.error(f"Error en análisis de rendimiento: {str(e)}")

    @monitor_performance
    def _persist_channel(self, channel: Channel):
        """Persiste datos de un canal en la base de datos"""
        try:
            with sqlite3.connect('data/orchestrator.db') as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO channels (
                        id, name, niche, platforms, character, created_at,
                        status, stats, oauth_tokens, performance_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    channel.id,
                    channel.name,
                    channel.niche,
                    json.dumps(channel.platforms),
                    channel.character,
                    channel.created_at,
                    channel.status.value,
                    json.dumps(channel.stats),
                    json.dumps(channel.oauth_tokens),
                    json.dumps(channel.performance_metrics)
                ))
                conn.commit()
            logger.info(f"Canal persistido: {channel.id}")
        except Exception as e:
            logger.error(f"Error al persistir canal {channel.id}: {str(e)}")

    @monitor_performance
    def _dashboard_updater(self):
        """Actualiza el dashboard periódicamente"""
        while self.active:
            try:
                self._update_dashboard_metrics()
                self.dashboard_manager.generate_html_dashboard()
                time.sleep(300)  # Actualizar cada 5 minutos
            except Exception as e:
                logger.error(f"Error al actualizar dashboard: {str(e)}")
                time.sleep(60)

    def _update_dashboard_metrics(self):
        """Actualiza métricas del dashboard"""
        dashboard_data = {
            'system_metrics': self.resource_metrics,
            'channel_metrics': {
                channel_id: {
                    'name': channel.name,
                    'status': channel.status.value,
                    'views': channel.stats.get('views', 0),
                    'subscribers': channel.stats.get('subscribers', 0),
                    'roi': channel.performance_metrics.get('roi', 0.0)
                } for channel_id, channel in self.channels.items()
            },
            'task_metrics': {
                'active_tasks': len(self.current_tasks),
                'paused_tasks': len(self.paused_tasks),
                'completion_rate': self.performance_metrics.get('task_completion_rate', 0.0)
            },
            'publication_metrics': self.publication_metrics,
            'shadowban_status': self.shadowban_status
        }
        self.dashboard_manager.update_dashboard(dashboard_data)

    @monitor_performance
    def _recovery_monitor(self):
        """Monitorea el progreso de recuperación de shadowbans"""
        while self.active:
            try:
                for recovery_key, plan in list(self.recovery_plans.items()):
                    channel_id, platform = recovery_key.split('_')
                    shadowban_detector = self._load_component('shadowban_detector')
                    if shadowban_detector:
                        status = shadowban_detector.check_status(channel_id, platform)
                        if status['shadowbanned']:
                            plan['progress'] = min(plan['progress'] + 0.1, 1.0)
                            self.shadowban_status[channel_id][platform] = status
                        else:
                            plan['progress'] = 1.0
                            self.update_channel_status(channel_id, ChannelStatus.ACTIVE)
                            self.shadowban_status[channel_id][platform] = {'shadowbanned': False}
                            del self.recovery_plans[recovery_key]
                            logger.info(f"Shadowban recuperado para canal {channel_id} en {platform}")
                            activity_logger.info(f"Recuperación completada para {channel_id} en {platform}")
                    self._update_dashboard_metrics()
                time.sleep(3600)  # Revisar cada hora
            except Exception as e:
                logger.error(f"Error en monitoreo de recuperación: {str(e)}")
                time.sleep(300)

    @monitor_performance
    def _main_loop(self):
        """Bucle principal para procesamiento de tareas"""
        while self.active:
            try:
                if not self.task_queue.empty():
                    _, task = self.task_queue.get()
                    if task.status == TaskStatus.INITIATED:
                        task.status = TaskStatus.PROCESSING
                        task.execution_history.append({
                            'event': 'processing_started',
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        self._persist_task(task)
                        self.executor.submit(self._execute_task, task)
                    self.task_queue.task_done()
                self._optimize_resources()
                time.sleep(self.task_check_interval)
            except Exception as e:
                logger.error(f"Error en bucle principal: {str(e)}")
                time.sleep(60)

    @monitor_performance
    def _execute_task(self, task: Task):
        """Ejecuta una tarea específica"""
        try:
            task.execution_history.append({
                'event': 'execution_attempt',
                'timestamp': datetime.datetime.now().isoformat(),
                'attempt': task.retries + 1
            })

            if task.dependencies:
                for dep_id in task.dependencies:
                    dep_task = self.current_tasks.get(dep_id) or self.paused_tasks.get(dep_id)
                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                        task.status = TaskStatus.PAUSED
                        self.paused_tasks[task.id] = task
                        self._persist_task(task)
                        logger.info(f"Tarea {task.id} pausada: dependencias incompletas")
                        return

            if task.type == TaskType.CONTENT_CREATION:
                self._execute_content_creation_task(task)
            elif task.type == TaskType.CHANNEL_OPTIMIZATION:
                self._execute_optimization_task(task)
            elif task.type == TaskType.SHADOWBAN_RECOVERY:
                self._execute_shadowban_recovery_task(task)
            elif task.type == TaskType.PUBLICATION:
                self._execute_publication_task(task)
            elif task.type == TaskType.ANALYSIS:
                self._execute_analysis_task(task)
            elif task.type == TaskType.MONETIZATION:
                self._execute_monetization_task(task)
            elif task.type == TaskType.AUDIT:
                self._execute_audit_task(task)
            elif task.type == TaskType.MAINTENANCE:
                self._execute_maintenance_task(task)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.datetime.now().isoformat()
            task.execution_history.append({
                'event': 'completed',
                'timestamp': task.completed_at
            })
            self.task_history.append(task)
            self.current_tasks.pop(task.id, None)
            self._persist_task(task)
            task_counter.labels(type=task.type.value, status=task.status.value).inc()
            logger.info(f"Tarea {task.id} completada: {task.type}")
            activity_logger.info(f"Tarea {task.id} completada: tipo={task.type}, canal={task.channel_id}")
            self._update_dashboard_metrics()

        except Exception as e:
            task.retries += 1
            task.error = str(e)
            task.execution_history.append({
                'event': 'error',
                'timestamp': datetime.datetime.now().isoformat(),
                'error': task.error
            })

            if task.retries >= self.max_retries:
                task.status = TaskStatus.FAILED
                self.current_tasks.pop(task.id, None)
                self.task_history.append(task)
                logger.error(f"Tarea {task.id} fallida tras {task.retries} intentos: {task.error}")
                activity_logger.error(f"Tarea {task.id} fallida: {task.error}")
            else:
                task.status = TaskStatus.INITIATED
                delay = self.base_retry_delay ** task.retries
                jitter = random.uniform(0, 0.1 * delay)
                time.sleep(delay + jitter)
                self.task_queue.put((task.priority.value, task))
                logger.warning(f"Reintentando tarea {task.id} ({task.retries}/{self.max_retries}): {task.error}")

            self._persist_task(task)
            task_counter.labels(type=task.type.value, status=task.status.value).inc()
            self._update_dashboard_metrics()

    @monitor_performance
    def _execute_content_creation_task(self, task: Task):
        """Ejecuta tarea de creación de contenido"""
        channel_id = task.channel_id
        channel = self.channels[channel_id]
        metadata = task.metadata
        content_type = metadata.get('content_type', 'video')
        topic = metadata.get('topic')

        steps = task.steps
        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'trend_detection':
                    trend_radar = self._load_subsystem('trends', 'trend_radar')
                    if trend_radar:
                        trends = trend_radar.detect_trends(channel.niche)
                        metadata['trends'] = trends
                        logger.info(f"Tendencias detectadas para {channel_id}: {trends}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'script_creation':
                    script_factory = self._load_subsystem('creation', 'script_factory')
                    if script_factory:
                        script = script_factory.generate_script(
                            niche=channel.niche,
                            character=channel.character,
                            topic=topic,
                            content_type=content_type,
                            trends=metadata.get('trends', [])
                        )
                        metadata['script'] = script
                        logger.info(f"Script generado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'visual_generation':
                    video_composer = self._load_subsystem('creation', 'video_composer')
                    if video_composer:
                        visuals = video_composer.generate_visuals(
                            script=metadata.get('script'),
                            character=channel.character,
                            content_type=content_type
                        )
                        metadata['visuals'] = visuals
                        logger.info(f"Visuales generados para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'audio_generation':
                    audio_synthesizer = self._load_subsystem('creation', 'audio_synthesizer')
                    if audio_synthesizer:
                        audio = audio_synthesizer.generate_audio(
                            script=metadata.get('script'),
                            character=channel.character,
                            content_type=content_type
                        )
                        metadata['audio'] = audio
                        logger.info(f"Audio generado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'video_production':
                    video_composer = self._load_subsystem('creation', 'video_composer')
                    if video_composer:
                        content = video_composer.compose_video(
                            visuals=metadata.get('visuals'),
                            audio=metadata.get('audio'),
                            content_type=content_type
                        )
                        metadata['content'] = content
                        logger.info(f"Video producido para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'compliance_check':
                    content_auditor = self._load_subsystem('compliance', 'content_auditor')
                    if content_auditor:
                        compliance_result = content_auditor.audit_content(
                            content=metadata.get('content'),
                            platform_rules=self.config['platforms']
                        )
                        if not compliance_result['compliant']:
                            raise ComplianceError(f"Contenido no cumple normativas: {compliance_result['issues']}")
                        metadata['compliance'] = compliance_result
                        logger.info(f"Contenido auditado para {channel_id}: Cumple normativas")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'publication':
                    self._execute_publication_task(task)

                elif step_name == 'monetization':
                    revenue_optimizer = self._load_subsystem('monetization', 'revenue_optimizer')
                    if revenue_optimizer:
                        monetization_plan = revenue_optimizer.optimize(
                            content=metadata.get('content'),
                            channel=channel.__dict__,
                            platforms=channel.platforms
                        )
                        metadata['monetization_plan'] = monetization_plan
                        logger.info(f"Plan de monetización generado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'post_publication_analysis':
                    analytics_engine = self._load_component('analytics_engine')
                    if analytics_engine:
                        performance = analytics_engine.analyze_content_performance(
                            content_id=metadata.get('content_id'),
                            channel_id=channel_id
                        )
                        metadata['performance'] = performance
                        self.channels[channel_id].stats['views'] += performance.get('views', 0)
                        self.channels[channel_id].stats['revenue'] += performance.get('revenue', 0)
                        self._persist_channel(channel)
                        logger.info(f"Análisis post-publicación completado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise OrchestratorError(f"Error en paso {step_name} de tarea {task.id}: {str(e)}")

    @monitor_performance
    def _execute_optimization_task(self, task: Task):
        """Ejecuta tarea de optimización de canal"""
        channel_id = task.channel_id
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'performance_analysis':
                    analytics_engine = self._load_component('analytics_engine')
                    if analytics_engine:
                        stats = analytics_engine.get_channel_stats(channel_id)
                        self.channels[channel_id].stats.update(stats)
                        logger.info(f"Análisis de rendimiento completado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'competitor_analysis':
                    competitor_analyzer = self._load_subsystem('analysis', 'competitor_analyzer')
                    if competitor_analyzer:
                        competitors = competitor_analyzer.analyze_competitors(
                            niche=self.channels[channel_id].niche,
                            platforms=self.channels[channel_id].platforms
                        )
                        task.metadata['competitors'] = competitors
                        logger.info(f"Análisis de competidores completado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'content_strategy_update':
                    content_optimizer = self._load_subsystem('optimization', 'content_optimizer')
                    if content_optimizer:
                        strategy = content_optimizer.update_strategy(
                            channel=self.channels[channel_id].__dict__,
                            competitors=task.metadata.get('competitors', [])
                        )
                        task.metadata['content_strategy'] = strategy
                        logger.info(f"Estrategia de contenido actualizada para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'cta_optimization':
                    content_optimizer = self._load_subsystem('optimization', 'content_optimizer')
                    if content_optimizer:
                        cta_plan = content_optimizer.optimize_cta(
                            channel=self.channels[channel_id].__dict__,
                            performance=self.channels[channel_id].stats
                        )
                        task.metadata['cta_plan'] = cta_plan
                        logger.info(f"CTA optimizado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'monetization_optimization':
                    revenue_optimizer = self._load_subsystem('monetization', 'revenue_optimizer')
                    if revenue_optimizer:
                        monetization_plan = revenue_optimizer.optimize(
                            channel=self.channels[channel_id].__dict__,
                            platforms=self.channels[channel_id].platforms
                        )
                        task.metadata['monetization_plan'] = monetization_plan
                        logger.info(f"Plan de monetización optimizado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'audience_segmentation':
                    audience_analyzer = self._load_subsystem('analysis', 'audience_analyzer')
                    if audience_analyzer:
                        segments = audience_analyzer.segment_audience(
                            channel_id=channel_id,
                            performance=self.channels[channel_id].stats
                        )
                        task.metadata['audience_segments'] = segments
                        logger.info(f"Segmentación de audiencia completada para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise OrchestratorError(f"Error en paso {step_name} de tarea {task.id}: {str(e)}")

    @monitor_performance
    def _execute_shadowban_recovery_task(self, task: Task):
        """Ejecuta tarea de recuperación de shadowban"""
        channel_id = task.channel_id
        platform = task.metadata['platform']
        recovery_plan = task.metadata['recovery_plan']
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'reduce_frequency':
                    publication_adapter = self._load_subsystem('publication', f'{platform}_adapter')
                    if publication_adapter:
                        publication_adapter.adjust_frequency(channel_id, reduction=0.5)
                        logger.info(f"Frecuencia de publicación reducida para {channel_id} en {platform}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'content_adjustment':
                    content_optimizer = self._load_subsystem('optimization', 'content_optimizer')
                    if content_optimizer:
                        content_optimizer.adjust_content(
                            channel_id=channel_id,
                            platform=platform,
                            shadowban_status=self.shadowban_status.get(channel_id, {}).get(platform, {})
                        )
                        logger.info(f"Contenido ajustado para {channel_id} en {platform}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'engagement_recovery':
                    audience_analyzer = self._load_subsystem('analysis', 'audience_analyzer')
                    if audience_analyzer:
                        engagement_plan = audience_analyzer.plan_engagement_recovery(
                            channel_id=channel_id,
                            platform=platform
                        )
                        task.metadata['engagement_plan'] = engagement_plan
                        logger.info(f"Plan de recuperación de engagement generado para {channel_id} en {platform}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'appeal_submission':
                    publication_adapter = self._load_subsystem('publication', f'{platform}_adapter')
                    if publication_adapter:
                        appeal_result = publication_adapter.submit_appeal(
                            channel_id=channel_id,
                            reason="Shadowban recovery"
                        )
                        task.metadata['appeal_result'] = appeal_result
                        logger.info(f"Apelación enviada para {channel_id} en {platform}")
                    steps[step_name]['status'] = 'completed'

                recovery_plan['progress'] = min(recovery_plan['progress'] + 0.25, 1.0)
                task.metadata['recovery_plan'] = recovery_plan
                self.recovery_plans[f"{channel_id}_{platform}"] = recovery_plan
                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise ShadowbanError(f"Error en paso {step_name} de recuperación para {channel_id} en {platform}: {str(e)}")

    @monitor_performance
    def _execute_publication_task(self, task: Task):
        """Ejecuta tarea de publicación"""
        channel_id = task.channel_id
        content = task.metadata.get('content')
        platforms = task.metadata.get('platforms', self.channels[channel_id].platforms)
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'pre_publish_check':
                    platform_compliance = self._load_subsystem('compliance', 'platform_compliance')
                    if platform_compliance:
                        for platform in platforms:
                            compliance_result = platform_compliance.check_compliance(
                                content=content,
                                platform=platform
                            )
                            if not compliance_result['compliant']:
                                raise ComplianceError(f"Contenido no cumple con {platform}: {compliance_result['issues']}")
                        logger.info(f"Chequeo previo a publicación completado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'content_adaptation':
                    content_optimizer = self._load_subsystem('optimization', 'content_optimizer')
                    if content_optimizer:
                        adapted_content = {}
                        for platform in platforms:
                            adapted_content[platform] = content_optimizer.adapt_content(
                                content=content,
                                platform=platform,
                                channel=self.channels[channel_id].__dict__
                            )
                        task.metadata['adapted_content'] = adapted_content
                        logger.info(f"Contenido adaptado para {channel_id} en {platforms}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'platform_publication':
                    publication_results = {}
                    for platform in platforms:
                        if self.rate_limits[platform]['remaining'] <= 0:
                            raise PlatformError(f"Límite de tasa alcanzado para {platform}")
                        adapter = self._load_subsystem('publication', f'{platform}_adapter')
                        if adapter:
                            result = adapter.publish(
                                channel_id=channel_id,
                                content=task.metadata['adapted_content'][platform]
                            )
                            publication_results[platform] = result
                            self.rate_limits[platform]['remaining'] -= 1
                            self.publication_metrics['total_attempts'] += 1
                            self.publication_metrics['by_platform'][platform]['attempts'] += 1
                            if result['status'] == 'success':
                                self.publication_metrics['successful_publishes'] += 1
                                self.publication_metrics['by_platform'][platform]['successes'] += 1
                                publication_counter.labels(platform=platform, status='success').inc()
                            else:
                                self.publication_metrics['failed_publishes'] += 1
                                self.publication_metrics['by_platform'][platform]['failures'] += 1
                                publication_counter.labels(platform=platform, status='failed').inc()
                                raise PlatformError(f"Error al publicar en {platform}: {result['message']}")
                    task.metadata['publication_results'] = publication_results
                    logger.info(f"Publicación completada para {channel_id} en {platforms}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'post_publish_verification':
                    analytics_engine = self._load_component('analytics_engine')
                    if analytics_engine:
                        for platform in platforms:
                            verification = analytics_engine.verify_publication(
                                channel_id=channel_id,
                                platform=platform,
                                publication_id=task.metadata['publication_results'][platform].get('publication_id')
                            )
                            if not verification['published']:
                                raise PlatformError(f"Publicación no verificada en {platform}")
                        logger.info(f"Verificación post-publicación completada para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise PlatformError(f"Error en paso {step_name} de publicación para {channel_id}: {str(e)}")

    @monitor_performance
    def _execute_analysis_task(self, task: Task):
        """Ejecuta tarea de análisis"""
        channel_id = task.channel_id
        platform = task.metadata.get('platform')
        time_range = task.metadata.get('time_range', '30d')
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'deep_analysis':
                    analytics_engine = self._load_component('analytics_engine')
                    if analytics_engine:
                        analysis_result = analytics_engine.deep_analysis(
                            channel_id=channel_id,
                            platform=platform,
                            time_range=time_range
                        )
                        task.metadata['analysis_result'] = analysis_result
                        self.channels[channel_id].performance_metrics.update({
                            'engagement_rate': analysis_result.get('engagement_rate', 0.0),
                            'view_growth': analysis_result.get('view_growth', 0.0)
                        })
                        self._persist_channel(self.channels[channel_id])
                        logger.info(f"Análisis profundo completado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise OrchestratorError(f"Error en paso {step_name} de análisis para {channel_id}: {str(e)}")

    @monitor_performance
    def _execute_monetization_task(self, task: Task):
        """Ejecuta tarea de monetización"""
        channel_id = task.channel_id
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'revenue_optimization':
                    revenue_optimizer = self._load_subsystem('monetization', 'revenue_optimizer')
                    if revenue_optimizer:
                        revenue_plan = revenue_optimizer.optimize(
                            channel=self.channels[channel_id].__dict__,
                            platforms=self.channels[channel_id].platforms
                        )
                        task.metadata['revenue_plan'] = revenue_plan
                        logger.info(f"Plan de ingresos optimizado para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'affiliate_integration':
                    affiliate_manager = self._load_subsystem('monetization', 'affiliate_manager')
                    if affiliate_manager:
                        affiliate_links = affiliate_manager.generate_links(
                            channel_id=channel_id,
                            niche=self.channels[channel_id].niche
                        )
                        task.metadata['affiliate_links'] = affiliate_links
                        logger.info(f"Enlaces de afiliados generados para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'ad_optimization':
                    ad_optimizer = self._load_subsystem('monetization', 'ad_optimizer')
                    if ad_optimizer:
                        ad_plan = ad_optimizer.optimize_ads(
                            channel_id=channel_id,
                            performance=self.channels[channel_id].stats
                        )
                        task.metadata['ad_plan'] = ad_plan
                        logger.info(f"Publicidad optimizada para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise OrchestratorError(f"Error en paso {step_name} de monetización para {channel_id}: {str(e)}")

    @monitor_performance
    def _execute_audit_task(self, task: Task):
        """Ejecuta tarea de auditoría"""
        channel_id = task.channel_id
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'compliance_audit':
                    content_auditor = self._load_subsystem('compliance', 'content_auditor')
                    if content_auditor:
                        audit_result = content_auditor.audit_channel(
                            channel_id=channel_id,
                            platforms=self.channels[channel_id].platforms
                        )
                        task.metadata['audit_result'] = audit_result
                        if not audit_result['compliant']:
                            self.update_channel_status(channel_id, ChannelStatus.RESTRICTED)
                            logger.warning(f"Canal {channel_id} no cumple normativas: {audit_result['issues']}")
                        logger.info(f"Auditoría de cumplimiento completada para {channel_id}")
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise ComplianceError(f"Error en paso {step_name} de auditoría para {channel_id}: {str(e)}")

    @monitor_performance
    def _execute_maintenance_task(self, task: Task):
        """Ejecuta tarea de mantenimiento"""
        steps = task.steps

        for step_name in steps:
            if steps[step_name]['status'] != 'pending':
                continue

            steps[step_name]['timestamp'] = datetime.datetime.now().isoformat()
            try:
                if step_name == 'cache_cleanup':
                    self._cleanup_cache()
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'log_rotation':
                    self._rotate_logs()
                    steps[step_name]['status'] = 'completed'

                elif step_name == 'metrics_aggregation':
                    self._aggregate_metrics()
                    steps[step_name]['status'] = 'completed'

                task.steps = steps
                self._persist_task(task)
            except Exception as e:
                steps[step_name]['status'] = 'failed'
                steps[step_name]['error'] = str(e)
                steps[step_name]['retries'] += 1
                task.steps = steps
                self._persist_task(task)
                raise OrchestratorError(f"Error en paso {step_name} de mantenimiento: {str(e)}")

    @monitor_performance
    def _priority_handler(self):
        """Maneja rebalanceo dinámico de prioridades"""
        while self.active:
            try:
                high_priority_count = sum(1 for _, task in self.task_queue.queue if task.priority == Priority.CRITICAL)
                if high_priority_count > self.max_concurrent_tasks / 2:
                    for task_id, task in self.current_tasks.items():
                        if task.priority in [Priority.LOW, Priority.MAINTENANCE]:
                            task.status = TaskStatus.PAUSED
                            self.paused_tasks[task_id] = task
                            self.current_tasks.pop(task_id)
                            self._persist_task(task)
                            logger.info(f"Tarea {task_id} pausada por exceso de tareas críticas")
                else:
                    for task_id, task in list(self.paused_tasks.items()):
                        if task.status == TaskStatus.PAUSED:
                            task.status = TaskStatus.INITIATED
                            self.current_tasks[task_id] = task
                            self.task_queue.put((task.priority.value, task))
                            self.paused_tasks.pop(task_id)
                            self._persist_task(task)
                            logger.info(f"Tarea {task_id} reanudada")
                time.sleep(300)  # Rebalancear cada 5 minutos
            except Exception as e:
                logger.error(f"Error en rebalanceo de prioridades: {str(e)}")
                time.sleep(60)

    @monitor_performance
    def _run_daily_sequence(self):
        """Ejecuta la secuencia diaria para los 5 canales"""
        while self.active:
            try:
                self._execute_daily_sequence()
                time.sleep(self.daily_sequence_interval)
            except Exception as e:
                logger.error(f"Error en secuencia diaria: {str(e)}")
                time.sleep(3600)

    @monitor_performance
    def _execute_daily_sequence(self):
        """Ejecuta la secuencia diaria de tareas para todos los canales"""
        for channel_id, channel in self.channels.items():
            try:
                if channel.status in [ChannelStatus.ACTIVE, ChannelStatus.RECOVERING]:
                    # 1. Análisis de rendimiento
                    self.analyze_performance(channel_id=channel_id, time_range='24h')

                    # 2. Detección de shadowbans
                    shadowban_detector = self._load_component('shadowban_detector')
                    if shadowban_detector:
                        for platform in channel.platforms:
                            status = shadowban_detector.check_status(channel_id, platform)
                            if status['shadowbanned']:
                                self.update_channel_status(channel_id, ChannelStatus.SHADOWBANNED)
                                self._process_shadowban_recovery(channel_id, platform)
                                shadowban_counter.labels(platform=platform).inc()
                            self.shadowban_status[channel_id][platform] = status

                    # 3. Creación de contenido
                    if channel.status == ChannelStatus.ACTIVE:
                        content_type = self.config['creation']['content_types'][0]
                        self.create_content(
                            channel_id=channel_id,
                            content_type=content_type,
                            topic=None,
                            priority=Priority.NORMAL
                        )

                    # 4. Optimización de canal
                    self.optimize_channel(channel_id=channel_id)

                    # 5. Publicación si no hay shadowban
                    if channel.status == ChannelStatus.ACTIVE:
                        content = {'type': 'video', 'data': 'placeholder_content'}  # Simulación
                        self.publish_content(
                            channel_id=channel_id,
                            content=content,
                            platforms=channel.platforms
                        )

                    # 6. Auditoría de cumplimiento
                    task_id = f"audit_{channel_id}_{uuid.uuid4().hex}"
                    task_data = {
                        'id': task_id,
                        'type': TaskType.AUDIT.value,
                        'channel_id': channel_id,
                        'metadata': {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'version': '1.0'
                        },
                        'steps': {
                            'compliance_audit': {'status': 'pending', 'retries': 0, 'timestamp': None}
                        }
                    }
                    self.add_task(task_data, Priority.HIGH)

                    logger.info(f"Secuencia diaria completada para canal {channel_id}")
                    activity_logger.info(f"Secuencia diaria ejecutada para {channel_id}")
            except Exception as e:
                logger.error(f"Error en secuencia diaria para canal {channel_id}: {str(e)}")
            finally:
                self._update_dashboard_metrics()

if __name__ == "__main__":
    orchestrator = Orchestrator()
    try:
        orchestrator.start()
        while orchestrator.active:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()
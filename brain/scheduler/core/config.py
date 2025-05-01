"""
Gestión de configuración para el sistema de planificación.

Este módulo proporciona funcionalidades para cargar, validar y acceder
a la configuración del sistema de planificación de tareas.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
import copy

logger = logging.getLogger('Scheduler.Config')

# Configuración por defecto
DEFAULT_CONFIG = {
    # Configuración general
    'max_workers': {
        'thread': 10,
        'process': 5,
        'distributed': 20
    },
    'executors': {
        'local': True,
        'thread': True,
        'process': True,
        'distributed': False,
        'cron': True
    },
    'default_timeout': 300,  # 5 minutos
    'default_retry_policy': {
        'max_retries': 3,
        'retry_delay': 5,  # segundos
        'backoff_factor': 2,
        'max_delay': 60  # segundos
    },
    
    # Configuración de throttling
    'throttling': {
        'enabled': True,
        'default_rate': 10,  # tareas por minuto
        'rates': {
            'content_creation': 5,
            'publish': 2,
            'analytics': 20
        }
    },
    
    # Configuración de circuit breaker
    'circuit_breaker': {
        'enabled': True,
        'failure_threshold': 5,  # número de fallos para abrir el circuito
        'recovery_timeout': 60,  # segundos antes de intentar recuperación
        'half_open_max_calls': 3,  # máximo de llamadas en estado semi-abierto
        'monitored_tasks': ['publish', 'content_creation', 'analytics']
    },
    
    # Configuración de persistencia
    'persistence': {
        'enabled': True,
        'storage_type': 'sqlite',  # sqlite, mongodb, redis
        'connection_string': 'scheduler.db',
        'auto_save_interval': 60,  # segundos
        'max_history_size': 1000,  # número máximo de tareas en historial
        'cleanup_interval': 86400  # segundos (1 día)
    },
    
    # Configuración de monitoreo
    'monitoring': {
        'enabled': True,
        'metrics_interval': 60,  # segundos
        'alert_thresholds': {
            'queue_size': 100,
            'task_latency': 300,  # segundos
            'error_rate': 0.1  # 10% de tasa de error
        },
        'notifier_integration': True
    },
    
    # Configuración de priorización
    'priority': {
        'dynamic_adjustment': True,
        'aging_factor': 0.1,  # factor de envejecimiento para tareas antiguas
        'max_starvation_time': 3600,  # segundos máximos de espera para tareas de baja prioridad
        'roi_based_priority': True  # priorizar basado en ROI estimado
    },
    
    # Configuración de tareas recurrentes
    'recurring_tasks': {
        'timezone': 'UTC',
        'max_missed_runs': 3,  # máximo de ejecuciones perdidas a recuperar
        'jitter': 30,  # segundos de variación aleatoria para evitar picos
        'default_misfire_policy': 'reschedule'  # reschedule, ignore
    }
}

class SchedulerConfig:
    """
    Gestiona la configuración del sistema de planificación.
    Permite cargar desde archivos, variables de entorno y overrides.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Inicializa la configuración con valores por defecto y overrides.
        
        Args:
            config_override: Configuración personalizada que sobrescribe los valores por defecto
        """
        # Inicializar con configuración por defecto
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        
        # Cargar configuración desde archivo si existe
        self._load_from_file()
        
        # Cargar configuración desde variables de entorno
        self._load_from_env()
        
        # Aplicar overrides específicos
        if config_override:
            self._apply_overrides(config_override)
        
        # Validar configuración
        self._validate_config()
        
        logger.info("Configuración del scheduler cargada")
    
    def _load_from_file(self) -> None:
        """
        Carga configuración desde archivo JSON.
        """
        config_paths = [
            os.path.join(os.getcwd(), 'config', 'scheduler.json'),
            os.path.join(os.getcwd(), 'scheduler.json'),
            os.path.expanduser('~/.content-bot/scheduler.json')
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                        self._apply_overrides(file_config)
                    logger.info(f"Configuración cargada desde {path}")
                    break
                except Exception as e:
                    logger.warning(f"Error al cargar configuración desde {path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """
        Carga configuración desde variables de entorno.
        
        Formato: SCHEDULER_SECTION_KEY=value
        Ejemplo: SCHEDULER_THROTTLING_ENABLED=false
        """
        prefix = "SCHEDULER_"
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                try:
                    # Separar sección y clave
                    parts = env_var[len(prefix):].lower().split('_', 1)
                    if len(parts) != 2:
                        continue
                    
                    section, key = parts
                    
                    # Convertir valor según tipo
                    if value.lower() in ('true', 'yes', '1'):
                        value = True
                    elif value.lower() in ('false', 'no', '0'):
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                        value = float(value)
                    
                    # Aplicar a la configuración
                    if section in self.config:
                        if key in self.config[section]:
                            self.config[section][key] = value
                            logger.debug(f"Configuración actualizada desde variable de entorno: {env_var}={value}")
                except Exception as e:
                    logger.warning(f"Error al procesar variable de entorno {env_var}: {str(e)}")
    
    def _apply_overrides(self, overrides: Dict[str, Any], target: Optional[Dict[str, Any]] = None, path: str = "") -> None:
        """
        Aplica overrides de configuración de forma recursiva.
        
        Args:
            overrides: Diccionario con overrides a aplicar
            target: Diccionario objetivo (por defecto self.config)
            path: Ruta actual para logging
        """
        if target is None:
            target = self.config
        
        for key, value in overrides.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursión para diccionarios anidados
                self._apply_overrides(value, target[key], current_path)
            else:
                # Asignar valor directamente
                target[key] = value
                logger.debug(f"Configuración override aplicado: {current_path}={value}")
    
    def _validate_config(self) -> None:
        """
        Valida la configuración y aplica correcciones si es necesario.
        """
        # Validar límites de workers
        for executor, workers in self.config.get('max_workers', {}).items():
            if not isinstance(workers, int) or workers < 1:
                logger.warning(f"Valor inválido para max_workers.{executor}: {workers}. Usando valor por defecto.")
                self.config['max_workers'][executor] = DEFAULT_CONFIG['max_workers'].get(executor, 5)
        
        # Validar timeout
        if not isinstance(self.config.get('default_timeout'), (int, float)) or self.config['default_timeout'] <= 0:
            logger.warning(f"Valor inválido para default_timeout: {self.config.get('default_timeout')}. Usando valor por defecto.")
            self.config['default_timeout'] = DEFAULT_CONFIG['default_timeout']
        
        # Validar política de reintentos
        retry_policy = self.config.get('default_retry_policy', {})
        if not isinstance(retry_policy.get('max_retries'), int) or retry_policy['max_retries'] < 0:
            logger.warning(f"Valor inválido para default_retry_policy.max_retries. Usando valor por defecto.")
            retry_policy['max_retries'] = DEFAULT_CONFIG['default_retry_policy']['max_retries']
        
        # Otras validaciones según sea necesario...
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración.
        
        Args:
            section: Sección de configuración
            key: Clave específica (opcional)
            default: Valor por defecto si no existe
            
        Returns:
            Valor de configuración o default si no existe
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Establece un valor de configuración.
        
        Args:
            section: Sección de configuración
            key: Clave específica
            value: Valor a establecer
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        logger.debug(f"Configuración actualizada: {section}.{key}={value}")
    
    def reload(self) -> None:
        """
        Recarga la configuración desde fuentes externas.
        """
        # Guardar overrides actuales
        current_config = copy.deepcopy(self.config)
        
        # Reiniciar a valores por defecto
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        
        # Recargar desde archivo y variables de entorno
        self._load_from_file()
        self._load_from_env()
        
        # Aplicar overrides anteriores
        self._apply_overrides(current_config)
        
        # Validar configuración
        self._validate_config()
        
        logger.info("Configuración recargada")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a un diccionario.
        
        Returns:
            Dict: Configuración completa como diccionario
        """
        return copy.deepcopy(self.config)
    
    def save_to_file(self, path: Optional[str] = None) -> bool:
        """
        Guarda la configuración actual a un archivo.
        
        Args:
            path: Ruta del archivo (opcional)
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        if path is None:
            path = os.path.join(os.getcwd(), 'config', 'scheduler.json')
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuración guardada en {path}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar configuración en {path}: {str(e)}")
            return False
"""
Middleware de control de tasas de ejecución para el sistema de planificación.

Este módulo implementa un middleware que limita la frecuencia de ejecución
de tareas según reglas configurables, protegiendo APIs externas y recursos
del sistema contra sobrecarga.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ..core.task_model import Task, TaskStatus
from .base_middleware import BaseMiddleware

logger = logging.getLogger('Scheduler.Middleware.Throttling')

class ThrottlingRule:
    """
    Regla de limitación de tasas para un tipo específico de tarea o recurso.
    
    Atributos:
        key: Identificador de la regla (tipo de tarea, API, etc.)
        rate_limit: Número máximo de ejecuciones permitidas
        time_window: Ventana de tiempo en segundos para aplicar el límite
        priority_bypass: Si True, tareas de alta prioridad pueden saltarse la limitación
        cooldown_period: Tiempo de espera adicional tras alcanzar el límite
    """
    
    def __init__(self, key: str, rate_limit: int, time_window: int, 
                priority_bypass: bool = False, cooldown_period: int = 0):
        """
        Inicializa una regla de limitación de tasas.
        
        Args:
            key: Identificador de la regla (tipo de tarea, API, etc.)
            rate_limit: Número máximo de ejecuciones permitidas
            time_window: Ventana de tiempo en segundos para aplicar el límite
            priority_bypass: Si True, tareas de alta prioridad pueden saltarse la limitación
            cooldown_period: Tiempo de espera adicional tras alcanzar el límite
        """
        self.key = key
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.priority_bypass = priority_bypass
        self.cooldown_period = cooldown_period
        
        # Validación básica
        if rate_limit <= 0:
            raise ValueError("El límite de tasa debe ser mayor que cero")
        if time_window <= 0:
            raise ValueError("La ventana de tiempo debe ser mayor que cero")
        if cooldown_period < 0:
            raise ValueError("El período de enfriamiento no puede ser negativo")

class ThrottlingMiddleware(BaseMiddleware):
    """
    Middleware que controla la tasa de ejecución de tareas.
    
    Características:
    - Limita ejecuciones por tipo de tarea, API o recurso
    - Soporta múltiples reglas con diferentes configuraciones
    - Permite excepciones para tareas de alta prioridad
    - Proporciona estadísticas de uso y rechazos
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el middleware de control de tasas.
        
        Args:
            config: Configuración específica del middleware
        """
        super().__init__(config)
        
        # Inicializar reglas y contadores
        self.rules: Dict[str, ThrottlingRule] = {}
        self.execution_history: Dict[str, deque] = defaultdict(deque)
        self.stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'total_requests': 0,
            'throttled_requests': 0,
            'bypassed_requests': 0,
            'last_throttled': None
        })
        
        # Bloqueo para acceso thread-safe
        self.lock = threading.RLock()
        
        # Cargar reglas desde configuración
        self._load_rules_from_config()
        
        logger.info(f"ThrottlingMiddleware inicializado con {len(self.rules)} reglas")
    
    def _load_rules_from_config(self) -> None:
        """
        Carga reglas de limitación desde la configuración.
        """
        if not self.config or 'rules' not in self.config:
            # Configurar reglas por defecto
            self._set_default_rules()
            return
        
        # Cargar reglas personalizadas
        for rule_config in self.config.get('rules', []):
            try:
                rule = ThrottlingRule(
                    key=rule_config['key'],
                    rate_limit=rule_config['rate_limit'],
                    time_window=rule_config['time_window'],
                    priority_bypass=rule_config.get('priority_bypass', False),
                    cooldown_period=rule_config.get('cooldown_period', 0)
                )
                self.add_rule(rule)
                logger.debug(f"Regla cargada: {rule.key} - {rule.rate_limit}/{rule.time_window}s")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar regla de throttling: {str(e)}")
    
    def _set_default_rules(self) -> None:
        """
        Configura reglas predeterminadas para APIs comunes.
        """
        # Reglas generales por tipo de tarea
        default_rules = [
            # Regla general para todas las tareas (fallback)
            ThrottlingRule("default", 100, 60),
            
            # APIs externas comunes
            ThrottlingRule("youtube_api", 10, 60, priority_bypass=True),
            ThrottlingRule("tiktok_api", 10, 60),
            ThrottlingRule("instagram_api", 15, 60),
            ThrottlingRule("twitter_api", 15, 60),
            
            # APIs de IA
            ThrottlingRule("openai_api", 20, 60),
            ThrottlingRule("stability_api", 10, 60),
            ThrottlingRule("elevenlabs_api", 10, 60),
            
            # Tipos de tareas específicas
            ThrottlingRule("content_creation", 5, 60),
            ThrottlingRule("video_rendering", 2, 60),
            ThrottlingRule("batch_upload", 5, 300)
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: ThrottlingRule) -> None:
        """
        Añade o actualiza una regla de limitación.
        
        Args:
            rule: Regla a añadir
        """
        with self.lock:
            self.rules[rule.key] = rule
            # Inicializar estadísticas si no existen
            if rule.key not in self.stats:
                self.stats[rule.key] = {
                    'total_requests': 0,
                    'throttled_requests': 0,
                    'bypassed_requests': 0,
                    'last_throttled': None
                }
    
    def remove_rule(self, key: str) -> bool:
        """
        Elimina una regla de limitación.
        
        Args:
            key: Identificador de la regla
            
        Returns:
            bool: True si se eliminó, False si no existía
        """
        with self.lock:
            if key in self.rules:
                del self.rules[key]
                return True
            return False
    
    def _get_applicable_rule(self, task: Task) -> Optional[ThrottlingRule]:
        """
        Determina la regla aplicable para una tarea.
        
        Args:
            task: Tarea a evaluar
            
        Returns:
            ThrottlingRule o None si no hay regla aplicable
        """
        # Intentar encontrar regla específica por tipo de tarea
        if task.task_type in self.rules:
            return self.rules[task.task_type]
        
        # Buscar regla por API si está en los datos de la tarea
        if 'api' in task.data and task.data['api'] in self.rules:
            return self.rules[task.data['api']]
        
        # Regla por defecto
        if 'default' in self.rules:
            return self.rules['default']
        
        return None
    
    def _is_rate_limited(self, rule: ThrottlingRule, task: Task) -> Tuple[bool, int]:
        """
        Verifica si una tarea debe ser limitada según la regla.
        
        Args:
            rule: Regla a aplicar
            task: Tarea a evaluar
            
        Returns:
            Tupla (está_limitada, tiempo_espera)
        """
        with self.lock:
            # Actualizar estadísticas
            self.stats[rule.key]['total_requests'] += 1
            
            # Verificar bypass por prioridad
            if rule.priority_bypass and hasattr(task, 'priority') and task.priority.value >= 8:
                self.stats[rule.key]['bypassed_requests'] += 1
                return False, 0
            
            # Obtener historial de ejecuciones para esta regla
            history = self.execution_history[rule.key]
            
            # Limpiar entradas antiguas
            current_time = time.time()
            cutoff_time = current_time - rule.time_window
            
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Verificar si se excede el límite
            if len(history) >= rule.rate_limit:
                # Calcular tiempo de espera
                if history:
                    # Tiempo hasta que la ejecución más antigua salga de la ventana
                    wait_time = int(history[0] + rule.time_window - current_time)
                    # Añadir período de enfriamiento si está configurado
                    wait_time += rule.cooldown_period
                else:
                    wait_time = rule.cooldown_period
                
                # Actualizar estadísticas
                self.stats[rule.key]['throttled_requests'] += 1
                self.stats[rule.key]['last_throttled'] = current_time
                
                return True, max(0, wait_time)
            
            # No limitada, registrar esta ejecución
            history.append(current_time)
            return False, 0
    
    def process_task_pre_execution(self, task: Task, context: Dict[str, Any]) -> Optional[Task]:
        """
        Verifica si una tarea debe ser limitada antes de su ejecución.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            La tarea si puede ejecutarse, None si debe ser limitada
        """
        # Obtener regla aplicable
        rule = self._get_applicable_rule(task)
        
        if not rule:
            # Sin regla, permitir ejecución
            return task
        
        # Verificar limitación
        is_limited, wait_time = self._is_rate_limited(rule, task)
        
        if is_limited:
            # Registrar limitación
            logger.info(f"Tarea {task.task_id} ({task.task_type}) limitada por regla {rule.key}. "
                       f"Espera recomendada: {wait_time}s")
            
            # Añadir información al contexto para posible reintento
            if 'throttling' not in context:
                context['throttling'] = {}
            
            context['throttling'].update({
                'limited': True,
                'rule': rule.key,
                'wait_time': wait_time,
                'timestamp': time.time()
            })
            
            # Si la tarea tiene un callback de rechazo, ejecutarlo
            if hasattr(task, 'on_reject') and callable(task.on_reject):
                try:
                    task.on_reject(task, "rate_limited", wait_time)
                except Exception as e:
                    logger.error(f"Error en callback on_reject: {str(e)}")
            
            return None
        
        return task
    
    def process_task_post_execution(self, task: Task, result: Any, error: Optional[str],
                                   context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Procesa el resultado después de la ejecución.
        
        Args:
            task: Tarea ejecutada
            result: Resultado de la ejecución
            error: Error si la tarea falló
            context: Contexto adicional
            
        Returns:
            Tupla (resultado, error)
        """
        # Verificar si el error está relacionado con limitación de API
        if error and any(term in error.lower() for term in ['rate limit', 'too many requests', '429']):
            rule = self._get_applicable_rule(task)
            
            if rule:
                # Registrar como limitación no detectada previamente
                with self.lock:
                    self.stats[rule.key]['throttled_requests'] += 1
                    self.stats[rule.key]['last_throttled'] = time.time()
                
                logger.warning(f"Detectada limitación de API en tarea {task.task_id} ({task.task_type}). "
                              f"Considere ajustar la regla {rule.key}.")
        
        return result, error
    
    def get_stats(self, rule_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de limitación.
        
        Args:
            rule_key: Clave de regla específica o None para todas
            
        Returns:
            Diccionario con estadísticas
        """
        with self.lock:
            if rule_key:
                if rule_key in self.stats:
                    return {rule_key: self.stats[rule_key]}
                return {}
            
            return {k: v.copy() for k, v in self.stats.items()}
    
    def get_current_usage(self, rule_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene el uso actual de cada regla.
        
        Args:
            rule_key: Clave de regla específica o None para todas
            
        Returns:
            Diccionario con uso actual
        """
        with self.lock:
            result = {}
            
            keys_to_check = [rule_key] if rule_key else self.rules.keys()
            
            for key in keys_to_check:
                if key in self.rules and key in self.execution_history:
                    # Limpiar entradas antiguas
                    history = self.execution_history[key]
                    current_time = time.time()
                    cutoff_time = current_time - self.rules[key].time_window
                    
                    while history and history[0] < cutoff_time:
                        history.popleft()
                    
                    # Calcular uso
                    rule = self.rules[key]
                    usage = len(history)
                    percentage = (usage / rule.rate_limit) * 100 if rule.rate_limit > 0 else 0
                    
                    result[key] = {
                        'current_usage': usage,
                        'limit': rule.rate_limit,
                        'window': rule.time_window,
                        'percentage': percentage,
                        'available': rule.rate_limit - usage
                    }
            
            return result
    
    def reset_stats(self, rule_key: Optional[str] = None) -> None:
        """
        Reinicia las estadísticas de limitación.
        
        Args:
            rule_key: Clave de regla específica o None para todas
        """
        with self.lock:
            if rule_key:
                if rule_key in self.stats:
                    self.stats[rule_key] = {
                        'total_requests': 0,
                        'throttled_requests': 0,
                        'bypassed_requests': 0,
                        'last_throttled': None
                    }
            else:
                for key in self.stats:
                    self.stats[key] = {
                        'total_requests': 0,
                        'throttled_requests': 0,
                        'bypassed_requests': 0,
                        'last_throttled': None
                    }
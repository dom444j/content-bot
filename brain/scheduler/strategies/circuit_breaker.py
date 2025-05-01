"""
Implementación del patrón Circuit Breaker para el sistema de planificación.

Este módulo proporciona una estrategia que implementa el patrón Circuit Breaker
para prevenir fallos en cascada, proteger recursos externos y mejorar la resiliencia
del sistema ante fallos temporales o permanentes en servicios dependientes.
"""

import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
import threading
import statistics

from ..core.task_model import Task, TaskStatus
from .base_strategy import BaseStrategy

logger = logging.getLogger('Scheduler.Strategies.CircuitBreaker')

class CircuitState(Enum):
    """
    Estados posibles del Circuit Breaker.
    """
    CLOSED = 'closed'      # Funcionamiento normal, las solicitudes pasan
    OPEN = 'open'          # Circuito abierto, las solicitudes fallan rápido
    HALF_OPEN = 'half_open'  # Estado de prueba, permite algunas solicitudes para verificar recuperación


class CircuitBreaker:
    """
    Implementación del patrón Circuit Breaker.
    
    Atributos:
        name: Nombre identificativo del circuit breaker
        failure_threshold: Número de fallos para abrir el circuito
        success_threshold: Número de éxitos en estado half-open para cerrar el circuito
        timeout: Tiempo en segundos que el circuito permanece abierto antes de pasar a half-open
        window_size: Tamaño de la ventana para calcular la tasa de fallos
        error_rate_threshold: Porcentaje de fallos para abrir el circuito (alternativa a failure_threshold)
    """
    
    def __init__(self, 
                name: str,
                failure_threshold: int = 5,
                success_threshold: int = 3,
                timeout: float = 60.0,
                window_size: int = 10,
                error_rate_threshold: float = 0.5):
        """
        Inicializa un circuit breaker.
        
        Args:
            name: Nombre identificativo
            failure_threshold: Número de fallos consecutivos para abrir el circuito
            success_threshold: Número de éxitos en half-open para cerrar
            timeout: Tiempo en segundos que el circuito permanece abierto
            window_size: Tamaño de la ventana para calcular tasa de fallos
            error_rate_threshold: Porcentaje de fallos para abrir el circuito (0.0-1.0)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.window_size = window_size
        self.error_rate_threshold = error_rate_threshold
        
        # Estado actual
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        
        # Contadores
        self.failure_count = 0
        self.success_count = 0
        
        # Historial para ventana deslizante
        self.execution_history = []  # Lista de tuplas (timestamp, success)
        
        # Estadísticas
        self.total_failures = 0
        self.total_successes = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # Lock para operaciones thread-safe
        self.lock = threading.RLock()
        
        logger.debug(f"Circuit Breaker '{name}' inicializado en estado {self.state.value}")
    
    def allow_request(self) -> bool:
        """
        Determina si una solicitud debe ser permitida según el estado actual.
        
        Returns:
            True si la solicitud puede proceder, False en caso contrario
        """
        with self.lock:
            current_time = datetime.now()
            
            # Si el circuito está cerrado, permitir solicitud
            if self.state == CircuitState.CLOSED:
                return True
            
            # Si el circuito está abierto, verificar si ha pasado el timeout
            elif self.state == CircuitState.OPEN:
                if (current_time - self.last_state_change).total_seconds() >= self.timeout:
                    # Transición a half-open
                    self._transition_to_half_open()
                    return True  # Permitir una solicitud de prueba
                return False  # Circuito abierto, rechazar solicitud
            
            # Si el circuito está half-open, permitir solicitudes limitadas
            elif self.state == CircuitState.HALF_OPEN:
                # En half-open, permitimos un número limitado de solicitudes
                # para probar si el servicio se ha recuperado
                return True
            
            return False
    
    def record_success(self) -> None:
        """
        Registra una ejecución exitosa.
        """
        with self.lock:
            current_time = datetime.now()
            
            # Actualizar historial
            self.execution_history.append((current_time, True))
            self._trim_history()
            
            # Actualizar contadores
            self.total_successes += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # Actualizar estado según el estado actual
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            
            logger.debug(f"Circuit Breaker '{self.name}': Éxito registrado, estado actual {self.state.value}")
    
    def record_failure(self) -> None:
        """
        Registra una ejecución fallida.
        """
        with self.lock:
            current_time = datetime.now()
            
            # Actualizar historial
            self.execution_history.append((current_time, False))
            self._trim_history()
            
            # Actualizar contadores
            self.total_failures += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Actualizar estado según el estado actual
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                
                # Verificar si debemos abrir el circuito
                if self._should_trip():
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Cualquier fallo en half-open vuelve a abrir el circuito
                self._transition_to_open()
            
            logger.debug(f"Circuit Breaker '{self.name}': Fallo registrado, estado actual {self.state.value}")
    
    def _should_trip(self) -> bool:
        """
        Determina si el circuito debe abrirse basado en fallos o tasa de error.
        
        Returns:
            True si el circuito debe abrirse
        """
        # Verificar fallos consecutivos
        if self.consecutive_failures >= self.failure_threshold:
            return True
        
        # Verificar tasa de error en la ventana
        if len(self.execution_history) >= self.window_size:
            # Calcular tasa de error en la ventana actual
            recent_results = [success for _, success in self.execution_history[-self.window_size:]]
            failure_count = recent_results.count(False)
            error_rate = failure_count / len(recent_results)
            
            if error_rate >= self.error_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self) -> None:
        """
        Transiciona el circuito al estado abierto.
        """
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit Breaker '{self.name}' abierto debido a fallos excesivos")
    
    def _transition_to_half_open(self) -> None:
        """
        Transiciona el circuito al estado semi-abierto.
        """
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit Breaker '{self.name}' en estado semi-abierto, probando recuperación")
    
    def _transition_to_closed(self) -> None:
        """
        Transiciona el circuito al estado cerrado.
        """
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit Breaker '{self.name}' cerrado, servicio recuperado")
    
    def _trim_history(self) -> None:
        """
        Elimina entradas antiguas del historial.
        """
        # Mantener solo las últimas window_size * 2 entradas para análisis
        if len(self.execution_history) > self.window_size * 2:
            self.execution_history = self.execution_history[-self.window_size * 2:]
    
    def get_state(self) -> CircuitState:
        """
        Obtiene el estado actual del circuit breaker.
        
        Returns:
            Estado actual
        """
        with self.lock:
            return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del circuit breaker.
        
        Returns:
            Diccionario con métricas
        """
        with self.lock:
            # Calcular tasa de error reciente
            recent_error_rate = 0
            if self.execution_history:
                recent_results = [success for _, success in self.execution_history[-self.window_size:]]
                if recent_results:
                    failure_count = recent_results.count(False)
                    recent_error_rate = failure_count / len(recent_results)
            
            # Calcular tiempo en estado actual
            time_in_state = (datetime.now() - self.last_state_change).total_seconds()
            
            return {
                'name': self.name,
                'state': self.state.value,
                'time_in_state': time_in_state,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'consecutive_failures': self.consecutive_failures,
                'consecutive_successes': self.consecutive_successes,
                'recent_error_rate': recent_error_rate,
                'failure_threshold': self.failure_threshold,
                'success_threshold': self.success_threshold,
                'timeout': self.timeout,
                'window_size': self.window_size,
                'error_rate_threshold': self.error_rate_threshold
            }
    
    def reset(self) -> None:
        """
        Reinicia el circuit breaker a su estado inicial.
        """
        with self.lock:
            self.state = CircuitState.CLOSED
            self.last_state_change = datetime.now()
            self.failure_count = 0
            self.success_count = 0
            self.execution_history = []
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            logger.info(f"Circuit Breaker '{self.name}' reiniciado a estado inicial")


class CircuitBreakerStrategy(BaseStrategy):
    """
    Estrategia que implementa el patrón Circuit Breaker para tareas.
    
    Esta estrategia protege recursos externos y previene fallos en cascada
    al detectar patrones de fallos y evitar ejecuciones que probablemente fallarán.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa la estrategia de Circuit Breaker.
        
        Args:
            config: Configuración específica para la estrategia
        """
        super().__init__(config)
        
        # Circuit breakers por tipo de recurso
        self.resource_breakers: Dict[str, CircuitBreaker] = {}
        
        # Circuit breakers por tipo de tarea
        self.task_type_breakers: Dict[str, CircuitBreaker] = {}
        
        # Cargar circuit breakers desde configuración
        self._load_breakers_from_config()
        
        # Si no hay circuit breakers configurados, usar valores predeterminados
        if not self.resource_breakers and not self.task_type_breakers:
            self._set_default_breakers()
        
        logger.info(f"CircuitBreakerStrategy inicializada con {len(self.resource_breakers)} breakers por recurso "
                   f"y {len(self.task_type_breakers)} por tipo de tarea")
    
    def _load_breakers_from_config(self) -> None:
        """
        Carga circuit breakers desde la configuración.
        """
        if not self.config:
            return
        
        # Cargar circuit breakers por recurso
        for breaker_config in self.config.get('resource_breakers', []):
            try:
                resource_id = breaker_config.get('resource_id')
                if not resource_id:
                    logger.warning("Circuit breaker sin recurso especificado, ignorando")
                    continue
                
                breaker = CircuitBreaker(
                    name=f"resource:{resource_id}",
                    failure_threshold=breaker_config.get('failure_threshold', 5),
                    success_threshold=breaker_config.get('success_threshold', 3),
                    timeout=breaker_config.get('timeout', 60.0),
                    window_size=breaker_config.get('window_size', 10),
                    error_rate_threshold=breaker_config.get('error_rate_threshold', 0.5)
                )
                
                self.resource_breakers[resource_id] = breaker
                logger.debug(f"Circuit breaker cargado para recurso '{resource_id}'")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar circuit breaker por recurso: {str(e)}")
        
        # Cargar circuit breakers por tipo de tarea
        for breaker_config in self.config.get('task_type_breakers', []):
            try:
                task_type = breaker_config.get('task_type')
                if not task_type:
                    logger.warning("Circuit breaker sin tipo de tarea especificado, ignorando")
                    continue
                
                breaker = CircuitBreaker(
                    name=f"task_type:{task_type}",
                    failure_threshold=breaker_config.get('failure_threshold', 5),
                    success_threshold=breaker_config.get('success_threshold', 3),
                    timeout=breaker_config.get('timeout', 60.0),
                    window_size=breaker_config.get('window_size', 10),
                    error_rate_threshold=breaker_config.get('error_rate_threshold', 0.5)
                )
                
                self.task_type_breakers[task_type] = breaker
                logger.debug(f"Circuit breaker cargado para tipo de tarea '{task_type}'")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar circuit breaker por tipo de tarea: {str(e)}")
    
    def _set_default_breakers(self) -> None:
        """
        Configura circuit breakers predeterminados.
        """
        # Circuit breakers por recurso (APIs externas)
        self.resource_breakers = {
            # YouTube API (muy sensible a fallos)
            "youtube_api": CircuitBreaker(
                name="resource:youtube_api",
                failure_threshold=3,
                success_threshold=2,
                timeout=120.0,
                window_size=10,
                error_rate_threshold=0.3
            ),
            
            # TikTok API
            "tiktok_api": CircuitBreaker(
                name="resource:tiktok_api",
                failure_threshold=4,
                success_threshold=3,
                timeout=90.0,
                window_size=10,
                error_rate_threshold=0.4
            ),
            
            # Instagram API
            "instagram_api": CircuitBreaker(
                name="resource:instagram_api",
                failure_threshold=4,
                success_threshold=3,
                timeout=90.0,
                window_size=10,
                error_rate_threshold=0.4
            ),
            
            # OpenAI API
            "openai_api": CircuitBreaker(
                name="resource:openai_api",
                failure_threshold=5,
                success_threshold=3,
                timeout=60.0,
                window_size=15,
                error_rate_threshold=0.5
            ),
            
            # Base de datos
            "database": CircuitBreaker(
                name="resource:database",
                failure_threshold=3,
                success_threshold=5,
                timeout=30.0,
                window_size=10,
                error_rate_threshold=0.2
            )
        }
        
        # Circuit breakers por tipo de tarea
        self.task_type_breakers = {
            # Publicaciones (críticas, sensibles a fallos)
            "publish": CircuitBreaker(
                name="task_type:publish",
                failure_threshold=3,
                success_threshold=3,
                timeout=300.0,  # 5 minutos
                window_size=10,
                error_rate_threshold=0.3
            ),
            
            # Análisis (menos crítico)
            "analysis": CircuitBreaker(
                name="task_type:analysis",
                failure_threshold=5,
                success_threshold=2,
                timeout=120.0,  # 2 minutos
                window_size=15,
                error_rate_threshold=0.5
            ),
            
            # Creación de contenido (equilibrado)
            "content_creation": CircuitBreaker(
                name="task_type:content_creation",
                failure_threshold=4,
                success_threshold=3,
                timeout=180.0,  # 3 minutos
                window_size=10,
                error_rate_threshold=0.4
            )
        }
    
    def should_apply(self, task: Task, context: Dict[str, Any] = None) -> bool:
        """
        Determina si la estrategia debe aplicarse a una tarea.
        
        Args:
            task: Tarea a evaluar
            context: Contexto adicional
            
        Returns:
            True si la estrategia debe aplicarse
        """
        # Solo aplicar a tareas que están a punto de ejecutarse
        if task.status != TaskStatus.SCHEDULED:
            return False
        
        # No aplicar a tareas que explícitamente lo deshabilitan
        if task.metadata.get('skip_circuit_breaker', False):
            return False
        
        # Verificar si hay circuit breakers aplicables
        has_applicable_breakers = False
        
        # Verificar circuit breaker por tipo de tarea
        if task.task_type in self.task_type_breakers:
            has_applicable_breakers = True
        
        # Verificar circuit breaker por recurso
        resource_id = task.data.get('resource_id') or task.metadata.get('resource_id')
        if resource_id and resource_id in self.resource_breakers:
            has_applicable_breakers = True
        
        # Verificar circuit breaker por API
        api_name = task.data.get('api_name') or task.metadata.get('api_name')
        if api_name and api_name in self.resource_breakers:
            has_applicable_breakers = True
        
        return has_applicable_breakers
    
    def apply(self, task: Task, context: Dict[str, Any] = None) -> Task:
        """
        Aplica la estrategia de Circuit Breaker a una tarea.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            Tarea posiblemente modificada según estado de los circuit breakers
        """
        context = context or {}
        
        # Verificar si podemos proceder con la tarea
        can_proceed, blocking_breaker = self._check_circuit_breakers(task)
        
        if can_proceed:
            # Registrar la tarea como pendiente de verificación de resultado
            if 'pending_circuit_breakers' not in task.metadata:
                task.metadata['pending_circuit_breakers'] = []
            
            # Añadir breakers aplicables para actualizar después de la ejecución
            applicable_breakers = self._get_applicable_breakers(task)
            for breaker_type, breaker_name in applicable_breakers:
                task.metadata['pending_circuit_breakers'].append({
                    'type': breaker_type,
                    'name': breaker_name
                })
            
            return task
        
        # No podemos proceder, marcar la tarea como fallida
        original_status = task.status
        task.status = TaskStatus.FAILED
        
        # Registrar información en metadatos
        if 'circuit_breaker_history' not in task.metadata:
            task.metadata['circuit_breaker_history'] = []
        
        task.metadata['circuit_breaker_history'].append({
            'timestamp': datetime.now().isoformat(),
            'original_status': original_status.value if isinstance(original_status, Enum) else original_status,
            'breaker_type': blocking_breaker.get('type'),
            'breaker_name': blocking_breaker.get('name'),
            'breaker_state': blocking_breaker.get('state')
        })
        
        # Añadir mensaje de error
        task.error = f"Tarea bloqueada por circuit breaker ({blocking_breaker.get('type')}:{blocking_breaker.get('name')}) en estado {blocking_breaker.get('state')}"
        
        logger.info(f"Tarea {task.task_id} ({task.task_type}) bloqueada por circuit breaker "
                   f"({blocking_breaker.get('type')}:{blocking_breaker.get('name')}) "
                   f"en estado {blocking_breaker.get('state')}")
        
        return task
    
        def _check_circuit_breakers(self, task: Task) -> Tuple[bool, Dict[str, str]]:
        """
        Verifica todos los circuit breakers aplicables a una tarea.
        
        Args:
            task: Tarea a verificar
            
        Returns:
            Tupla (puede_proceder, info_breaker_bloqueante)
        """
        # Obtener breakers aplicables
        applicable_breakers = self._get_applicable_breakers(task)
        
        # Verificar cada breaker
        for breaker_type, breaker_name in applicable_breakers:
            breaker = self._get_breaker(breaker_type, breaker_name)
            
            if breaker and not breaker.allow_request():
                # Si algún breaker no permite la solicitud, devolver False
                return False, {
                    'type': breaker_type,
                    'name': breaker_name,
                    'state': breaker.get_state().value
                }
        
        # Todos los breakers permiten la solicitud
        return True, {}
    
    def _get_applicable_breakers(self, task: Task) -> List[Tuple[str, str]]:
        """
        Obtiene todos los circuit breakers aplicables a una tarea.
        
        Args:
            task: Tarea a verificar
            
        Returns:
            Lista de tuplas (tipo_breaker, nombre_breaker)
        """
        result = []
        
        # Verificar breaker por tipo de tarea
        if task.task_type in self.task_type_breakers:
            result.append(('task_type', task.task_type))
        
        # Verificar breaker por recurso
        resource_id = task.data.get('resource_id') or task.metadata.get('resource_id')
        if resource_id and resource_id in self.resource_breakers:
            result.append(('resource', resource_id))
        
        # Verificar breaker por API
        api_name = task.data.get('api_name') or task.metadata.get('api_name')
        if api_name and api_name in self.resource_breakers:
            result.append(('resource', api_name))
        
        return result
    
    def _get_breaker(self, breaker_type: str, breaker_name: str) -> Optional[CircuitBreaker]:
        """
        Obtiene un circuit breaker específico.
        
        Args:
            breaker_type: Tipo de breaker ('task_type' o 'resource')
            breaker_name: Nombre del breaker
            
        Returns:
            CircuitBreaker o None si no existe
        """
        if breaker_type == 'task_type':
            return self.task_type_breakers.get(breaker_name)
        elif breaker_type == 'resource':
            return self.resource_breakers.get(breaker_name)
        return None
    
    def update_task_result(self, task: Task, success: bool) -> None:
        """
        Actualiza los circuit breakers basado en el resultado de una tarea.
        
        Args:
            task: Tarea ejecutada
            success: True si la ejecución fue exitosa, False en caso contrario
        """
        # Verificar si hay circuit breakers pendientes de actualización
        pending_breakers = task.metadata.get('pending_circuit_breakers', [])
        
        for breaker_info in pending_breakers:
            breaker_type = breaker_info.get('type')
            breaker_name = breaker_info.get('name')
            
            breaker = self._get_breaker(breaker_type, breaker_name)
            if not breaker:
                continue
            
            # Actualizar el circuit breaker según el resultado
            if success:
                breaker.record_success()
                logger.debug(f"Circuit breaker {breaker_type}:{breaker_name} actualizado con éxito")
            else:
                breaker.record_failure()
                logger.debug(f"Circuit breaker {breaker_type}:{breaker_name} actualizado con fallo")
        
        # Limpiar los circuit breakers pendientes
        if 'pending_circuit_breakers' in task.metadata:
            task.metadata['pending_circuit_breakers'] = []
    
    def add_resource_breaker(self, resource_id: str, 
                           failure_threshold: int = 5,
                           success_threshold: int = 3,
                           timeout: float = 60.0,
                           window_size: int = 10,
                           error_rate_threshold: float = 0.5) -> CircuitBreaker:
        """
        Añade un nuevo circuit breaker para un recurso.
        
        Args:
            resource_id: ID del recurso
            failure_threshold: Número de fallos para abrir el circuito
            success_threshold: Número de éxitos para cerrar el circuito
            timeout: Tiempo en segundos que el circuito permanece abierto
            window_size: Tamaño de la ventana para calcular tasa de fallos
            error_rate_threshold: Porcentaje de fallos para abrir el circuito
            
        Returns:
            El circuit breaker creado
        """
        breaker = CircuitBreaker(
            name=f"resource:{resource_id}",
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            window_size=window_size,
            error_rate_threshold=error_rate_threshold
        )
        
        self.resource_breakers[resource_id] = breaker
        logger.info(f"Circuit breaker añadido para recurso '{resource_id}'")
        
        return breaker
    
    def add_task_type_breaker(self, task_type: str,
                            failure_threshold: int = 5,
                            success_threshold: int = 3,
                            timeout: float = 60.0,
                            window_size: int = 10,
                            error_rate_threshold: float = 0.5) -> CircuitBreaker:
        """
        Añade un nuevo circuit breaker para un tipo de tarea.
        
        Args:
            task_type: Tipo de tarea
            failure_threshold: Número de fallos para abrir el circuito
            success_threshold: Número de éxitos para cerrar el circuito
            timeout: Tiempo en segundos que el circuito permanece abierto
            window_size: Tamaño de la ventana para calcular tasa de fallos
            error_rate_threshold: Porcentaje de fallos para abrir el circuito
            
        Returns:
            El circuit breaker creado
        """
        breaker = CircuitBreaker(
            name=f"task_type:{task_type}",
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            window_size=window_size,
            error_rate_threshold=error_rate_threshold
        )
        
        self.task_type_breakers[task_type] = breaker
        logger.info(f"Circuit breaker añadido para tipo de tarea '{task_type}'")
        
        return breaker
    
    def remove_resource_breaker(self, resource_id: str) -> bool:
        """
        Elimina un circuit breaker para un recurso.
        
        Args:
            resource_id: ID del recurso
            
        Returns:
            True si se eliminó, False si no existía
        """
        if resource_id in self.resource_breakers:
            del self.resource_breakers[resource_id]
            logger.info(f"Circuit breaker eliminado para recurso '{resource_id}'")
            return True
        return False
    
    def remove_task_type_breaker(self, task_type: str) -> bool:
        """
        Elimina un circuit breaker para un tipo de tarea.
        
        Args:
            task_type: Tipo de tarea
            
        Returns:
            True si se eliminó, False si no existía
        """
        if task_type in self.task_type_breakers:
            del self.task_type_breakers[task_type]
            logger.info(f"Circuit breaker eliminado para tipo de tarea '{task_type}'")
            return True
        return False
    
    def get_breaker_metrics(self, breaker_type: str, breaker_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene métricas de un circuit breaker específico.
        
        Args:
            breaker_type: Tipo de breaker ('task_type' o 'resource')
            breaker_name: Nombre del breaker
            
        Returns:
            Diccionario con métricas o None si no existe
        """
        breaker = self._get_breaker(breaker_type, breaker_name)
        if not breaker:
            return None
        
        return breaker.get_metrics()
    
    def get_all_breaker_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtiene métricas de todos los circuit breakers.
        
        Returns:
            Diccionario con métricas agrupadas por tipo
        """
        result = {
            'resource': [],
            'task_type': []
        }
        
        # Métricas de breakers por recurso
        for resource_id, breaker in self.resource_breakers.items():
            result['resource'].append(breaker.get_metrics())
        
        # Métricas de breakers por tipo de tarea
        for task_type, breaker in self.task_type_breakers.items():
            result['task_type'].append(breaker.get_metrics())
        
        return result
    
    def reset_breaker(self, breaker_type: str, breaker_name: str) -> bool:
        """
        Reinicia un circuit breaker específico.
        
        Args:
            breaker_type: Tipo de breaker ('task_type' o 'resource')
            breaker_name: Nombre del breaker
            
        Returns:
            True si se reinició, False si no existe
        """
        breaker = self._get_breaker(breaker_type, breaker_name)
        if not breaker:
            return False
        
        breaker.reset()
        logger.info(f"Circuit breaker {breaker_type}:{breaker_name} reiniciado")
        return True
    
    def reset_all_breakers(self) -> None:
        """
        Reinicia todos los circuit breakers.
        """
        # Reiniciar breakers por recurso
        for resource_id, breaker in self.resource_breakers.items():
            breaker.reset()
        
        # Reiniciar breakers por tipo de tarea
        for task_type, breaker in self.task_type_breakers.items():
            breaker.reset()
        
        logger.info("Todos los circuit breakers han sido reiniciados")
    
    def export_config(self) -> Dict[str, Any]:
        """
        Exporta la configuración actual de circuit breakers.
        
        Returns:
            Diccionario con configuración exportable
        """
        config = {
            'resource_breakers': [],
            'task_type_breakers': []
        }
        
        # Exportar breakers por recurso
        for resource_id, breaker in self.resource_breakers.items():
            config['resource_breakers'].append({
                'resource_id': resource_id,
                'failure_threshold': breaker.failure_threshold,
                'success_threshold': breaker.success_threshold,
                'timeout': breaker.timeout,
                'window_size': breaker.window_size,
                'error_rate_threshold': breaker.error_rate_threshold
            })
        
        # Exportar breakers por tipo de tarea
        for task_type, breaker in self.task_type_breakers.items():
            config['task_type_breakers'].append({
                'task_type': task_type,
                'failure_threshold': breaker.failure_threshold,
                'success_threshold': breaker.success_threshold,
                'timeout': breaker.timeout,
                'window_size': breaker.window_size,
                'error_rate_threshold': breaker.error_rate_threshold
            })
        
        return config
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """
        Importa configuración de circuit breakers.
        
        Args:
            config: Diccionario con configuración
        """
        # Limpiar configuración actual
        self.resource_breakers = {}
        self.task_type_breakers = {}
        
        # Importar breakers por recurso
        for breaker_config in config.get('resource_breakers', []):
            try:
                resource_id = breaker_config.get('resource_id')
                if not resource_id:
                    logger.warning("Circuit breaker sin recurso especificado, ignorando")
                    continue
                
                self.add_resource_breaker(
                    resource_id=resource_id,
                    failure_threshold=breaker_config.get('failure_threshold', 5),
                    success_threshold=breaker_config.get('success_threshold', 3),
                    timeout=breaker_config.get('timeout', 60.0),
                    window_size=breaker_config.get('window_size', 10),
                    error_rate_threshold=breaker_config.get('error_rate_threshold', 0.5)
                )
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al importar circuit breaker por recurso: {str(e)}")
        
        # Importar breakers por tipo de tarea
        for breaker_config in config.get('task_type_breakers', []):
            try:
                task_type = breaker_config.get('task_type')
                if not task_type:
                    logger.warning("Circuit breaker sin tipo de tarea especificado, ignorando")
                    continue
                
                self.add_task_type_breaker(
                    task_type=task_type,
                    failure_threshold=breaker_config.get('failure_threshold', 5),
                    success_threshold=breaker_config.get('success_threshold', 3),
                    timeout=breaker_config.get('timeout', 60.0),
                    window_size=breaker_config.get('window_size', 10),
                    error_rate_threshold=breaker_config.get('error_rate_threshold', 0.5)
                )
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al importar circuit breaker por tipo de tarea: {str(e)}")
        
        logger.info(f"Configuración importada: {len(self.resource_breakers)} breakers por recurso, "
                   f"{len(self.task_type_breakers)} por tipo de tarea")
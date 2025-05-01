"""
Estrategia de priorización para el sistema de planificación.

Este módulo implementa diferentes políticas para ajustar dinámicamente
la prioridad de las tareas según diversos factores como tiempo de espera,
importancia del canal, tipo de contenido, etc.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta

from ..core.task_model import Task, TaskStatus, TaskPriority
from .base_strategy import BaseStrategy

logger = logging.getLogger('Scheduler.Strategies.Priority')

class PriorityRule:
    """
    Regla para ajustar la prioridad de una tarea según criterios específicos.
    
    Atributos:
        name: Nombre descriptivo de la regla
        condition: Función que evalúa si la regla aplica
        adjustment: Valor de ajuste de prioridad (positivo o negativo)
        description: Descripción detallada de la regla
    """
    
    def __init__(self, 
                name: str,
                condition: Callable[[Task, Dict[str, Any]], bool],
                adjustment: int,
                description: str = None):
        """
        Inicializa una regla de prioridad.
        
        Args:
            name: Nombre descriptivo de la regla
            condition: Función que evalúa si la regla aplica
            adjustment: Valor de ajuste de prioridad (positivo o negativo)
            description: Descripción detallada de la regla
        """
        self.name = name
        self.condition = condition
        self.adjustment = adjustment
        self.description = description or f"Regla de prioridad '{name}'"
    
    def applies_to(self, task: Task, context: Dict[str, Any] = None) -> bool:
        """
        Verifica si la regla aplica a una tarea específica.
        
        Args:
            task: Tarea a evaluar
            context: Contexto adicional
            
        Returns:
            True si la regla aplica, False en caso contrario
        """
        context = context or {}
        try:
            return self.condition(task, context)
        except Exception as e:
            logger.error(f"Error al evaluar regla '{self.name}': {str(e)}")
            return False
    
    def get_adjustment(self) -> int:
        """
        Obtiene el valor de ajuste de prioridad.
        
        Returns:
            Valor de ajuste (positivo o negativo)
        """
        return self.adjustment


class PriorityStrategy(BaseStrategy):
    """
    Estrategia que ajusta dinámicamente la prioridad de las tareas.
    
    Características:
    - Reglas configurables por tipo de tarea
    - Ajustes basados en tiempo de espera, importancia, etc.
    - Priorización de tareas críticas
    - Degradación gradual para tareas antiguas
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa la estrategia de priorización.
        
        Args:
            config: Configuración específica para la estrategia
        """
        super().__init__(config)
        
        # Reglas de priorización por tipo de tarea
        self.rules_by_type: Dict[str, List[PriorityRule]] = {}
        
        # Reglas globales (aplican a todas las tareas)
        self.global_rules: List[PriorityRule] = []
        
        # Cargar reglas desde configuración
        self._load_rules_from_config()
        
        # Si no hay reglas configuradas, usar reglas predeterminadas
        if not self.global_rules and not self.rules_by_type:
            self._set_default_rules()
        
        logger.info(f"PriorityStrategy inicializada con {len(self.global_rules)} reglas globales "
                   f"y reglas específicas para {len(self.rules_by_type)} tipos de tareas")
    
    def _load_rules_from_config(self) -> None:
        """
        Carga reglas de priorización desde la configuración.
        """
        if not self.config:
            return
        
        # Cargar reglas globales
        for rule_config in self.config.get('global_rules', []):
            try:
                rule_name = rule_config.get('name', 'Regla sin nombre')
                rule_adjustment = rule_config.get('adjustment', 0)
                rule_description = rule_config.get('description')
                rule_condition_type = rule_config.get('condition_type')
                
                # Crear función de condición según el tipo
                condition = self._create_condition_from_config(rule_condition_type, rule_config)
                
                if condition:
                    rule = PriorityRule(
                        name=rule_name,
                        condition=condition,
                        adjustment=rule_adjustment,
                        description=rule_description
                    )
                    self.global_rules.append(rule)
                    logger.debug(f"Regla global cargada: {rule_name}")
            
            except Exception as e:
                logger.error(f"Error al cargar regla global: {str(e)}")
        
        # Cargar reglas específicas por tipo
        for type_rules in self.config.get('type_rules', []):
            try:
                task_type = type_rules.get('task_type')
                if not task_type:
                    logger.warning("Reglas sin tipo de tarea especificado, ignorando")
                    continue
                
                rules = []
                for rule_config in type_rules.get('rules', []):
                    rule_name = rule_config.get('name', f'Regla para {task_type}')
                    rule_adjustment = rule_config.get('adjustment', 0)
                    rule_description = rule_config.get('description')
                    rule_condition_type = rule_config.get('condition_type')
                    
                    # Crear función de condición según el tipo
                    condition = self._create_condition_from_config(rule_condition_type, rule_config)
                    
                    if condition:
                        rule = PriorityRule(
                            name=rule_name,
                            condition=condition,
                            adjustment=rule_adjustment,
                            description=rule_description
                        )
                        rules.append(rule)
                
                if rules:
                    self.rules_by_type[task_type] = rules
                    logger.debug(f"Reglas cargadas para tipo '{task_type}': {len(rules)}")
            
            except Exception as e:
                logger.error(f"Error al cargar reglas para tipo: {str(e)}")
    
    def _create_condition_from_config(self, condition_type: str, rule_config: Dict[str, Any]) -> Optional[Callable]:
        """
        Crea una función de condición a partir de la configuración.
        
        Args:
            condition_type: Tipo de condición
            rule_config: Configuración de la regla
            
        Returns:
            Función de condición o None si no se puede crear
        """
        try:
            if condition_type == 'waiting_time':
                threshold_minutes = rule_config.get('threshold_minutes', 30)
                return lambda task, ctx: (datetime.now() - task.created_at).total_seconds() / 60 > threshold_minutes
            
            elif condition_type == 'retry_count':
                threshold = rule_config.get('threshold', 1)
                return lambda task, ctx: task.metadata.get('retry_count', 0) >= threshold
            
            elif condition_type == 'channel_importance':
                important_channels = rule_config.get('important_channels', [])
                return lambda task, ctx: task.data.get('channel_id') in important_channels
            
            elif condition_type == 'content_type':
                content_types = rule_config.get('content_types', [])
                return lambda task, ctx: task.data.get('content_type') in content_types
            
            elif condition_type == 'has_deadline':
                return lambda task, ctx: 'deadline' in task.metadata
            
            elif condition_type == 'approaching_deadline':
                threshold_minutes = rule_config.get('threshold_minutes', 60)
                return lambda task, ctx: ('deadline' in task.metadata and 
                                        (datetime.fromisoformat(task.metadata['deadline']) - datetime.now()).total_seconds() / 60 < threshold_minutes)
            
            elif condition_type == 'has_dependencies':
                return lambda task, ctx: bool(task.dependencies)
            
            elif condition_type == 'is_trending':
                return lambda task, ctx: task.metadata.get('is_trending', False)
            
            elif condition_type == 'custom_expression':
                expression = rule_config.get('expression')
                if not expression:
                    return None
                
                # Advertencia: eval puede ser peligroso, solo usar en entornos controlados
                # En producción, considerar alternativas más seguras
                return lambda task, ctx: eval(expression, {'task': task, 'ctx': ctx, 'datetime': datetime})
            
            logger.warning(f"Tipo de condición no soportado: {condition_type}")
            return None
        
        except Exception as e:
            logger.error(f"Error al crear condición: {str(e)}")
            return None
    
    def _set_default_rules(self) -> None:
        """
        Configura reglas predeterminadas de priorización.
        """
        # Reglas globales
        self.global_rules = [
            # Aumentar prioridad para tareas que han esperado mucho tiempo
            PriorityRule(
                name="waiting_time_boost",
                condition=lambda task, ctx: (datetime.now() - task.created_at).total_seconds() / 60 > 30,
                adjustment=1,
                description="Aumenta prioridad para tareas que han esperado más de 30 minutos"
            ),
            
            # Aumentar prioridad para tareas con reintentos
            PriorityRule(
                name="retry_boost",
                condition=lambda task, ctx: task.metadata.get('retry_count', 0) > 0,
                adjustment=1,
                description="Aumenta prioridad para tareas que han sido reintentadas"
            ),
            
            # Aumentar prioridad para tareas con deadline cercano
            PriorityRule(
                name="deadline_boost",
                condition=lambda task, ctx: ('deadline' in task.metadata and 
                                          (datetime.fromisoformat(task.metadata['deadline']) - datetime.now()).total_seconds() / 60 < 60),
                adjustment=2,
                description="Aumenta prioridad para tareas con deadline en menos de 1 hora"
            ),
            
            # Reducir prioridad para tareas muy antiguas (posiblemente obsoletas)
            PriorityRule(
                name="very_old_penalty",
                condition=lambda task, ctx: (datetime.now() - task.created_at).total_seconds() / 3600 > 24,
                adjustment=-1,
                description="Reduce prioridad para tareas creadas hace más de 24 horas"
            )
        ]
        
        # Reglas específicas por tipo
        
        # Tareas de publicación
        publish_rules = [
            # Priorizar publicaciones en horarios óptimos
            PriorityRule(
                name="optimal_time_publish",
                condition=lambda task, ctx: ('optimal_time' in task.metadata and 
                                          abs((datetime.now() - datetime.fromisoformat(task.metadata['optimal_time'])).total_seconds()) < 300),
                adjustment=3,
                description="Prioriza publicaciones cercanas a su horario óptimo (±5 min)"
            ),
            
            # Priorizar publicaciones de tendencias
            PriorityRule(
                name="trending_content_publish",
                condition=lambda task, ctx: task.metadata.get('is_trending', False),
                adjustment=2,
                description="Prioriza publicación de contenido sobre tendencias"
            )
        ]
        
        # Tareas de análisis
        analysis_rules = [
            # Reducir prioridad de análisis rutinarios
            PriorityRule(
                name="routine_analysis_penalty",
                condition=lambda task, ctx: task.metadata.get('routine', False),
                adjustment=-1,
                description="Reduce prioridad de análisis rutinarios"
            ),
            
            # Aumentar prioridad de análisis de rendimiento crítico
            PriorityRule(
                name="performance_analysis_boost",
                condition=lambda task, ctx: task.data.get('analysis_type') == 'performance' and task.metadata.get('critical', False),
                adjustment=2,
                description="Aumenta prioridad de análisis de rendimiento críticos"
            )
        ]
        
        # Tareas de creación de contenido
        content_rules = [
            # Priorizar contenido para canales importantes
            PriorityRule(
                name="important_channel_boost",
                condition=lambda task, ctx: task.data.get('channel_tier', 0) >= 2,
                adjustment=1,
                description="Prioriza contenido para canales de alto rendimiento (tier >= 2)"
            ),
            
            # Priorizar contenido sobre tendencias
            PriorityRule(
                name="trending_content_creation",
                condition=lambda task, ctx: task.metadata.get('is_trending', False),
                adjustment=2,
                description="Prioriza creación de contenido sobre tendencias"
            )
        ]
        
        # Registrar reglas específicas
        self.rules_by_type = {
            "publish": publish_rules,
            "analysis": analysis_rules,
            "content_creation": content_rules
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
        # La estrategia de priorización se aplica a todas las tareas
        # excepto las que ya están en ejecución o completadas
        if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False
        
        # No aplicar a tareas que explícitamente lo deshabilitan
        if task.metadata.get('skip_priority_adjustment', False):
            return False
        
        return True
    
    def apply(self, task: Task, context: Dict[str, Any] = None) -> Task:
        """
        Aplica la estrategia de priorización a una tarea.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            Tarea con prioridad ajustada
        """
        context = context or {}
        
        # Obtener prioridad base
        base_priority = task.priority.value if isinstance(task.priority, TaskPriority) else task.priority
        
        # Inicializar ajuste total
        total_adjustment = 0
        applied_rules = []
        
        # Aplicar reglas globales
        for rule in self.global_rules:
            if rule.applies_to(task, context):
                total_adjustment += rule.get_adjustment()
                applied_rules.append(rule.name)
        
        # Aplicar reglas específicas por tipo
        if task.task_type in self.rules_by_type:
            for rule in self.rules_by_type[task.task_type]:
                if rule.applies_to(task, context):
                    total_adjustment += rule.get_adjustment()
                    applied_rules.append(rule.name)
        
        # Calcular nueva prioridad (menor valor = mayor prioridad)
        # Asegurar que la prioridad se mantiene dentro de los límites válidos
        new_priority_value = max(1, min(5, base_priority - total_adjustment))
        
        # Actualizar prioridad solo si ha cambiado
        if new_priority_value != base_priority:
            # Convertir valor numérico a enum si es necesario
            if isinstance(task.priority, TaskPriority):
                # Mapear valor a enum
                priority_map = {
                    1: TaskPriority.CRITICAL,
                    2: TaskPriority.HIGH,
                    3: TaskPriority.MEDIUM,
                    4: TaskPriority.LOW,
                    5: TaskPriority.BACKGROUND
                }
                task.priority = priority_map.get(new_priority_value, task.priority)
            else:
                # Usar valor numérico directamente
                task.priority = new_priority_value
            
            # Registrar cambio en metadatos
            if 'priority_adjustments' not in task.metadata:
                task.metadata['priority_adjustments'] = []
            
            task.metadata['priority_adjustments'].append({
                'timestamp': datetime.now().isoformat(),
                'previous': base_priority,
                'new': new_priority_value,
                'adjustment': total_adjustment,
                'rules': applied_rules
            })
            
            logger.info(f"Prioridad ajustada para tarea {task.task_id} ({task.task_type}): "
                       f"{base_priority} -> {new_priority_value} ({total_adjustment:+d}) "
                       f"por reglas: {', '.join(applied_rules)}")
        
        return task
    
    def add_global_rule(self, rule: PriorityRule) -> None:
        """
        Añade una regla global de priorización.
        
        Args:
            rule: Regla a añadir
        """
        self.global_rules.append(rule)
        logger.debug(f"Regla global añadida: {rule.name}")
    
    def add_type_rule(self, task_type: str, rule: PriorityRule) -> None:
        """
        Añade una regla específica para un tipo de tarea.
        
        Args:
            task_type: Tipo de tarea
            rule: Regla a añadir
        """
        if task_type not in self.rules_by_type:
            self.rules_by_type[task_type] = []
        
        self.rules_by_type[task_type].append(rule)
        logger.debug(f"Regla añadida para tipo '{task_type}': {rule.name}")
    
    def remove_global_rule(self, rule_name: str) -> bool:
        """
        Elimina una regla global por nombre.
        
        Args:
            rule_name: Nombre de la regla a eliminar
            
        Returns:
            True si se eliminó, False si no se encontró
        """
        for i, rule in enumerate(self.global_rules):
            if rule.name == rule_name:
                self.global_rules.pop(i)
                logger.debug(f"Regla global eliminada: {rule_name}")
                return True
        
        return False
    
    def remove_type_rule(self, task_type: str, rule_name: str) -> bool:
        """
        Elimina una regla específica de un tipo de tarea.
        
        Args:
            task_type: Tipo de tarea
            rule_name: Nombre de la regla a eliminar
            
        Returns:
            True si se eliminó, False si no se encontró
        """
        if task_type not in self.rules_by_type:
            return False
        
        for i, rule in enumerate(self.rules_by_type[task_type]):
            if rule.name == rule_name:
                self.rules_by_type[task_type].pop(i)
                logger.debug(f"Regla eliminada para tipo '{task_type}': {rule_name}")
                
                # Si no quedan reglas, eliminar la entrada del diccionario
                if not self.rules_by_type[task_type]:
                    del self.rules_by_type[task_type]
                
                return True
        
        return False
    
    def get_all_rules(self) -> Dict[str, List[PriorityRule]]:
        """
        Obtiene todas las reglas de priorización.
        
        Returns:
            Diccionario con todas las reglas (globales y por tipo)
        """
        result = {'global': self.global_rules}
        result.update(self.rules_by_type)
        return result
    
    def clear_rules(self) -> None:
        """
        Elimina todas las reglas de priorización.
        """
        self.global_rules = []
        self.rules_by_type = {}
        logger.debug("Todas las reglas de priorización han sido eliminadas")
        
        # Recargar reglas predeterminadas
        self._set_default_rules()
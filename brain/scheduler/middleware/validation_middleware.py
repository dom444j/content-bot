"""
Middleware de validación para el sistema de planificación.

Este módulo implementa un middleware que valida las tareas antes de su ejecución,
asegurando que cumplan con los requisitos y esquemas definidos.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime

from ..core.task_model import Task, TaskStatus
from .base_middleware import BaseMiddleware

logger = logging.getLogger('Scheduler.Middleware.Validation')

class ValidationRule:
    """
    Regla de validación para un campo específico de una tarea.
    
    Atributos:
        field_path: Ruta al campo en formato dot notation (ej: 'data.channel_id')
        validator: Función que valida el valor del campo
        error_message: Mensaje de error personalizado
        required: Si el campo es obligatorio
    """
    
    def __init__(self, field_path: str, validator: Callable[[Any], bool], 
                error_message: str = None, required: bool = True):
        """
        Inicializa una regla de validación.
        
        Args:
            field_path: Ruta al campo en formato dot notation
            validator: Función que valida el valor del campo
            error_message: Mensaje de error personalizado
            required: Si el campo es obligatorio
        """
        self.field_path = field_path
        self.validator = validator
        self.error_message = error_message
        self.required = required

class ValidationSchema:
    """
    Esquema de validación para un tipo específico de tarea.
    
    Atributos:
        task_type: Tipo de tarea al que aplica este esquema
        rules: Lista de reglas de validación
        description: Descripción del esquema
    """
    
    def __init__(self, task_type: str, rules: List[ValidationRule], description: str = None):
        """
        Inicializa un esquema de validación.
        
        Args:
            task_type: Tipo de tarea al que aplica este esquema
            rules: Lista de reglas de validación
            description: Descripción del esquema
        """
        self.task_type = task_type
        self.rules = rules
        self.description = description or f"Esquema de validación para tareas de tipo {task_type}"

class ValidationMiddleware(BaseMiddleware):
    """
    Middleware que valida las tareas antes de su ejecución.
    
    Características:
    - Validación basada en esquemas por tipo de tarea
    - Validadores predefinidos para tipos comunes
    - Soporte para validación personalizada
    - Registro detallado de errores de validación
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el middleware de validación.
        
        Args:
            config: Configuración específica del middleware
        """
        super().__init__(config)
        
        # Esquemas de validación por tipo de tarea
        self.schemas: Dict[str, ValidationSchema] = {}
        
        # Estadísticas de validación
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'failures_by_type': {}
        }
        
        # Cargar esquemas predefinidos
        self._load_predefined_schemas()
        
        # Cargar esquemas desde configuración
        self._load_schemas_from_config()
        
        logger.info(f"ValidationMiddleware inicializado con {len(self.schemas)} esquemas")
    
    def _load_predefined_schemas(self) -> None:
        """
        Carga esquemas de validación predefinidos para tipos comunes de tareas.
        """
        # Esquema para tareas de creación de contenido
        content_creation_schema = ValidationSchema(
            task_type="content_creation",
            rules=[
                ValidationRule(
                    field_path="data.channel_id",
                    validator=lambda v: isinstance(v, str) and len(v) > 0,
                    error_message="El ID del canal es obligatorio"
                ),
                ValidationRule(
                    field_path="data.content_type",
                    validator=lambda v: isinstance(v, str) and v in [
                        "short_video", "long_video", "post", "image", "story"
                    ],
                    error_message="Tipo de contenido no válido"
                ),
                ValidationRule(
                    field_path="data.theme",
                    validator=lambda v: isinstance(v, str) and len(v) > 0,
                    error_message="El tema del contenido es obligatorio"
                )
            ],
            description="Validación para tareas de creación de contenido"
        )
        
        # Esquema para tareas de publicación
        publish_schema = ValidationSchema(
            task_type="publish",
            rules=[
                ValidationRule(
                    field_path="data.channel_id",
                    validator=lambda v: isinstance(v, str) and len(v) > 0,
                    error_message="El ID del canal es obligatorio"
                ),
                ValidationRule(
                    field_path="data.content_id",
                    validator=lambda v: isinstance(v, str) and len(v) > 0,
                    error_message="El ID del contenido es obligatorio"
                ),
                ValidationRule(
                    field_path="data.platform",
                    validator=lambda v: isinstance(v, str) and v in [
                        "youtube", "tiktok", "instagram", "twitter", "threads", "bluesky"
                    ],
                    error_message="Plataforma no válida"
                )
            ],
            description="Validación para tareas de publicación"
        )
        
        # Esquema para tareas de análisis
        analysis_schema = ValidationSchema(
            task_type="analysis",
            rules=[
                ValidationRule(
                    field_path="data.channel_id",
                    validator=lambda v: isinstance(v, str) and len(v) > 0,
                    error_message="El ID del canal es obligatorio"
                ),
                ValidationRule(
                    field_path="data.analysis_type",
                    validator=lambda v: isinstance(v, str) and v in [
                        "performance", "engagement", "audience", "competition", "trend"
                    ],
                    error_message="Tipo de análisis no válido"
                ),
                ValidationRule(
                    field_path="data.time_range",
                    validator=lambda v: isinstance(v, dict) and "start" in v and "end" in v,
                    error_message="Rango de tiempo no válido",
                    required=False
                )
            ],
            description="Validación para tareas de análisis"
        )
        
        # Registrar esquemas predefinidos
        self.add_schema(content_creation_schema)
        self.add_schema(publish_schema)
        self.add_schema(analysis_schema)
    
    def _load_schemas_from_config(self) -> None:
        """
        Carga esquemas de validación desde la configuración.
        """
        if not self.config or 'schemas' not in self.config:
            return
        
        for schema_config in self.config.get('schemas', []):
            try:
                # Extraer información básica del esquema
                task_type = schema_config.get('task_type')
                description = schema_config.get('description')
                
                if not task_type:
                    logger.error("Esquema de validación sin tipo de tarea")
                    continue
                
                # Construir reglas
                rules = []
                for rule_config in schema_config.get('rules', []):
                    field_path = rule_config.get('field_path')
                    validator_type = rule_config.get('validator_type')
                    error_message = rule_config.get('error_message')
                    required = rule_config.get('required', True)
                    
                    if not field_path or not validator_type:
                        logger.error(f"Regla de validación incompleta para {task_type}")
                        continue
                    
                    # Crear validador según el tipo
                    validator = self._create_validator_from_config(validator_type, rule_config)
                    
                    if validator:
                        rules.append(ValidationRule(
                            field_path=field_path,
                            validator=validator,
                            error_message=error_message,
                            required=required
                        ))
                
                # Crear y registrar esquema
                if rules:
                    schema = ValidationSchema(
                        task_type=task_type,
                        rules=rules,
                        description=description
                    )
                    self.add_schema(schema)
                    logger.debug(f"Esquema cargado para {task_type} con {len(rules)} reglas")
            
            except Exception as e:
                logger.error(f"Error al cargar esquema de validación: {str(e)}")
    
    def _create_validator_from_config(self, validator_type: str, rule_config: Dict[str, Any]) -> Optional[Callable]:
        """
        Crea una función validadora a partir de la configuración.
        
        Args:
            validator_type: Tipo de validador
            rule_config: Configuración de la regla
            
        Returns:
            Función validadora o None si no se puede crear
        """
        try:
            if validator_type == 'type':
                expected_type = rule_config.get('expected_type')
                if expected_type == 'string':
                    return lambda v: isinstance(v, str)
                elif expected_type == 'integer':
                    return lambda v: isinstance(v, int)
                elif expected_type == 'number':
                    return lambda v: isinstance(v, (int, float))
                elif expected_type == 'boolean':
                    return lambda v: isinstance(v, bool)
                elif expected_type == 'array':
                    return lambda v: isinstance(v, list)
                elif expected_type == 'object':
                    return lambda v: isinstance(v, dict)
            
            elif validator_type == 'enum':
                allowed_values = rule_config.get('allowed_values', [])
                return lambda v: v in allowed_values
            
            elif validator_type == 'regex':
                pattern = rule_config.get('pattern')
                if pattern:
                    compiled_regex = re.compile(pattern)
                    return lambda v: isinstance(v, str) and bool(compiled_regex.match(v))
            
            elif validator_type == 'range':
                min_value = rule_config.get('min')
                max_value = rule_config.get('max')
                
                if min_value is not None and max_value is not None:
                    return lambda v: min_value <= v <= max_value
                elif min_value is not None:
                    return lambda v: v >= min_value
                elif max_value is not None:
                    return lambda v: v <= max_value
            
            elif validator_type == 'length':
                min_length = rule_config.get('min')
                max_length = rule_config.get('max')
                
                if min_length is not None and max_length is not None:
                    return lambda v: hasattr(v, '__len__') and min_length <= len(v) <= max_length
                elif min_length is not None:
                    return lambda v: hasattr(v, '__len__') and len(v) >= min_length
                elif max_length is not None:
                    return lambda v: hasattr(v, '__len__') and len(v) <= max_length
            
            logger.warning(f"Tipo de validador no soportado: {validator_type}")
            return None
        
        except Exception as e:
            logger.error(f"Error al crear validador: {str(e)}")
            return None
    
    def add_schema(self, schema: ValidationSchema) -> None:
        """
        Añade o actualiza un esquema de validación.
        
        Args:
            schema: Esquema a añadir
        """
        self.schemas[schema.task_type] = schema
        
        # Inicializar estadísticas para este tipo si no existen
        if schema.task_type not in self.stats['failures_by_type']:
            self.stats['failures_by_type'][schema.task_type] = 0
    
    def remove_schema(self, task_type: str) -> bool:
        """
        Elimina un esquema de validación.
        
        Args:
            task_type: Tipo de tarea del esquema a eliminar
            
        Returns:
            bool: True si se eliminó, False si no existía
        """
        if task_type in self.schemas:
            del self.schemas[task_type]
            return True
        return False
    
    def _get_field_value(self, task: Task, field_path: str) -> Tuple[Any, bool]:
        """
        Obtiene el valor de un campo en la tarea según su ruta.
        
        Args:
            task: Tarea a evaluar
            field_path: Ruta al campo en formato dot notation
            
        Returns:
            Tupla (valor, existe)
        """
        parts = field_path.split('.')
        current = task.__dict__
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None, False
        
        return current, True
    
    def _validate_task(self, task: Task) -> Tuple[bool, List[str]]:
        """
        Valida una tarea según su esquema correspondiente.
        
        Args:
            task: Tarea a validar
            
        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        # Verificar si existe esquema para este tipo de tarea
        if task.task_type not in self.schemas:
            # Sin esquema, se considera válida
            return True, []
        
        schema = self.schemas[task.task_type]
        errors = []
        
        # Aplicar cada regla del esquema
        for rule in schema.rules:
            value, exists = self._get_field_value(task, rule.field_path)
            
            # Verificar si el campo es obligatorio
            if not exists:
                if rule.required:
                    error_msg = rule.error_message or f"Campo obligatorio '{rule.field_path}' no encontrado"
                    errors.append(error_msg)
                    continue
                
                # Si el campo existe, validar su valor
                if not rule.validator(value):
                    error_msg = rule.error_message or f"Valor inválido para '{rule.field_path}'"
                    errors.append(error_msg)
            
            # Actualizar estadísticas
            self.stats['total_validations'] += 1
            
            if errors:
                self.stats['failed_validations'] += 1
                if task.task_type in self.stats['failures_by_type']:
                    self.stats['failures_by_type'][task.task_type] += 1
                return False, errors
            
            self.stats['passed_validations'] += 1
            return True, []
        
        def process_task_pre_execution(self, task: Task, context: Dict[str, Any]) -> Optional[Task]:
            """
            Valida una tarea antes de su ejecución.
            
            Args:
                task: Tarea a procesar
                context: Contexto adicional
                
            Returns:
                La tarea si es válida, None si no pasa la validación
            """
            # Validar la tarea
            is_valid, errors = self._validate_task(task)
            
            if not is_valid:
                # Registrar errores de validación
                error_str = "; ".join(errors)
                logger.warning(f"Tarea {task.task_id} ({task.task_type}) rechazada por validación: {error_str}")
                
                # Añadir información al contexto
                if 'validation' not in context:
                    context['validation'] = {}
                
                context['validation'].update({
                    'valid': False,
                    'errors': errors,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Si la tarea tiene un callback de rechazo, ejecutarlo
                if hasattr(task, 'on_reject') and callable(task.on_reject):
                    try:
                        task.on_reject(task, "validation_failed", errors)
                    except Exception as e:
                        logger.error(f"Error en callback on_reject: {str(e)}")
                
                return None
            
            # Tarea válida, añadir información al contexto
            if 'validation' not in context:
                context['validation'] = {}
            
            context['validation'].update({
                'valid': True,
                'timestamp': datetime.now().isoformat()
            })
            
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
            # Este middleware no modifica el resultado post-ejecución
            return result, error
        
        def get_stats(self) -> Dict[str, Any]:
            """
            Obtiene estadísticas de validación.
            
            Returns:
                Diccionario con estadísticas
            """
            return self.stats.copy()
        
        def reset_stats(self) -> None:
            """
            Reinicia las estadísticas de validación.
            """
            self.stats = {
                'total_validations': 0,
                'passed_validations': 0,
                'failed_validations': 0,
                'failures_by_type': {k: 0 for k in self.stats['failures_by_type']}
            }
        
        def get_schema(self, task_type: str) -> Optional[ValidationSchema]:
            """
            Obtiene un esquema de validación.
            
            Args:
                task_type: Tipo de tarea
                
            Returns:
                Esquema de validación o None si no existe
            """
            return self.schemas.get(task_type)
        
        def get_all_schemas(self) -> Dict[str, ValidationSchema]:
            """
            Obtiene todos los esquemas de validación.
            
            Returns:
                Diccionario con todos los esquemas
            """
            return self.schemas.copy()
        
        def validate_task_data(self, task_type: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
            """
            Valida solo los datos de una tarea sin necesidad de crear un objeto Task completo.
            Útil para validación previa en APIs o interfaces de usuario.
            
            Args:
                task_type: Tipo de tarea
                data: Datos a validar
                
            Returns:
                Tupla (es_válido, lista_de_errores)
            """
            if task_type not in self.schemas:
                return True, []
            
            schema = self.schemas[task_type]
            errors = []
            
            # Crear un objeto Task temporal para validación
            temp_task = Task(
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                data=data,
                scheduled_time=datetime.now()
            )
            
            # Aplicar validación
            is_valid, validation_errors = self._validate_task(temp_task)
            
            # Actualizar estadísticas (opcional para validación previa)
            self.stats['total_validations'] += 1
            if not is_valid:
                self.stats['failed_validations'] += 1
                if task_type in self.stats['failures_by_type']:
                    self.stats['failures_by_type'][task_type] += 1
            else:
                self.stats['passed_validations'] += 1
            
            return is_valid, validation_errors
"""
Middleware de auditoría para el sistema de planificación.

Este módulo implementa un middleware que registra información detallada
sobre las tareas programadas y ejecutadas, proporcionando trazabilidad
completa para fines de auditoría y cumplimiento.
"""

import logging
import time
import json
import os
import socket
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from ..core.task_model import Task, TaskStatus
from .base_middleware import BaseMiddleware

logger = logging.getLogger('Scheduler.Middleware.Audit')

class AuditMiddleware(BaseMiddleware):
    """
    Middleware que registra información detallada sobre las tareas para auditoría.
    
    Características:
    - Registro de quién programa cada tarea
    - Información de origen (IP, componente)
    - Historial completo de operaciones
    - Exportación de registros para cumplimiento
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el middleware de auditoría.
        
        Args:
            config: Configuración específica del middleware
        """
        super().__init__(config)
        
        # Configuración
        self.log_dir = self.config.get('log_dir', os.path.join('logs', 'audit'))
        self.enable_file_logging = self.config.get('enable_file_logging', True)
        self.log_level = self.config.get('log_level', 'INFO')
        self.hostname = socket.gethostname()
        
        # Crear directorio de logs si no existe
        if self.enable_file_logging and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"AuditMiddleware inicializado (log_dir: {self.log_dir})")
    
    def process_task_pre_execution(self, task: Task, context: Dict[str, Any]) -> Optional[Task]:
        """
        Registra información de auditoría antes de la ejecución de una tarea.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            La tarea procesada
        """
        # Obtener información de contexto
        user_id = context.get('user_id', 'system')
        source_ip = context.get('source_ip', '127.0.0.1')
        source_component = context.get('source_component', 'unknown')
        
        # Crear registro de auditoría
        audit_entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_type': 'task_scheduled',
            'task_id': task.task_id,
            'task_type': task.task_type,
            'user_id': user_id,
            'source_ip': source_ip,
            'source_component': source_component,
            'hostname': self.hostname,
            'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
            'data': {k: v for k, v in task.data.items() if k not in self.config.get('sensitive_fields', [])}
        }
        
        # Registrar en log
        logger.info(f"AUDIT: Tarea {task.task_id} programada por {user_id} desde {source_ip} ({source_component})")
        
        # Guardar en archivo si está habilitado
        if self.enable_file_logging:
            self._write_audit_log(audit_entry)
        
        # Añadir información de auditoría a la tarea
        if not hasattr(task, 'audit_trail') or task.audit_trail is None:
            task.audit_trail = []
        
        task.audit_trail.append(audit_entry)
        
        return task
    
    def process_task_post_execution(self, task: Task, result: Any, error: Optional[str],
                                   context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Registra información de auditoría después de la ejecución de una tarea.
        
        Args:
            task: Tarea ejecutada
            result: Resultado de la ejecución
            error: Error si la tarea falló
            context: Contexto adicional
            
        Returns:
            Tupla (resultado, error)
        """
        # Crear registro de auditoría
        event_type = 'task_completed' if error is None else 'task_failed'
        
        audit_entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'task_id': task.task_id,
            'task_type': task.task_type,
            'execution_time': context.get('execution_time', 0),
            'hostname': self.hostname,
            'error': error
        }
        
        # No incluir resultado completo en el log para evitar datos sensibles
        if result is not None and not error:
            if isinstance(result, dict):
                # Incluir solo metadatos del resultado
                audit_entry['result_summary'] = {
                    'size': len(str(result)),
                    'keys': list(result.keys()) if isinstance(result, dict) else None
                }
            else:
                audit_entry['result_summary'] = {
                    'type': type(result).__name__,
                    'size': len(str(result)) if hasattr(result, '__len__') else 'N/A'
                }
        
        # Registrar en log
        if error:
            logger.info(f"AUDIT: Tarea {task.task_id} falló: {error}")
        else:
            logger.info(f"AUDIT: Tarea {task.task_id} completada exitosamente")
        
        # Guardar en archivo si está habilitado
        if self.enable_file_logging:
            self._write_audit_log(audit_entry)
        
        # Añadir información de auditoría a la tarea
        if not hasattr(task, 'audit_trail') or task.audit_trail is None:
            task.audit_trail = []
        
        task.audit_trail.append(audit_entry)
        
        return result, error
    
    def _write_audit_log(self, audit_entry: Dict[str, Any]) -> None:
        """
        Escribe un registro de auditoría en archivo.
        
        Args:
            audit_entry: Registro de auditoría
        """
        try:
            # Crear nombre de archivo basado en la fecha
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(self.log_dir, f"audit_{date_str}.jsonl")
            
            # Escribir registro en formato JSON Lines
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
        
        except Exception as e:
            logger.error(f"Error al escribir registro de auditoría: {str(e)}")
    
    def get_audit_trail(self, task_id: str = None, user_id: str = None, 
                       start_time: datetime = None, end_time: datetime = None,
                       event_type: str = None) -> List[Dict[str, Any]]:
        """
        Obtiene registros de auditoría filtrados.
        
        Args:
            task_id: Filtrar por ID de tarea
            user_id: Filtrar por ID de usuario
            start_time: Tiempo de inicio
            end_time: Tiempo de fin
            event_type: Tipo de evento
            
        Returns:
            Lista de registros de auditoría
        """
        if not self.enable_file_logging:
            logger.warning("Registro de auditoría en archivo no está habilitado")
            return []
        
        results = []
        
        # Determinar archivos a buscar
        if start_time and end_time:
            # Generar lista de fechas entre start_time y end_time
            current_date = start_time.date()
            end_date = end_time.date()
            dates = []
            
            while current_date <= end_date:
                dates.append(current_date.strftime('%Y-%m-%d'))
                current_date = current_date.replace(day=current_date.day + 1)
        else:
            # Usar solo el archivo actual
            dates = [datetime.now().strftime('%Y-%m-%d')]
        
        # Buscar en cada archivo
        for date_str in dates:
            log_file = os.path.join(self.log_dir, f"audit_{date_str}.jsonl")
            
            if not os.path.exists(log_file):
                continue
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Aplicar filtros
                            if task_id and entry.get('task_id') != task_id:
                                continue
                            
                            if user_id and entry.get('user_id') != user_id:
                                continue
                            
                            if event_type and entry.get('event_type') != event_type:
                                continue
                            
                            if start_time:
                                entry_time = datetime.fromisoformat(entry.get('timestamp'))
                                if entry_time < start_time:
                                    continue
                            
                            if end_time:
                                entry_time = datetime.fromisoformat(entry.get('timestamp'))
                                if entry_time > end_time:
                                    continue
                            
                            results.append(entry)
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Formato inválido en registro de auditoría: {line}")
            
            except Exception as e:
                logger.error(f"Error al leer archivo de auditoría {log_file}: {str(e)}")
        
        # Ordenar por timestamp
        results.sort(key=lambda x: x.get('timestamp', ''))
        
        return results
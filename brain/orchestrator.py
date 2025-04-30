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
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Importaciones de módulos internos
from brain.orchestrator.core.facade import OrchestratorFacade
from brain.orchestrator.core.config import ConfigManager
from brain.orchestrator.core.task import TaskManager, Task, TaskPriority, TaskStatus, TaskType
from brain.orchestrator.core.channel import ChannelManager, Channel, ChannelStatus

from brain.orchestrator.managers.content import ContentManager
from brain.orchestrator.managers.publish import PublishManager
from brain.orchestrator.managers.analysis import AnalysisManager
from brain.orchestrator.managers.monetization import MonetizationManager
from brain.orchestrator.managers.compliance import ComplianceManager
from brain.orchestrator.managers.shadowban import ShadowbanManager

from brain.orchestrator.utils.errors import (
    OrchestratorError, ContentGenerationError, AnalysisError, 
    PublishingError, MonetizationError, RateLimitError
)
from brain.orchestrator.utils.logging import setup_logging, get_logger
from brain.orchestrator.utils.monitoring import MetricsCollector
from brain.orchestrator.utils.persistence import persistence_manager

from brain.decision_engine import DecisionEngine
from brain.scheduler import Scheduler
from brain.notifier import Notifier

# Configuración de logging
logger = get_logger(__name__)
activity_logger = get_logger("orchestrator_activity", log_file="logs/activity/orchestrator_activity.log")

class Orchestrator:
    """
    Clase principal que implementa el patrón Facade para coordinar todos los subsistemas
    del sistema de monetización.
    
    Esta clase es un punto de entrada único que delega las operaciones a los componentes
    especializados, manteniendo una interfaz simple para los clientes.
    """
    
    def __init__(self, config_path: str = "config/strategy.json"):
        """
        Inicializa el Orchestrator con todos sus componentes.
        
        Args:
            config_path: Ruta al archivo de configuración principal
        """
        activity_logger.info("Iniciando Orchestrator...")
        
        # Inicializar componentes de configuración y persistencia
        self.config_manager = ConfigManager(config_path)
        self.persistence = persistence_manager
        
        # Inicializar gestores especializados
        self.task_manager = TaskManager(self.persistence)
        self.channel_manager = ChannelManager(self.persistence)
        
        self.content_manager = ContentManager(self.config_manager)
        self.publish_manager = PublishManager(self.config_manager)
        self.analysis_manager = AnalysisManager(self.config_manager)
        self.monetization_manager = MonetizationManager(self.config_manager)
        self.compliance_manager = ComplianceManager(self.config_manager)
        self.shadowban_manager = ShadowbanManager(self.config_manager)
        
        # Inicializar componentes de brain
        self.decision_engine = DecisionEngine()
        self.scheduler = Scheduler()
        self.notifier = Notifier()
        
        # Inicializar métricas y monitoreo
        self.metrics_collector = MetricsCollector()
        
        # Estado del sistema
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Hilos de trabajo
        self.worker_threads = []
        self.num_workers = self.config_manager.get("system.num_workers", 5)
        
        # Inicializar facade para acceso externo
        self.facade = OrchestratorFacade(self)
        
        activity_logger.info("Orchestrator inicializado correctamente")
    
    def start(self):
        """Inicia el orchestrator y todos sus subsistemas"""
        if self.running:
            logger.warning("Orchestrator ya está en ejecución")
            return
        
        activity_logger.info("Iniciando servicios del Orchestrator...")
        
        # Iniciar componentes
        self.persistence.start()
        self.metrics_collector.start()
        
        # Iniciar hilos de trabajo
        self.running = True
        self.shutdown_event.clear()
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"OrchestratorWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Iniciar hilo de monitoreo
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="OrchestratorMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        activity_logger.info(f"Orchestrator iniciado con {self.num_workers} trabajadores")
    
    def stop(self):
        """Detiene el orchestrator y todos sus subsistemas de manera ordenada"""
        if not self.running:
            logger.warning("Orchestrator no está en ejecución")
            return
        
        activity_logger.info("Deteniendo Orchestrator...")
        
        # Señalizar a los hilos para que se detengan
        self.running = False
        self.shutdown_event.set()
        
        # Esperar a que los hilos terminen
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        # Detener componentes
        self.metrics_collector.stop()
        self.persistence.stop()
        
        self.worker_threads = []
        
        activity_logger.info("Orchestrator detenido correctamente")
    
    def _worker_loop(self):
        """Bucle principal de los hilos trabajadores"""
        logger.info(f"Iniciando hilo trabajador {threading.current_thread().name}")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Obtener siguiente tarea de la cola
                task = self.task_manager.get_next_task()
                
                if task:
                    self._process_task(task)
                else:
                    # Si no hay tareas, esperar un poco
                    time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error en hilo trabajador: {str(e)}", exc_info=True)
                # Esperar un poco antes de continuar
                time.sleep(5.0)
        
        logger.info(f"Finalizando hilo trabajador {threading.current_thread().name}")
    
    def _monitor_loop(self):
        """Bucle de monitoreo del sistema"""
        logger.info("Iniciando hilo de monitoreo")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Recolectar métricas del sistema
                self._collect_system_metrics()
                
                # Verificar estado de los canales
                self._check_channels_health()
                
                # Verificar tareas pendientes
                self._check_pending_tasks()
                
                # Esperar antes de la siguiente iteración
                time.sleep(60.0)  # Monitoreo cada minuto
            
            except Exception as e:
                logger.error(f"Error en hilo de monitoreo: {str(e)}", exc_info=True)
                time.sleep(60.0)
        
        logger.info("Finalizando hilo de monitoreo")
    
    def _process_task(self, task: Task):
        """
        Procesa una tarea según su tipo.
        
        Args:
            task: La tarea a procesar
        """
        activity_logger.info(f"Procesando tarea {task.id} de tipo {task.type}")
        
        try:
            # Actualizar estado de la tarea
            task.status = TaskStatus.PROCESSING
            self.task_manager.update_task(task)
            
            # Procesar según tipo
            if task.type == TaskType.CONTENT_CREATION:
                result = self.content_manager.create_content(task.data)
                task.result = result
            
            elif task.type == TaskType.CONTENT_PUBLISHING:
                result = self.publish_manager.publish_content(task.data)
                task.result = result
            
            elif task.type == TaskType.CONTENT_ANALYSIS:
                result = self.analysis_manager.analyze_content(task.data)
                task.result = result
            
            elif task.type == TaskType.MONETIZATION:
                result = self.monetization_manager.optimize_monetization(task.data)
                task.result = result
            
            elif task.type == TaskType.COMPLIANCE_CHECK:
                result = self.compliance_manager.check_compliance(task.data)
                task.result = result
            
            elif task.type == TaskType.SHADOWBAN_RECOVERY:
                result = self.shadowban_manager.execute_recovery_plan(task.data)
                task.result = result
            
            else:
                logger.warning(f"Tipo de tarea desconocido: {task.type}")
                task.status = TaskStatus.FAILED
                task.error = "Tipo de tarea desconocido"
                self.task_manager.update_task(task)
                return
            
            # Marcar como completada
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.datetime.now().isoformat()
            self.task_manager.update_task(task)
            
            activity_logger.info(f"Tarea {task.id} completada exitosamente")
            
            # Crear tareas dependientes si es necesario
            self._create_dependent_tasks(task)
        
        except Exception as e:
            logger.error(f"Error al procesar tarea {task.id}: {str(e)}", exc_info=True)
            
            # Actualizar estado de la tarea
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.retry_count = task.retry_count + 1
            
            # Verificar si se debe reintentar
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                activity_logger.warning(f"Reintentando tarea {task.id} (intento {task.retry_count + 1}/{task.max_retries})")
            else:
                activity_logger.error(f"Tarea {task.id} falló después de {task.retry_count} intentos")
            
            self.task_manager.update_task(task)
    
    def _create_dependent_tasks(self, completed_task: Task):
        """
        Crea tareas dependientes basadas en una tarea completada.
        
        Args:
            completed_task: La tarea completada que puede generar tareas dependientes
        """
        if completed_task.type == TaskType.CONTENT_CREATION:
            # Después de crear contenido, programar publicación
            self.task_manager.create_task(
                task_type=TaskType.CONTENT_PUBLISHING,
                channel_id=completed_task.channel_id,
                priority=TaskPriority.HIGH,
                data={
                    "content_id": completed_task.result.get("content_id"),
                    "platform": completed_task.data.get("platform"),
                    "schedule_time": completed_task.data.get("schedule_time")
                }
            )
        
        elif completed_task.type == TaskType.CONTENT_PUBLISHING:
            # Después de publicar, programar análisis
            self.task_manager.create_task(
                task_type=TaskType.CONTENT_ANALYSIS,
                channel_id=completed_task.channel_id,
                priority=TaskPriority.NORMAL,
                data={
                    "content_id": completed_task.data.get("content_id"),
                    "platform": completed_task.data.get("platform"),
                    "post_id": completed_task.result.get("post_id")
                }
            )
            
            # También programar verificación de cumplimiento
            self.task_manager.create_task(
                task_type=TaskType.COMPLIANCE_CHECK,
                channel_id=completed_task.channel_id,
                priority=TaskPriority.HIGH,
                data={
                    "content_id": completed_task.data.get("content_id"),
                    "platform": completed_task.data.get("platform"),
                    "post_id": completed_task.result.get("post_id")
                }
            )
    
    def _collect_system_metrics(self):
        """Recolecta métricas del sistema para monitoreo"""
        try:
            # Métricas de CPU y memoria
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_info = psutil.virtual_memory()
            
            # Métricas de tareas
            pending_tasks = self.task_manager.count_tasks_by_status(TaskStatus.PENDING)
            processing_tasks = self.task_manager.count_tasks_by_status(TaskStatus.PROCESSING)
            completed_tasks = self.task_manager.count_tasks_by_status(TaskStatus.COMPLETED)
            failed_tasks = self.task_manager.count_tasks_by_status(TaskStatus.FAILED)
            
            # Registrar métricas
            self.metrics_collector.record_metric("system.cpu_percent", cpu_percent)
            self.metrics_collector.record_metric("system.memory_percent", memory_info.percent)
            self.metrics_collector.record_metric("tasks.pending", pending_tasks)
            self.metrics_collector.record_metric("tasks.processing", processing_tasks)
            self.metrics_collector.record_metric("tasks.completed", completed_tasks)
            self.metrics_collector.record_metric("tasks.failed", failed_tasks)
            
            logger.debug(f"Métricas del sistema: CPU={cpu_percent}%, MEM={memory_info.percent}%, "
                         f"Tareas: {pending_tasks} pendientes, {processing_tasks} en proceso, "
                         f"{completed_tasks} completadas, {failed_tasks} fallidas")
        
        except Exception as e:
            logger.error(f"Error al recolectar métricas del sistema: {str(e)}")
    
    def _check_channels_health(self):
        """Verifica el estado de salud de los canales"""
        try:
            channels = self.channel_manager.get_all_channels()
            
            for channel in channels:
                if channel.status == ChannelStatus.ACTIVE:
                    # Verificar shadowban para canales activos
                    is_shadowbanned = self.shadowban_manager.check_shadowban(channel.id)
                    
                    if is_shadowbanned:
                        activity_logger.warning(f"Canal {channel.id} detectado como shadowbanned")
                        
                        # Cambiar estado del canal
                        self.channel_manager.update_channel_status(
                            channel_id=channel.id,
                            status=ChannelStatus.SHADOWBANNED
                        )
                        
                        # Crear plan de recuperación
                        recovery_plan = self.shadowban_manager.create_recovery_plan(channel.id)
                        
                        # Crear tarea de recuperación
                        self.task_manager.create_task(
                            task_type=TaskType.SHADOWBAN_RECOVERY,
                            channel_id=channel.id,
                            priority=TaskPriority.CRITICAL,
                            data={"recovery_plan": recovery_plan}
                        )
                        
                        # Notificar
                        self.notifier.send_notification(
                            title=f"Shadowban detectado en canal {channel.id}",
                            message=f"Se ha detectado un shadowban en el canal {channel.id}. "
                                    f"Se ha creado un plan de recuperación automático.",
                            level="warning"
                        )
        
        except Exception as e:
            logger.error(f"Error al verificar salud de canales: {str(e)}")
    
    def _check_pending_tasks(self):
        """Verifica tareas pendientes y realiza acciones necesarias"""
        try:
            # Verificar tareas atascadas (en proceso por mucho tiempo)
            stuck_tasks = self.task_manager.get_stuck_tasks(
                status=TaskStatus.PROCESSING,
                older_than_minutes=30
            )
            
            for task in stuck_tasks:
                logger.warning(f"Tarea {task.id} atascada por más de 30 minutos")
                
                # Reiniciar tarea
                task.status = TaskStatus.PENDING
                task.retry_count += 1
                
                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    task.error = "Máximo número de reintentos alcanzado"
                    
                    # Notificar
                    self.notifier.send_notification(
                        title=f"Tarea {task.id} fallida",
                        message=f"La tarea {task.id} de tipo {task.type} ha fallado después de "
                                f"{task.max_retries} intentos.",
                        level="error"
                    )
                
                self.task_manager.update_task(task)
        
        except Exception as e:
            logger.error(f"Error al verificar tareas pendientes: {str(e)}")
    
    def run_pipeline(self, channel_id: str):
        """
        Ejecuta el pipeline completo para un canal específico.
        
        Args:
            channel_id: ID del canal para ejecutar el pipeline
        """
        activity_logger.info(f"Iniciando pipeline para canal {channel_id}")
        
        try:
            # Verificar estado del canal
            channel = self.channel_manager.get_channel(channel_id)
            
            if not channel:
                raise ValueError(f"Canal {channel_id} no encontrado")
            
            if channel.status != ChannelStatus.ACTIVE:
                activity_logger.warning(f"Canal {channel_id} no está activo (estado: {channel.status})")
                return
            
            # 1. Detectar tendencias
            trends = self.content_manager.detect_trends(channel.niche)
            
            if not trends:
                activity_logger.warning(f"No se encontraron tendencias para el nicho {channel.niche}")
                return
            
            # 2. Seleccionar mejor tendencia
            selected_trend = self.decision_engine.select_best_trend(trends, channel)
            
            # 3. Crear tarea de generación de contenido
            content_task = self.task_manager.create_task(
                task_type=TaskType.CONTENT_CREATION,
                channel_id=channel_id,
                priority=TaskPriority.HIGH,
                data={
                    "trend": selected_trend,
                    "niche": channel.niche,
                    "platform": channel.primary_platform,
                    "content_type": channel.content_type,
                    "schedule_time": self.scheduler.get_optimal_time(channel_id, channel.primary_platform)
                }
            )
            
            activity_logger.info(f"Pipeline iniciado para canal {channel_id}, tarea de creación {content_task.id} creada")
            
            return content_task.id
        
        except Exception as e:
            logger.error(f"Error al ejecutar pipeline para canal {channel_id}: {str(e)}", exc_info=True)
            
            # Notificar error
            self.notifier.send_notification(
                title=f"Error en pipeline para canal {channel_id}",
                message=f"Error al ejecutar pipeline: {str(e)}",
                level="error"
            )
            
            return None
    
    def create_channel(self, channel_data: Dict[str, Any]) -> str:
        """
        Crea un nuevo canal en el sistema.
        
        Args:
            channel_data: Datos del canal a crear
            
        Returns:
            ID del canal creado
        """
        try:
            # Validar datos mínimos
            required_fields = ["name", "niche", "primary_platform", "content_type"]
            for field in required_fields:
                if field not in channel_data:
                    raise ValueError(f"Campo requerido '{field}' no presente en datos del canal")
            
            # Crear canal
            channel_id = self.channel_manager.create_channel(channel_data)
            
            activity_logger.info(f"Canal {channel_id} creado: {channel_data['name']} - {channel_data['niche']}")
            
            # Crear tarea inicial de análisis de nicho
            self.task_manager.create_task(
                task_type=TaskType.CONTENT_ANALYSIS,
                channel_id=channel_id,
                priority=TaskPriority.NORMAL,
                data={
                    "analysis_type": "niche_research",
                    "niche": channel_data["niche"],
                    "platform": channel_data["primary_platform"]
                }
            )
            
            return channel_id
        
        except Exception as e:
            logger.error(f"Error al crear canal: {str(e)}", exc_info=True)
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Diccionario con información del estado del sistema
        """
        try:
            # Obtener métricas del sistema
            system_metrics = self.metrics_collector.get_recent_metrics()
            
            # Obtener conteo de tareas
            task_counts = {
                "pending": self.task_manager.count_tasks_by_status(TaskStatus.PENDING),
                "processing": self.task_manager.count_tasks_by_status(TaskStatus.PROCESSING),
                "completed": self.task_manager.count_tasks_by_status(TaskStatus.COMPLETED),
                "failed": self.task_manager.count_tasks_by_status(TaskStatus.FAILED)
            }
            
            # Obtener conteo de canales
            channel_counts = {
                "active": self.channel_manager.count_channels_by_status(ChannelStatus.ACTIVE),
                "shadowbanned": self.channel_manager.count_channels_by_status(ChannelStatus.SHADOWBANNED),
                "recovering": self.channel_manager.count_channels_by_status(ChannelStatus.RECOVERING),
                "paused": self.channel_manager.count_channels_by_status(ChannelStatus.PAUSED),
                "archived": self.channel_manager.count_channels_by_status(ChannelStatus.ARCHIVED)
            }
            
            # Obtener información del sistema
            uptime = time.time() - psutil.boot_time()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                "status": "running" if self.running else "stopped",
                "uptime_seconds": uptime,
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024 ** 3),
                    "memory_total_gb": memory.total / (1024 ** 3)
                },
                "tasks": task_counts,
                "channels": channel_counts,
                "metrics": system_metrics,
                "workers": {
                    "total": self.num_workers,
                    "active": sum(1 for worker in self.worker_threads if worker.is_alive())
                }
            }
        
        except Exception as e:
            logger.error(f"Error al obtener estado del sistema: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Instancia global del orchestrator
orchestrator = Orchestrator()

def get_orchestrator() -> Orchestrator:
    """
    Obtiene la instancia global del orchestrator.
    
    Returns:
        Instancia del orchestrator
    """
    return orchestrator

# Manejo de señales para cierre ordenado
def signal_handler(sig, frame):
    """Manejador de señales para cierre ordenado"""
    logger.info(f"Señal {sig} recibida, cerrando orchestrator...")
    orchestrator.stop()
    sys.exit(0)

# Registrar manejadores de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Iniciar orchestrator si se ejecuta directamente
    orchestrator.start()
    
    try:
        # Mantener proceso principal vivo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Detener orchestrator al recibir Ctrl+C
        orchestrator.stop()
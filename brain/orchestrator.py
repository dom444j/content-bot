"""
Orchestrator - Coordinador central del sistema de monetización

Este módulo actúa como el cerebro del sistema, coordinando todos los subsistemas:
- Detección de tendencias
- Creación de contenido
- Verificación de cumplimiento
- Publicación en plataformas
- Monetización
- Análisis y optimización

Mejoras implementadas:
1. Sistema de prioridades: Maneja tareas críticas (ej. shadowban) pausando publicaciones automáticas.
2. Persistencia de estado: Guarda estados de tareas para reanudar sin pérdida.
3. Reintentos con backoff: Reintenta pasos fallidos con espera exponencial.
4. Estados de canal: Maneja estados como ACTIVE, SHADOWBANNED, RECOVERING.
5. Planes de recuperación: Genera estrategias para superar shadowbans.
6. Monitoreo de recuperación: Verifica periódicamente el estado de recuperación.
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
from typing import Dict, List, Any, Optional, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'orchestrator.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('Orchestrator')

class Priority:
    """Clase para definir niveles de prioridad de tareas"""
    CRITICAL = 0  # Ej. Shadowban handling
    HIGH = 1      # Ej. Optimización urgente
    NORMAL = 2    # Ej. Creación de contenido
    LOW = 3       # Ej. Análisis en segundo plano

class ChannelStatus:
    """Clase para definir estados posibles de un canal"""
    ACTIVE = "active"             # Funcionamiento normal
    SHADOWBANNED = "shadowbanned" # Detectado shadowban
    RESTRICTED = "restricted"     # Restricciones de contenido
    PAUSED = "paused"             # Pausado manualmente
    RECOVERING = "recovering"     # En recuperación de penalización

class Orchestrator:
    """
    Clase principal que coordina todos los subsistemas del bot de contenido.
    Implementa el patrón Singleton para asegurar una única instancia.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Orchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el orquestador y sus componentes si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Orchestrator...")
        
        # Estado del sistema
        self.active = False
        self.channels = {}
        self.current_tasks = {}
        self.task_history = []
        self.task_queue = queue.PriorityQueue()
        self.paused_tasks = {}
        self.shadowban_status = {}
        self.channel_status = {}  # Estado actual de cada canal
        self.shadowban_history = {}  # Historial de shadowbans por canal
        self.recovery_plans = {}  # Planes de recuperación por canal
        
        # Configuración de reintentos
        self.max_retries = 3
        self.base_retry_delay = 5
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar componentes
        self.components = {
            'decision_engine': None,
            'scheduler': None,
            'notifier': None,
            'knowledge_base': None,
            'analytics_engine': None,
            'shadowban_detector': None
        }
        
        # Inicializar subsistemas
        self.subsystems = {
            'trends': {},
            'creation': {},
            'compliance': {},
            'publication': {},
            'monetization': {},
            'optimization': {},
            'analysis': {}
        }
        
        self._initialized = True
        logger.info("Orchestrator inicializado correctamente")
    
    def _load_config(self) -> Dict:
        """Carga la configuración del sistema desde archivos JSON"""
        config = {}
        config_dir = os.path.join('config')
        
        try:
            with open(os.path.join(config_dir, 'platforms.json'), 'r', encoding='utf-8') as f:
                config['platforms'] = json.load(f)
            with open(os.path.join(config_dir, 'strategy.json'), 'r', encoding='utf-8') as f:
                config['strategy'] = json.load(f)
            with open(os.path.join(config_dir, 'niches.json'), 'r', encoding='utf-8') as f:
                config['niches'] = json.load(f)
            with open(os.path.join(config_dir, 'character_profiles.json'), 'r', encoding='utf-8') as f:
                config['characters'] = json.load(f)
                
            logger.info("Configuración cargada correctamente")
            return config
            
        except FileNotFoundError as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            self._create_default_config()
            return self._load_config()
    
    def _create_default_config(self):
        """Crea archivos de configuración por defecto si no existen"""
        config_dir = os.path.join('config')
        os.makedirs(config_dir, exist_ok=True)
        
        logs_dir = os.path.join('logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        platforms_config = {
            "youtube": {"api_key": "", "channel_id": "", "enabled": True},
            "tiktok": {"api_key": "", "account_id": "", "enabled": True},
            "instagram": {"api_key": "", "account_id": "", "enabled": True},
            "threads": {"api_key": "", "account_id": "", "enabled": False},
            "bluesky": {"api_key": "", "handle": "", "enabled": False}
        }
        
        strategy_config = {
            "cta_timing": {"early": [0, 3], "middle": [4, 8], "end": [-2, 0]},
            "content_types": ["educational", "entertainment", "tutorial"],
            "monetization_priority": ["affiliate", "ads", "products"],
            "retry_policy": {"max_retries": 3, "base_delay": 5}
        }
        
        niches_config = {
            "finance": {"keywords": ["crypto", "investing", "money"], "enabled": True},
            "health": {"keywords": ["fitness", "mindfulness", "nutrition"], "enabled": True},
            "technology": {"keywords": ["ai", "gadgets", "programming"], "enabled": True},
            "gaming": {"keywords": ["strategy", "review", "gameplay"], "enabled": True},
            "humor": {"keywords": ["memes", "comedy", "funny"], "enabled": True}
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
            }
        }
        
        with open(os.path.join(config_dir, 'platforms.json'), 'w', encoding='utf-8') as f:
            json.dump(platforms_config, f, indent=4)
        with open(os.path.join(config_dir, 'strategy.json'), 'w', encoding='utf-8') as f:
            json.dump(strategy_config, f, indent=4)
        with open(os.path.join(config_dir, 'niches.json'), 'w', encoding='utf-8') as f:
            json.dump(niches_config, f, indent=4)
        with open(os.path.join(config_dir, 'character_profiles.json'), 'w', encoding='utf-8') as f:
            json.dump(characters_config, f, indent=4)
            
        logger.info("Creada configuración por defecto")
    
    def _load_component(self, component_name: str) -> Any:
        """Carga dinámicamente un componente del cerebro según sea necesario"""
        if self.components[component_name] is not None:
            return self.components[component_name]
        
        try:
            if component_name == 'decision_engine':
                from brain.decision_engine import DecisionEngine
                self.components[component_name] = DecisionEngine()
            elif component_name == 'scheduler':
                from brain.scheduler import Scheduler
                self.components[component_name] = Scheduler()
            elif component_name == 'notifier':
                from brain.notifier import Notifier
                self.components[component_name] = Notifier()
            elif component_name == 'knowledge_base':
                from data.knowledge_base import KnowledgeBase
                self.components[component_name] = KnowledgeBase()
            elif component_name == 'analytics_engine':
                from data.analytics_engine import AnalyticsEngine
                self.components[component_name] = AnalyticsEngine()
            elif component_name == 'shadowban_detector':
                from compliance.shadowban_detector import ShadowbanDetector
                self.components[component_name] = ShadowbanDetector()
            
            logger.info(f"Componente {component_name} cargado correctamente")
            return self.components[component_name]
            
        except ImportError as e:
            logger.error(f"Error al cargar componente {component_name}: {str(e)}")
            return None
    
    def _load_subsystem(self, subsystem: str, module: str) -> Any:
        """Carga dinámicamente un módulo de un subsistema según sea necesario"""
        if module in self.subsystems[subsystem]:
            return self.subsystems[subsystem][module]
        
        try:
            if subsystem == 'trends':
                if module == 'trend_radar':
                    from trends.trend_radar import TrendRadar
                    self.subsystems[subsystem][module] = TrendRadar()
                elif module == 'opportunity_scorer':
                    from trends.opportunity_scorer import OpportunityScorer
                    self.subsystems[subsystem][module] = OpportunityScorer()
                elif module == 'forecasting_engine':
                    from trends.forecasting_engine import ForecastingEngine
                    self.subsystems[subsystem][module] = ForecastingEngine()
            elif subsystem == 'creation':
                if module == 'story_engine':
                    from creation.narrative.story_engine import StoryEngine
                    self.subsystems[subsystem][module] = StoryEngine()
                elif module == 'visual_generator':
                    from creation.assembly.visual_generator import VisualGenerator
                    self.subsystems[subsystem][module] = VisualGenerator()
                elif module == 'voice_generator':
                    from creation.assembly.voice_trainer import VoiceTrainer
                    self.subsystems[subsystem][module] = VoiceTrainer()
                elif module == 'video_composer':
                    from creation.assembly.video_producer.video_composer import VideoComposer
                    self.subsystems[subsystem][module] = VideoComposer()
            elif subsystem == 'compliance':
                if module == 'compliance_checker':
                    from compliance.content_auditor import ContentAuditor
                    self.subsystems[subsystem][module] = ContentAuditor()
            elif subsystem == 'publication':
                if module in ['youtube_adapter', 'tiktok_adapter', 'instagram_adapter']:
                    from platform_adapters import api_router
                    self.subsystems[subsystem][module] = getattr(api_router, module)()
            elif subsystem == 'monetization':
                if module == 'revenue_optimizer':
                    from monetization.revenue_optimizer import RevenueOptimizer
                    self.subsystems[subsystem][module] = RevenueOptimizer()
            elif subsystem == 'optimization':
                if module == 'reputation_engine':
                    from optimization.reputation_engine import ReputationEngine
                    self.subsystems[subsystem][module] = ReputationEngine()
            elif subsystem == 'analysis':
                if module == 'competitor_analyzer':
                    from analysis.competitor_analyzer import CompetitorAnalyzer
                    self.subsystems[subsystem][module] = CompetitorAnalyzer()
            
            logger.info(f"Módulo {module} del subsistema {subsystem} cargado correctamente")
            return self.subsystems[subsystem][module]
            
        except ImportError as e:
            logger.error(f"Error al cargar módulo {module} del subsistema {subsystem}: {str(e)}")
            return None
    
    def start(self) -> bool:
        """Inicia el orquestador y todos los subsistemas necesarios"""
        if self.active:
            logger.warning("El orquestador ya está activo")
            return True
        
        logger.info("Iniciando Orchestrator...")
        
        try:
            # Cargar componentes esenciales
            self._load_component('decision_engine')
            self._load_component('scheduler')
            self._load_component('knowledge_base')
            self._load_component('shadowban_detector')
            
            # Verificar configuración
            if not self._verify_config():
                logger.error("Configuración incompleta o inválida")
                return False
            
            # Iniciar canales configurados
            self._initialize_channels()
            
            # Cargar tareas pendientes desde la base de conocimiento
            self._load_pending_tasks()
            
            # Marcar como activo
            self.active = True
            logger.info("Orchestrator iniciado correctamente")
            
            # Iniciar bucle principal, manejador de prioridades y monitoreo de recuperación
            threading.Thread(target=self._main_loop, daemon=True).start()
            threading.Thread(target=self._priority_handler, daemon=True).start()
            threading.Thread(target=self._recovery_monitor, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar Orchestrator: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """Detiene el orquestador y todos los subsistemas"""
        if not self.active:
            logger.warning("El orquestador ya está detenido")
            return True
        
        logger.info("Deteniendo Orchestrator...")
        
        try:
            # Pausar tareas en curso
            for task_id, task in self.current_tasks.items():
                logger.info(f"Pausando tarea {task_id}")
                self.paused_tasks[task_id] = task
                self.current_tasks[task_id]['status'] = 'paused'
                self._persist_task(task)
            
            # Vaciar cola de prioridades
            while not self.task_queue.empty():
                self.task_queue.get()
            
            # Marcar como inactivo
            self.active = False
            logger.info("Orchestrator detenido correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al detener Orchestrator: {str(e)}")
            return False
    
    def _verify_config(self) -> bool:
        """Verifica que la configuración sea válida y completa"""
        if not self.config:
            return False
        
        platforms_enabled = any(p.get('enabled', False) for p in self.config.get('platforms', {}).values())
        if not platforms_enabled:
            logger.warning("No hay plataformas habilitadas en la configuración")
            return False
        
        niches_enabled = any(n.get('enabled', False) for n in self.config.get('niches', {}).values())
        if not niches_enabled:
            logger.warning("No hay nichos habilitados en la configuración")
            return False
        
        return True
    
    def _initialize_channels(self):
        """Inicializa los canales configurados"""
        knowledge_base = self._load_component('knowledge_base')
        if knowledge_base:
            self.channels = knowledge_base.get_channels()
            
        if not self.channels:
            logger.info("No se encontraron canales existentes, creando canales predeterminados")
            self._create_default_channels()
        
        # Inicializar estados de canales
        for channel_id in self.channels:
            self.channel_status[channel_id] = ChannelStatus.ACTIVE
    
    def _create_default_channels(self):
        """Crea canales predeterminados basados en la configuración"""
        for niche_name, niche_data in self.config.get('niches', {}).items():
            if niche_data.get('enabled', False):
                channel_id = f"{niche_name}_{int(time.time())}"
                platforms = []
                for platform_name, platform_data in self.config.get('platforms', {}).items():
                    if platform_data.get('enabled', False):
                        platforms.append(platform_name)
                
                character = None
                for char_name, char_data in self.config.get('characters', {}).items():
                    if niche_name in char_name:
                        character = char_name
                        break
                
                if not character and self.config.get('characters'):
                    character = list(self.config.get('characters').keys())[0]
                
                self.channels[channel_id] = {
                    'id': channel_id,
                    'name': f"{niche_name.capitalize()} Channel",
                    'niche': niche_name,
                    'platforms': platforms,
                    'character': character,
                    'created_at': datetime.datetime.now().isoformat(),
                    'status': 'active',
                    'stats': {
                        'videos': 0,
                        'subscribers': 0,
                        'views': 0,
                        'revenue': 0
                    }
                }
                
                logger.info(f"Creado canal predeterminado: {self.channels[channel_id]['name']}")
                
                knowledge_base = self._load_component('knowledge_base')
                if knowledge_base:
                    knowledge_base.save_channel(self.channels[channel_id])
    
    def _load_pending_tasks(self):
        """Carga tareas pendientes desde la base de conocimiento"""
        knowledge_base = self._load_component('knowledge_base')
        if knowledge_base:
            pending_tasks = knowledge_base.get_pending_tasks()
            for task in pending_tasks:
                priority = task.get('priority', Priority.NORMAL)
                self.task_queue.put((priority, task))
                self.current_tasks[task['id']] = task
                logger.info(f"Cargada tarea pendiente: {task['id']}")
    
    def _persist_task(self, task: Dict):
        """Persiste el estado de una tarea en la base de conocimiento"""
        knowledge_base = self._load_component('knowledge_base')
        if knowledge_base:
            knowledge_base.save_task_state(task)
            logger.info(f"Estado de tarea persistido: {task['id']}")
    
    def add_task(self, task: Dict, priority: int):
        """Añade una tarea a la cola de prioridades"""
        task['id'] = task.get('id', f"task_{int(time.time())}")
        task['status'] = task.get('status', 'initiated')
        task['priority'] = priority
        task['created_at'] = task.get('created_at', datetime.datetime.now().isoformat())
        
        self.current_tasks[task['id']] = task
        self.task_queue.put((priority, task))
        self._persist_task(task)
        logger.info(f"Tarea añadida: {task['id']} con prioridad {priority}")
    
    def create_content(self, channel_id: str, content_type: str = None, priority: int = Priority.NORMAL) -> Dict:
        """Inicia el proceso de creación de contenido para un canal específico"""
        if not self.active:
            logger.error("El orquestador no está activo")
            return {'success': False, 'error': 'Orchestrator not active'}
        
        if channel_id not in self.channels:
            logger.error(f"Canal no encontrado: {channel_id}")
            return {'success': False, 'error': 'Channel not found'}
        
        # Verificar estado del canal
        if self.channel_status.get(channel_id) in [ChannelStatus.SHADOWBANNED, ChannelStatus.RECOVERING]:
            logger.warning(f"Canal {channel_id} en estado {self.channel_status[channel_id]}, pausando creación de contenido")
            return {'success': False, 'error': f'Channel in {self.channel_status[channel_id]} status'}
        
        logger.info(f"Iniciando creación de contenido para canal: {channel_id}")
        
        try:
            task_id = f"content_{channel_id}_{int(time.time())}"
            channel = self.channels[channel_id]
            
            if not content_type:
                decision_engine = self._load_component('decision_engine')
                if decision_engine:
                    content_type = decision_engine.decide_content_type(channel)
                else:
                    content_type = self.config.get('strategy', {}).get('content_types', ['educational'])[0]
            
            task = {
                'id': task_id,
                'type': 'content_creation',
                'channel_id': channel_id,
                'content_type': content_type,
                'status': 'initiated',
                'priority': priority,
                'created_at': datetime.datetime.now().isoformat(),
                'steps': {
                    'trend_detection': {'status': 'pending', 'retries': 0},
                    'script_creation': {'status': 'pending', 'retries': 0},
                    'visual_generation': {'status': 'pending', 'retries': 0},
                    'audio_generation': {'status': 'pending', 'retries': 0},
                    'video_production': {'status': 'pending', 'retries': 0},
                    'compliance_check': {'status': 'pending', 'retries': 0},
                    'publication': {'status': 'pending', 'retries': 0},
                    'monetization': {'status': 'pending', 'retries': 0}
                }
            }
            
            self.add_task(task, priority)
            
            logger.info(f"Tarea de creación de contenido iniciada: {task_id}")
            return {'success': True, 'task_id': task_id, 'task': task}
            
        except Exception as e:
            logger.error(f"Error al iniciar creación de contenido: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_task_status(self, task_id: str) -> Dict:
        """Obtiene el estado actual de una tarea"""
        if task_id in self.current_tasks:
            return {'success': True, 'task': self.current_tasks[task_id]}
        elif task_id in self.paused_tasks:
            return {'success': True, 'task': self.paused_tasks[task_id], 'paused': True}
        elif any(task['id'] == task_id for task in self.task_history):
            task = next(task for task in self.task_history if task['id'] == task_id)
            return {'success': True, 'task': task}
        else:
            return {'success': False, 'error': 'Task not found'}
    
    def get_channel_stats(self, channel_id: str) -> Dict:
        """Obtiene estadísticas de un canal específico"""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        analytics_engine = self._load_component('analytics_engine')
        if analytics_engine:
            stats = analytics_engine.get_channel_stats(channel_id)
            if stats:
                self.channels[channel_id]['stats'] = stats
        
        return {'success': True, 'channel': self.channels[channel_id]}
    
    def optimize_channel(self, channel_id: str, priority: int = Priority.HIGH) -> Dict:
        """Inicia el proceso de optimización para un canal específico"""
        if not self.active:
            return {'success': False, 'error': 'Orchestrator not active'}
        
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        logger.info(f"Iniciando optimización para canal: {channel_id}")
        
        try:
            task_id = f"optimize_{channel_id}_{int(time.time())}"
            
            task = {
                'id': task_id,
                'type': 'channel_optimization',
                'channel_id': channel_id,
                'status': 'initiated',
                'priority': priority,
                'created_at': datetime.datetime.now().isoformat(),
                'steps': {
                    'performance_analysis': {'status': 'pending', 'retries': 0},
                    'competitor_analysis': {'status': 'pending', 'retries': 0},
                    'content_strategy_update': {'status': 'pending', 'retries': 0},
                    'cta_optimization': {'status': 'pending', 'retries': 0},
                    'monetization_optimization': {'status': 'pending', 'retries': 0}
                }
            }
            
            self.add_task(task, priority)
            
            logger.info(f"Tarea de optimización iniciada: {task_id}")
            return {'success': True, 'task_id': task_id, 'task': task}
            
        except Exception as e:
            logger.error(f"Error al iniciar optimización: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_shadowban(self, channel_id: str) -> bool:
        """
        Verifica si un canal está en shadowban
        """
        if channel_id not in self.channels:
            return False
            
        if channel_id in self.channel_status and self.channel_status[channel_id] == ChannelStatus.SHADOWBANNED:
            return True
        
        shadowbanned = False
        for platform in self.channels[channel_id]['platforms']:
            platform_adapter = self._load_subsystem('publication', f'{platform}_adapter')
            if platform_adapter:
                result = platform_adapter.check_shadowban(
                    channel_id=self.channels[channel_id].get(f'{platform}_id', '')
                )
                if result.get('shadowbanned', False):
                    shadowbanned = True
                    logger.warning(f"Shadowban detectado en {platform} para canal {channel_id}")
                    
                    self.add_task({
                        'type': 'shadowban_recovery',
                        'channel_id': channel_id,
                        'platform': platform,
                        'detected_at': datetime.datetime.now().isoformat(),
                        'steps': {
                            'pause_publications': {'status': 'pending', 'retries': 0},
                            'analyze_cause': {'status': 'pending', 'retries': 0},
                            'implement_fix': {'status': 'pending', 'retries': 0},
                            'verify_resolution': {'status': 'pending', 'retries': 0}
                        }
                    }, Priority.CRITICAL)
                    
                    self.channel_status[channel_id] = ChannelStatus.SHADOWBANNED
                    
                    notifier = self._load_component('notifier')
                    if notifier:
                        notifier.send_notification(
                            title=f"⚠️ Shadowban detectado en {self.channels[channel_id]['name']}",
                            message=f"Se ha detectado un shadowban en {platform}. Se pausarán las publicaciones automáticas.",
                            level="warning"
                        )
                    
                    if channel_id not in self.shadowban_history:
                        self.shadowban_history[channel_id] = []
                    self.shadowban_history[channel_id].append({
                        'platform': platform,
                        'detected_at': datetime.datetime.now().isoformat(),
                        'resolved': False
                    })
                    
                    break
        
        return shadowbanned
    
    def _create_recovery_plan(self, channel_id: str) -> Dict[str, Any]:
        """
        Crea un plan de recuperación para un canal en shadowban
        """
        channel = self.channels.get(channel_id, {})
        
        recovery_duration = 72
        if channel_id in self.shadowban_history:
            previous_shadowbans = len([s for s in self.shadowban_history[channel_id] if s.get('resolved', False)])
            if previous_shadowbans > 0:
                recovery_duration *= (1 + 0.5 * previous_shadowbans)
        
        current_time = datetime.datetime.now()
        recovery_until = current_time + datetime.timedelta(hours=recovery_duration)
        
        plan = {
            'start_date': current_time.isoformat(),
            'end_date': recovery_until.isoformat(),
            'duration_hours': recovery_duration,
            'content_restrictions': [
                'reduced_posting_frequency',
                'no_controversial_topics',
                'reduced_ctas',
                'family_friendly_content'
            ],
            'posting_frequency': 'reduced',
            'content_strategy': 'safe'
        }
        
        self.recovery_plans[channel_id] = plan
        self.channel_status[channel_id] = ChannelStatus.RECOVERING
        
        return plan
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calcula el retraso para reintentos con backoff exponencial"""
        return self.base_retry_delay * (2 ** retry_count) + random.uniform(0, 0.1)
    
    def _main_loop(self):
        """Bucle principal que verifica shadowbans periódicamente"""
        while self.active:
            try:
                for channel_id in self.channels:
                    if self._check_shadowban(channel_id):
                        for platform in self.channels[channel_id]['platforms']:
                            if f"{channel_id}_{platform}" in self.shadowban_status and \
                               self.shadowban_status[f"{channel_id}_{platform}"]['active']:
                                self.add_task({
                                    'type': 'shadowban_recovery',
                                    'channel_id': channel_id,
                                    'platform': platform,
                                    'detected_at': datetime.datetime.now().isoformat(),
                                    'steps': {
                                        'pause_publications': {'status': 'pending', 'retries': 0},
                                        'analyze_cause': {'status': 'pending', 'retries': 0},
                                        'implement_fix': {'status': 'pending', 'retries': 0},
                                        'verify_resolution': {'status': 'pending', 'retries': 0}
                                    }
                                }, Priority.CRITICAL)
            
                time.sleep(3600)
            except Exception as e:
                logger.error(f"Error en bucle principal: {str(e)}")
                time.sleep(60)
    
    def _recovery_monitor(self):
        """Monitorea el estado de recuperación de canales"""
        while self.active:
            try:
                current_time = datetime.datetime.now()
                for channel_id, plan in list(self.recovery_plans.items()):
                    end_date = datetime.datetime.fromisoformat(plan['end_date'])
                    if current_time >= end_date:
                        # Verificar si el shadowban se resolvió
                        if not self._check_shadowban(channel_id):
                            self.channel_status[channel_id] = ChannelStatus.ACTIVE
                            del self.recovery_plans[channel_id]
                            logger.info(f"Canal {channel_id} recuperado, volviendo a estado ACTIVE")
                            
                            notifier = self._load_component('notifier')
                            if notifier:
                                notifier.send_notification(
                                    title=f"Canal {self.channels[channel_id]['name']} recuperado",
                                    message="El periodo de recuperación ha finalizado y no se detectan shadowbans.",
                                    level="info"
                                )
                            
                            # Reanudar tareas pausadas
                            tasks_to_resume = []
                            for t_id, t in self.paused_tasks.items():
                                if t['channel_id'] == channel_id and t.get('paused_reason', '').startswith("Shadowban en"):
                                    tasks_to_resume.append(t_id)
                                    t['status'] = 'initiated'
                                    t['priority'] = t.get('priority', Priority.NORMAL)
                                    self.task_queue.put((t['priority'], t))
                                    self.current_tasks[t_id] = t
                                    self._persist_task(t)
                                    logger.info(f"Reanudando tarea {t_id} tras recuperación del canal")
                            
                            for t_id in tasks_to_resume:
                                del self.paused_tasks[t_id]
                
                time.sleep(3600)  # Verificar cada hora
            except Exception as e:
                logger.error(f"Error en monitoreo de recuperación: {str(e)}")
                time.sleep(60)
    
    def _priority_handler(self):
        """Maneja la cola de prioridades y ejecuta tareas"""
        while self.active:
            try:
                priority, task = self.task_queue.get(block=True, timeout=1)
                task_id = task['id']
                
                if task['status'] == 'paused':
                    self.paused_tasks[task_id] = task
                    continue
                    
                if task['type'] == 'content_creation':
                    threading.Thread(target=self._process_content_creation, args=(task_id,), daemon=True).start()
                elif task['type'] == 'channel_optimization':
                    threading.Thread(target=self._process_optimization, args=(task_id,), daemon=True).start()
                elif task['type'] == 'shadowban_recovery':
                    threading.Thread(target=self._process_shadowban_recovery, args=(task_id,), daemon=True).start()
                    
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error en manejador de prioridades: {str(e)}")
    
    def _process_content_creation(self, task_id: str):
        """Procesa una tarea de creación de contenido con reintentos"""
        task = self.current_tasks.get(task_id)
        if not task:
            return
            
        task['status'] = 'processing'
        self._persist_task(task)
        channel_id = task['channel_id']
        channel = self.channels[channel_id]
        
        # Aplicar restricciones del plan de recuperación si existe
        content_restrictions = []
        if channel_id in self.recovery_plans:
            content_restrictions = self.recovery_plans[channel_id]['content_restrictions']
        
        try:
            if task['steps']['trend_detection']['status'] == 'pending':
                task['steps']['trend_detection']['status'] = 'processing'
                self._persist_task(task)
                trends = None
                for attempt in range(self.max_retries + 1):
                    try:
                        trend_radar = self._load_subsystem('trends', 'trend_radar')
                        if trend_radar:
                            trends = trend_radar.detect_trends(channel['niche'])
                            task['steps']['trend_detection']['result'] = trends
                            task['steps']['trend_detection']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Trend radar module not available")
                    except Exception as e:
                        task['steps']['trend_detection']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para trend_detection: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['trend_detection']['status'] = 'failed'
                            task['steps']['trend_detection']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['script_creation']['status'] == 'pending':
                task['steps']['script_creation']['status'] = 'processing'
                self._persist_task(task)
                script = None
                for attempt in range(self.max_retries + 1):
                    try:
                        story_engine = self._load_subsystem('creation', 'story_engine')
                        if story_engine:
                            script = story_engine.create_script(
                                trends=task['steps']['trend_detection']['result'],
                                content_type=task['content_type'],
                                character=channel['character'],
                                restrictions=content_restrictions
                            )
                            task['steps']['script_creation']['result'] = script
                            task['steps']['script_creation']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Story engine module not available")
                    except Exception as e:
                        task['steps']['script_creation']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para script_creation: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['script_creation']['status'] = 'failed'
                            task['steps']['script_creation']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['visual_generation']['status'] == 'pending':
                task['steps']['visual_generation']['status'] = 'processing'
                self._persist_task(task)
                visuals = None
                for attempt in range(self.max_retries + 1):
                    try:
                        visual_generator = self._load_subsystem('creation', 'visual_generator')
                        if visual_generator:
                            visuals = visual_generator.create_visuals(
                                script=task['steps']['script_creation']['result'],
                                character=channel['character'],
                                style=self.config.get('characters', {}).get(channel['character'], {}).get('visual_style', 'default'),
                                restrictions=content_restrictions
                            )
                            task['steps']['visual_generation']['result'] = visuals
                            task['steps']['visual_generation']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Visual generator module not available")
                    except Exception as e:
                        task['steps']['visual_generation']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para visual_generation: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['visual_generation']['status'] = 'failed'
                            task['steps']['visual_generation']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['audio_generation']['status'] == 'pending':
                task['steps']['audio_generation']['status'] = 'processing'
                self._persist_task(task)
                audio = None
                for attempt in range(self.max_retries + 1):
                    try:
                        voice_generator = self._load_subsystem('creation', 'voice_generator')
                        if voice_generator:
                            voice_type = self.config.get('characters', {}).get(channel['character'], {}).get('voice_type', 'default')
                            audio = voice_generator.create_audio(
                                script=task['steps']['script_creation']['result'],
                                voice_type=voice_type
                            )
                            task['steps']['audio_generation']['result'] = audio
                            task['steps']['audio_generation']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Voice generator module not available")
                    except Exception as e:
                        task['steps']['audio_generation']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para audio_generation: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['audio_generation']['status'] = 'failed'
                            task['steps']['audio_generation']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['video_production']['status'] == 'pending':
                task['steps']['video_production']['status'] = 'processing'
                self._persist_task(task)
                video = None
                for attempt in range(self.max_retries + 1):
                    try:
                        video_composer = self._load_subsystem('creation', 'video_composer')
                        if video_composer:
                            video = video_composer.create_video(
                                visuals=task['steps']['visual_generation']['result'],
                                audio=task['steps']['audio_generation']['result'],
                                script=task['steps']['script_creation']['result']
                            )
                            task['steps']['video_production']['result'] = video
                            task['steps']['video_production']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Video composer module not available")
                    except Exception as e:
                        task['steps']['video_production']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para video_production: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['video_production']['status'] = 'failed'
                            task['steps']['video_production']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['compliance_check']['status'] == 'pending':
                task['steps']['compliance_check']['status'] = 'processing'
                self._persist_task(task)
                compliance_result = None
                for attempt in range(self.max_retries + 1):
                    try:
                        compliance_checker = self._load_subsystem('compliance', 'compliance_checker')
                        if compliance_checker:
                            compliance_result = compliance_checker.check_content(
                                video=task['steps']['video_production']['result'],
                                script=task['steps']['script_creation']['result'],
                                platforms=channel['platforms']
                            )
                            if compliance_result['compliant']:
                                task['steps']['compliance_check']['result'] = compliance_result
                                task['steps']['compliance_check']['status'] = 'completed'
                                break
                            else:
                                corrected_video = compliance_checker.fix_content(
                                    video=task['steps']['video_production']['result'],
                                    issues=compliance_result['issues']
                                )
                                if corrected_video:
                                    task['steps']['video_production']['result'] = corrected_video
                                    task['steps']['compliance_check']['result'] = {'compliant': True, 'fixed': True}
                                    task['steps']['compliance_check']['status'] = 'completed'
                                    break
                                else:
                                    raise Exception("Could not fix compliance issues")
                        else:
                            task['steps']['compliance_check']['result'] = {'compliant': True, 'assumed': True}
                            task['steps']['compliance_check']['status'] = 'completed'
                            break
                    except Exception as e:
                        task['steps']['compliance_check']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para compliance_check: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['compliance_check']['status'] = 'failed'
                            task['steps']['compliance_check']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['publication']['status'] == 'pending':
                if self._check_shadowban(channel_id):
                    task['status'] = 'paused'
                    task['paused_reason'] = 'Shadowban detected during publication'
                    self.paused_tasks[task_id] = task
                    self._persist_task(task)
                    logger.info(f"Tarea {task_id} pausada por shadowban durante publicación")
                    return
                    
                task['steps']['publication']['status'] = 'processing'
                self._persist_task(task)
                publication_success = False
                for attempt in range(self.max_retries + 1):
                    try:
                        task['steps']['publication']['platforms_results'] = {}
                        for platform in channel['platforms']:
                            platform_adapter = self._load_subsystem('publication', f'{platform}_adapter')
                            if platform_adapter:
                                metadata = {
                                    'title': task['steps']['script_creation']['result'].get('title', f"Video for {channel['name']}"),
                                    'description': task['steps']['script_creation']['result'].get('description', ''),
                                    'tags': task['steps']['script_creation']['result'].get('tags', []),
                                    'category': channel['niche']
                                }
                                
                                publication_result = platform_adapter.publish(
                                    video=task['steps']['video_production']['result'],
                                    metadata=metadata,
                                    channel_id=channel_id
                                )
                                
                                task['steps']['publication']['platforms_results'][platform] = publication_result
                                
                                if publication_result['success']:
                                    logger.info(f"Contenido publicado en {platform}: {publication_result['url']}")
                                else:
                                    logger.error(f"Error al publicar en {platform}: {publication_result['error']}")
                            else:
                                logger.warning(f"Adaptador para {platform} no disponible")
                        
                        if task['steps']['publication']['platforms_results'] and any(
                            result['success'] for result in task['steps']['publication']['platforms_results'].values()
                        ):
                            publication_success = True
                            task['steps']['publication']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Failed to publish on any platform")
                    except Exception as e:
                        task['steps']['publication']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para publication: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['publication']['status'] = 'failed'
                            task['steps']['publication']['error'] = str(e)
                            raise
                self._persist_task(task)
                if not publication_success:
                    raise Exception("Failed to publish content")
            
            if task['steps']['monetization']['status'] == 'pending':
                task['steps']['monetization']['status'] = 'processing'
                self._persist_task(task)
                for attempt in range(self.max_retries + 1):
                    try:
                        revenue_optimizer = self._load_subsystem('monetization', 'revenue_optimizer')
                        if revenue_optimizer:
                            monetization_result = revenue_optimizer.optimize(
                                content_id=task_id,
                                channel_id=channel_id,
                                content_type=task['content_type'],
                                platforms=task['steps']['publication']['platforms_results']
                            )
                            task['steps']['monetization']['result'] = monetization_result
                            task['steps']['monetization']['status'] = 'completed'
                            break
                        else:
                            task['steps']['monetization']['result'] = {'optimized': False, 'reason': 'Module not available'}
                            task['steps']['monetization']['status'] = 'completed'
                            break
                    except Exception as e:
                        task['steps']['monetization']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para monetization: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['monetization']['status'] = 'failed'
                            task['steps']['monetization']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            self.channels[channel_id]['stats']['videos'] += 1
            
            task['status'] = 'completed'
            task['completed_at'] = datetime.datetime.now().isoformat()
            self._persist_task(task)
            
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Contenido creado para {channel['name']}",
                    message=f"Se ha creado y publicado nuevo contenido: {task['steps']['script_creation']['result'].get('title', 'Nuevo video')}",
                    level="info"
                )
            
            knowledge_base = self._load_component('knowledge_base')
            if knowledge_base:
                knowledge_base.save_content(task)
            
            logger.info(f"Tarea completada: {task_id}")
            
        except Exception as e:
            logger.error(f"Error en proceso de creación de contenido: {str(e)}")
            task['status'] = 'failed'
            task['error'] = str(e)
            task['failed_at'] = datetime.datetime.now().isoformat()
            self._persist_task(task)
            
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Error en creación de contenido para {self.channels[task['channel_id']]['name']}",
                    message=f"Error: {str(e)}",
                    level="error"
                )
        
        finally:
            self.task_history.append(task)
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
            if task_id in self.paused_tasks:
                del self.paused_tasks[task_id]
    
    def _process_optimization(self, task_id: str):
        """Procesa una tarea de optimización de canal con reintentos"""
        task = self.current_tasks.get(task_id)
        if not task:
            return
            
        task['status'] = 'processing'
        self._persist_task(task)
        channel_id = task['channel_id']
        channel = self.channels[channel_id]
        
        try:
            if task['steps']['performance_analysis']['status'] == 'pending':
                task['steps']['performance_analysis']['status'] = 'processing'
                self._persist_task(task)
                performance = None
                for attempt in range(self.max_retries + 1):
                    try:
                        analytics_engine = self._load_component('analytics_engine')
                        if analytics_engine:
                            performance = analytics_engine.analyze_channel_performance(channel_id)
                            task['steps']['performance_analysis']['result'] = performance
                            task['steps']['performance_analysis']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Analytics engine not available")
                    except Exception as e:
                        task['steps']['performance_analysis']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para performance_analysis: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['performance_analysis']['status'] = 'failed'
                            task['steps']['performance_analysis']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['competitor_analysis']['status'] == 'pending':
                task['steps']['competitor_analysis']['status'] = 'processing'
                self._persist_task(task)
                competitors = None
                for attempt in range(self.max_retries + 1):
                    try:
                        competitor_analyzer = self._load_subsystem('analysis', 'competitor_analyzer')
                        if competitor_analyzer:
                            competitors = competitor_analyzer.analyze_competitors(
                                niche=channel['niche'],
                                performance=task['steps']['performance_analysis']['result']
                            )
                            task['steps']['competitor_analysis']['result'] = competitors
                            task['steps']['competitor_analysis']['status'] = 'completed'
                            break
                        else:
                            task['steps']['competitor_analysis']['result'] = {'analyzed': False, 'reason': 'Module not available'}
                            task['steps']['competitor_analysis']['status'] = 'completed'
                            break
                    except Exception as e:
                        task['steps']['competitor_analysis']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para competitor_analysis: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['competitor_analysis']['status'] = 'failed'
                            task['steps']['competitor_analysis']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['content_strategy_update']['status'] == 'pending':
                task['steps']['content_strategy_update']['status'] = 'processing'
                self._persist_task(task)
                strategy_update = None
                for attempt in range(self.max_retries + 1):
                    try:
                        decision_engine = self._load_component('decision_engine')
                        if decision_engine:
                            strategy_update = decision_engine.update_content_strategy(
                                channel_id=channel_id,
                                performance=task['steps']['performance_analysis']['result'],
                                competitors=task['steps']['competitor_analysis']['result'] if 'result' in task['steps']['competitor_analysis'] else None
                            )
                            task['steps']['content_strategy_update']['result'] = strategy_update
                            task['steps']['content_strategy_update']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Decision engine not available")
                    except Exception as e:
                        task['steps']['content_strategy_update']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para content_strategy_update: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['content_strategy_update']['status'] = 'failed'
                            task['steps']['content_strategy_update']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['cta_optimization']['status'] == 'pending':
                task['steps']['cta_optimization']['status'] = 'processing'
                self._persist_task(task)
                cta_optimization = None
                for attempt in range(self.max_retries + 1):
                    try:
                        reputation_engine = self._load_subsystem('optimization', 'reputation_engine')
                        if reputation_engine:
                            cta_optimization = reputation_engine.optimize_ctas(
                                channel_id=channel_id,
                                performance=task['steps']['performance_analysis']['result']
                            )
                            task['steps']['cta_optimization']['result'] = cta_optimization
                            task['steps']['cta_optimization']['status'] = 'completed'
                            break
                        else:
                            task['steps']['cta_optimization']['result'] = {'optimized': False, 'reason': 'Module not available'}
                            task['steps']['cta_optimization']['status'] = 'completed'
                            break
                    except Exception as e:
                        task['steps']['cta_optimization']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para cta_optimization: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['cta_optimization']['status'] = 'failed'
                            task['steps']['cta_optimization']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['monetization_optimization']['status'] == 'pending':
                task['steps']['monetization_optimization']['status'] = 'processing'
                self._persist_task(task)
                monetization_optimization = None
                for attempt in range(self.max_retries + 1):
                    try:
                        revenue_optimizer = self._load_subsystem('monetization', 'revenue_optimizer')
                        if revenue_optimizer:
                            monetization_optimization = revenue_optimizer.optimize_channel(
                                channel_id=channel_id,
                                performance=task['steps']['performance_analysis']['result']
                            )
                            task['steps']['monetization_optimization']['result'] = monetization_optimization
                            task['steps']['monetization_optimization']['status'] = 'completed'
                            break
                        else:
                            task['steps']['monetization_optimization']['result'] = {'optimized': False, 'reason': 'Module not available'}
                            task['steps']['monetization_optimization']['status'] = 'completed'
                            break
                    except Exception as e:
                        task['steps']['monetization_optimization']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para monetization_optimization: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['monetization_optimization']['status'] = 'failed'
                            task['steps']['monetization_optimization']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            task['status'] = 'completed'
            task['completed_at'] = datetime.datetime.now().isoformat()
            self._persist_task(task)
            
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Optimización completada para {channel['name']}",
                    message="Se ha optimizado el canal con éxito",
                    level="info"
                )
            
            knowledge_base = self._load_component('knowledge_base')
            if knowledge_base:
                knowledge_base.save_optimization(task)
                knowledge_base.update_channel(channel)
            
            logger.info(f"Tarea de optimización completada: {task_id}")
            
        except Exception as e:
            logger.error(f"Error en proceso de optimización: {str(e)}")
            task['status'] = 'failed'
            task['error'] = str(e)
            task['failed_at'] = datetime.datetime.now().isoformat()
            self._persist_task(task)
            
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Error en optimización para {channel['name']}",
                    message=f"Error: {str(e)}",
                    level="error"
                )
        
        finally:
            self.task_history.append(task)
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
            if task_id in self.paused_tasks:
                del self.paused_tasks[task_id]
    
    def _process_shadowban_recovery(self, task_id: str):
        """Procesa una tarea de recuperación de shadowban"""
        task = self.current_tasks.get(task_id)
        if not task:
            return
            
        task['status'] = 'processing'
        self._persist_task(task)
        channel_id = task['channel_id']
        platform = task['platform']
        channel = self.channels[channel_id]
        
        try:
            if task['steps']['pause_publications']['status'] == 'pending':
                task['steps']['pause_publications']['status'] = 'processing'
                self._persist_task(task)
                for t_id, t in self.current_tasks.items():
                    if t['channel_id'] == channel_id and t['type'] == 'content_creation':
                        if t['steps']['publication']['status'] in ['pending', 'processing']:
                            t['status'] = 'paused'
                            t['paused_reason'] = f"Shadowban en {platform}"
                            self.paused_tasks[t_id] = t
                            self._persist_task(t)
                task['steps']['pause_publications']['status'] = 'completed'
                self._persist_task(task)
            
            if task['steps']['analyze_cause']['status'] == 'pending':
                task['steps']['analyze_cause']['status'] = 'processing'
                self._persist_task(task)
                shadowban_detector = self._load_component('shadowban_detector')
                cause = None
                for attempt in range(self.max_retries + 1):
                    try:
                        if shadowban_detector:
                            cause = shadowban_detector.analyze_shadowban_cause(channel_id, platform)
                            task['steps']['analyze_cause']['result'] = cause
                            task['steps']['analyze_cause']['status'] = 'completed'
                            break
                        else:
                            raise Exception("Shadowban detector not available")
                    except Exception as e:
                        task['steps']['analyze_cause']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para analyze_cause: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['analyze_cause']['status'] = 'failed'
                            task['steps']['analyze_cause']['error'] = str(e)
                            raise
                self._persist_task(task)
            
            if task['steps']['implement_fix']['status'] == 'pending':
                task['steps']['implement_fix']['status'] = 'processing'
                self._persist_task(task)
                # Crear plan de recuperación
                recovery_plan = self._create_recovery_plan(channel_id)
                fix_success = False
                for attempt in range(self.max_retries + 1):
                    try:
                        shadowban_detector = self._load_component('shadowban_detector')
                        if shadowban_detector:
                            fix_result = shadowban_detector.implement_shadowban_fix(
                                channel_id=channel_id,
                                platform=platform,
                                cause=task['steps']['analyze_cause']['result'],
                                recovery_plan=recovery_plan
                            )
                            task['steps']['implement_fix']['result'] = fix_result
                            task['steps']['implement_fix']['status'] = 'completed'
                            fix_success = True
                            break
                        else:
                            raise Exception("Shadowban detector not available")
                    except Exception as e:
                        task['steps']['implement_fix']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para implement_fix: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['implement_fix']['status'] = 'failed'
                            task['steps']['implement_fix']['error'] = str(e)
                            raise
                self._persist_task(task)
                if not fix_success:
                    raise Exception("Failed to implement shadowban fix")
            
            if task['steps']['verify_resolution']['status'] == 'pending':
                task['steps']['verify_resolution']['status'] = 'processing'
                self._persist_task(task)
                resolved = False
                for attempt in range(self.max_retries + 1):
                    try:
                        shadowban_detector = self._load_component('shadowban_detector')
                        if shadowban_detector:
                            result = shadowban_detector.check_shadowban(channel_id, platform)
                            if not result['shadowbanned']:
                                resolved = True
                                self.shadowban_status[f"{channel_id}_{platform}"]['active'] = False
                                task['steps']['verify_resolution']['result'] = {'resolved': True}
                                task['steps']['verify_resolution']['status'] = 'completed'
                                # Marcar shadowban como resuelto en el historial
                                for entry in self.shadowban_history[channel_id]:
                                    if entry['platform'] == platform and not entry['resolved']:
                                        entry['resolved'] = True
                                        entry['resolved_at'] = datetime.datetime.now().isoformat()
                                        break
                                break
                            else:
                                raise Exception("Shadowban persists")
                        else:
                            raise Exception("Shadowban detector not available")
                    except Exception as e:
                        task['steps']['verify_resolution']['retries'] = attempt + 1
                        if attempt < self.max_retries:
                            delay = self._calculate_retry_delay(attempt)
                            logger.warning(f"Reintento {attempt + 1}/{self.max_retries} para verify_resolution: {str(e)}, esperando {delay}s")
                            time.sleep(delay)
                        else:
                            task['steps']['verify_resolution']['status'] = 'failed'
                            task['steps']['verify_resolution']['error'] = str(e)
                            raise
                self._persist_task(task)
                if not resolved:
                    raise Exception("Failed to resolve shadowban")
            
            task['status'] = 'completed'
            task['completed_at'] = datetime.datetime.now().isoformat()
            self._persist_task(task)
            
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Shadowban resuelto para {channel['name']} en {platform}",
                    message="Se han implementado las correcciones necesarias",
                    level="info"
                )
            
            logger.info(f"Tarea de recuperación de shadowban completada: {task_id}")
            
        except Exception as e:
            logger.error(f"Error en proceso de recuperación de shadowban: {str(e)}")
            task['status'] = 'failed'
            task['error'] = str(e)
            task['failed_at'] = datetime.datetime.now().isoformat()
            self._persist_task(task)
            
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Error en recuperación de shadowban para {self.channels[channel_id]['name']} en {platform}",
                    message=f"Error: {str(e)}",
                    level="error"
                )
        
        finally:
            self.task_history.append(task)
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
            if task_id in self.paused_tasks:
                del self.paused_tasks[task_id]

# Punto de entrada para pruebas
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    orchestrator = Orchestrator()
    success = orchestrator.start()
    print(f"Orchestrator iniciado: {success}")
    
    if success and orchestrator.channels:
        channel_id = list(orchestrator.channels.keys())[0]
        result = orchestrator.create_content(channel_id)
        print(f"Tarea creada: {result}")
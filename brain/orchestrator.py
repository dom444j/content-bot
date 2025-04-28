"""
Orchestrator - Coordinador central del sistema de monetización

Este módulo actúa como el cerebro del sistema, coordinando todos los subsistemas:
- Detección de tendencias
- Creación de contenido
- Verificación de cumplimiento
- Publicación en plataformas
- Monetización
- Análisis y optimización
"""

import os
import sys
import logging
import time
import json
import datetime
import threading
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
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar componentes (se cargarán dinámicamente según necesidad)
        self.components = {
            'decision_engine': None,
            'scheduler': None,
            'notifier': None,
            'knowledge_base': None,
            'analytics_engine': None
        }
        
        # Inicializar subsistemas (se cargarán dinámicamente según necesidad)
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
            # Cargar configuración de plataformas
            with open(os.path.join(config_dir, 'platforms.json'), 'r', encoding='utf-8') as f:
                config['platforms'] = json.load(f)
            
            # Cargar configuración de estrategias
            with open(os.path.join(config_dir, 'strategy.json'), 'r', encoding='utf-8') as f:
                config['strategy'] = json.load(f)
            
            # Cargar configuración de nichos
            with open(os.path.join(config_dir, 'niches.json'), 'r', encoding='utf-8') as f:
                config['niches'] = json.load(f)
                
            # Cargar perfiles de personajes
            with open(os.path.join(config_dir, 'character_profiles.json'), 'r', encoding='utf-8') as f:
                config['characters'] = json.load(f)
                
            logger.info("Configuración cargada correctamente")
            return config
            
        except FileNotFoundError as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            # Crear configuración por defecto si no existe
            self._create_default_config()
            return self._load_config()
    
    def _create_default_config(self):
        """Crea archivos de configuración por defecto si no existen"""
        config_dir = os.path.join('config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Crear directorio de logs si no existe
        logs_dir = os.path.join('logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Configuración de plataformas por defecto
        platforms_config = {
            "youtube": {"api_key": "", "channel_id": "", "enabled": True},
            "tiktok": {"api_key": "", "account_id": "", "enabled": True},
            "instagram": {"api_key": "", "account_id": "", "enabled": True},
            "threads": {"api_key": "", "account_id": "", "enabled": False},
            "bluesky": {"api_key": "", "handle": "", "enabled": False}
        }
        
        # Configuración de estrategia por defecto
        strategy_config = {
            "cta_timing": {"early": [0, 3], "middle": [4, 8], "end": [-2, 0]},
            "content_types": ["educational", "entertainment", "tutorial"],
            "monetization_priority": ["affiliate", "ads", "products"]
        }
        
        # Configuración de nichos por defecto
        niches_config = {
            "finance": {"keywords": ["crypto", "investing", "money"], "enabled": True},
            "health": {"keywords": ["fitness", "mindfulness", "nutrition"], "enabled": True},
            "technology": {"keywords": ["ai", "gadgets", "programming"], "enabled": True},
            "gaming": {"keywords": ["strategy", "review", "gameplay"], "enabled": True},
            "humor": {"keywords": ["memes", "comedy", "funny"], "enabled": True}
        }
        
        # Perfiles de personajes por defecto
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
        
        # Guardar configuraciones por defecto
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
                # Importar dinámicamente
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
            # Importar dinámicamente según el subsistema y módulo
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
            
            # Otros subsistemas se cargarían de manera similar
            # ...
            
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
            
            # Verificar configuración
            if not self._verify_config():
                logger.error("Configuración incompleta o inválida")
                return False
            
            # Iniciar canales configurados
            self._initialize_channels()
            
            # Marcar como activo
            self.active = True
            logger.info("Orchestrator iniciado correctamente")
            
            # Iniciar bucle principal en un hilo separado
            threading.Thread(target=self._main_loop, daemon=True).start()
            
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
            # Detener tareas en curso
            for task_id, task in self.current_tasks.items():
                logger.info(f"Deteniendo tarea {task_id}")
                # Implementar lógica para detener tareas
            
            # Marcar como inactivo
            self.active = False
            logger.info("Orchestrator detenido correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al detener Orchestrator: {str(e)}")
            return False
    
    def _verify_config(self) -> bool:
        """Verifica que la configuración sea válida y completa"""
        # Verificar configuración mínima necesaria
        if not self.config:
            return False
        
        # Verificar que al menos una plataforma esté habilitada
        platforms_enabled = any(p.get('enabled', False) for p in self.config.get('platforms', {}).values())
        if not platforms_enabled:
            logger.warning("No hay plataformas habilitadas en la configuración")
            return False
        
        # Verificar que al menos un nicho esté habilitado
        niches_enabled = any(n.get('enabled', False) for n in self.config.get('niches', {}).values())
        if not niches_enabled:
            logger.warning("No hay nichos habilitados en la configuración")
            return False
        
        return True
    
    def _initialize_channels(self):
        """Inicializa los canales configurados"""
        # Obtener canales de la base de conocimiento
        knowledge_base = self._load_component('knowledge_base')
        if knowledge_base:
            self.channels = knowledge_base.get_channels()
            
        # Si no hay canales, crear los predeterminados según la configuración
        if not self.channels:
            logger.info("No se encontraron canales existentes, creando canales predeterminados")
            self._create_default_channels()
    
    def _create_default_channels(self):
        """Crea canales predeterminados basados en la configuración"""
        # Crear un canal por cada nicho habilitado
        for niche_name, niche_data in self.config.get('niches', {}).items():
            if niche_data.get('enabled', False):
                channel_id = f"{niche_name}_{int(time.time())}"
                
                # Crear canal para cada plataforma habilitada
                platforms = []
                for platform_name, platform_data in self.config.get('platforms', {}).items():
                    if platform_data.get('enabled', False):
                        platforms.append(platform_name)
                
                # Seleccionar personaje adecuado para el nicho
                character = None
                for char_name, char_data in self.config.get('characters', {}).items():
                    if niche_name in char_name:
                        character = char_name
                        break
                
                # Si no hay personaje específico, usar el primero disponible
                if not character and self.config.get('characters'):
                    character = list(self.config.get('characters').keys())[0]
                
                # Crear configuración del canal
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
                
                # Guardar en la base de conocimiento
                knowledge_base = self._load_component('knowledge_base')
                if knowledge_base:
                    knowledge_base.save_channel(self.channels[channel_id])
    
    def create_content(self, channel_id: str, content_type: str = None) -> Dict:
        """
        Inicia el proceso de creación de contenido para un canal específico
        
        Args:
            channel_id: ID del canal para el que se creará contenido
            content_type: Tipo de contenido a crear (opcional)
            
        Returns:
            Dict con información sobre la tarea creada
        """
        if not self.active:
            logger.error("El orquestador no está activo")
            return {'success': False, 'error': 'Orchestrator not active'}
        
        if channel_id not in self.channels:
            logger.error(f"Canal no encontrado: {channel_id}")
            return {'success': False, 'error': 'Channel not found'}
        
        logger.info(f"Iniciando creación de contenido para canal: {channel_id}")
        
        try:
            # Generar ID único para la tarea
            task_id = f"content_{channel_id}_{int(time.time())}"
            
            # Obtener información del canal
            channel = self.channels[channel_id]
            
            # Determinar tipo de contenido si no se especificó
            if not content_type:
                decision_engine = self._load_component('decision_engine')
                if decision_engine:
                    content_type = decision_engine.decide_content_type(channel)
                else:
                    # Usar tipo predeterminado si no hay motor de decisión
                    content_type = self.config.get('strategy', {}).get('content_types', ['educational'])[0]
            
            # Crear tarea
            task = {
                'id': task_id,
                'type': 'content_creation',
                'channel_id': channel_id,
                'content_type': content_type,
                'status': 'initiated',
                'created_at': datetime.datetime.now().isoformat(),
                'steps': {
                    'trend_detection': {'status': 'pending'},
                    'script_creation': {'status': 'pending'},
                    'visual_generation': {'status': 'pending'},
                    'audio_generation': {'status': 'pending'},
                    'video_production': {'status': 'pending'},
                    'compliance_check': {'status': 'pending'},
                    'publication': {'status': 'pending'},
                    'monetization': {'status': 'pending'}
                }
            }
            
            # Registrar tarea
            self.current_tasks[task_id] = task
            
            # Iniciar proceso asíncrono
            threading.Thread(target=self._process_content_creation, args=(task_id,), daemon=True).start()
            
            logger.info(f"Tarea de creación de contenido iniciada: {task_id}")
            return {'success': True, 'task_id': task_id, 'task': task}
            
        except Exception as e:
            logger.error(f"Error al iniciar creación de contenido: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_task_status(self, task_id: str) -> Dict:
        """Obtiene el estado actual de una tarea"""
        if task_id in self.current_tasks:
            return {'success': True, 'task': self.current_tasks[task_id]}
        elif any(task['id'] == task_id for task in self.task_history):
            task = next(task for task in self.task_history if task['id'] == task_id)
            return {'success': True, 'task': task}
        else:
            return {'success': False, 'error': 'Task not found'}
    
    def get_channel_stats(self, channel_id: str) -> Dict:
        """Obtiene estadísticas de un canal específico"""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        # Cargar analíticas actualizadas
        analytics_engine = self._load_component('analytics_engine')
        if analytics_engine:
            stats = analytics_engine.get_channel_stats(channel_id)
            if stats:
                # Actualizar estadísticas en el canal
                self.channels[channel_id]['stats'] = stats
        
        return {'success': True, 'channel': self.channels[channel_id]}
    
    def optimize_channel(self, channel_id: str) -> Dict:
        """Inicia el proceso de optimización para un canal específico"""
        if not self.active:
            return {'success': False, 'error': 'Orchestrator not active'}
        
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        logger.info(f"Iniciando optimización para canal: {channel_id}")
        
        try:
            # Generar ID único para la tarea
            task_id = f"optimize_{channel_id}_{int(time.time())}"
            
            # Crear tarea
            task = {
                'id': task_id,
                'type': 'channel_optimization',
                'channel_id': channel_id,
                'status': 'initiated',
                'created_at': datetime.datetime.now().isoformat(),
                'steps': {
                    'performance_analysis': {'status': 'pending'},
                    'competitor_analysis': {'status': 'pending'},
                    'content_strategy_update': {'status': 'pending'},
                    'cta_optimization': {'status': 'pending'},
                    'monetization_optimization': {'status': 'pending'}
                }
            }
            
            # Registrar tarea
            self.current_tasks[task_id] = task
            
            # Iniciar proceso asíncrono
            threading.Thread(target=self._process_optimization, args=(task_id,), daemon=True).start()
            
            logger.info(f"Tarea de optimización iniciada: {task_id}")
            return {'success': True, 'task_id': task_id, 'task': task}
            
        except Exception as e:
            logger.error(f"Error al iniciar optimización: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_content_creation(self, task_id: str):
        """Procesa una tarea de creación de contenido"""
        task = self.current_tasks.get(task_id)
        if not task:
            return
        
        # Actualizar estado
        task['status'] = 'processing'
        
        try:
            # 1. Detección de tendencias
            task['steps']['trend_detection']['status'] = 'processing'
            trend_radar = self._load_subsystem('trends', 'trend_radar')
            if trend_radar:
                channel_id = task['channel_id']
                channel = self.channels[channel_id]
                trends = trend_radar.detect_trends(channel['niche'])
                task['steps']['trend_detection']['result'] = trends
                task['steps']['trend_detection']['status'] = 'completed'
            else:
                task['steps']['trend_detection']['status'] = 'failed'
                task['steps']['trend_detection']['error'] = 'Trend radar module not available'
                raise Exception("Failed to load trend radar module")
            
            # 2. Creación de guion
            task['steps']['script_creation']['status'] = 'processing'
            story_engine = self._load_subsystem('creation', 'story_engine')
            if story_engine:
                script = story_engine.create_script(
                    trends=trends,
                    content_type=task['content_type'],
                    character=channel['character']
                )
                task['steps']['script_creation']['result'] = script
                task['steps']['script_creation']['status'] = 'completed'
            else:
                task['steps']['script_creation']['status'] = 'failed'
                task['steps']['script_creation']['error'] = 'Story engine module not available'
                raise Exception("Failed to load story engine module")
            
            # 3. Generación visual
            task['steps']['visual_generation']['status'] = 'processing'
            visual_generator = self._load_subsystem('creation', 'visual_generator')
            if visual_generator:
                visuals = visual_generator.create_visuals(
                    script=script,
                    character=channel['character'],
                    style=self.config.get('characters', {}).get(channel['character'], {}).get('visual_style', 'default')
                )
                task['steps']['visual_generation']['result'] = visuals
                task['steps']['visual_generation']['status'] = 'completed'
            else:
                task['steps']['visual_generation']['status'] = 'failed'
                task['steps']['visual_generation']['error'] = 'Visual generator module not available'
                raise Exception("Failed to load visual generator module")
            
            # 4. Generación de audio
            task['steps']['audio_generation']['status'] = 'processing'
            voice_generator = self._load_subsystem('creation', 'voice_generator')
            if voice_generator:
                voice_type = self.config.get('characters', {}).get(channel['character'], {}).get('voice_type', 'default')
                audio = voice_generator.create_audio(
                    script=script,
                    voice_type=voice_type
                )
                task['steps']['audio_generation']['result'] = audio
                task['steps']['audio_generation']['status'] = 'completed'
            else:
                task['steps']['audio_generation']['status'] = 'failed'
                task['steps']['audio_generation']['error'] = 'Voice generator module not available'
                raise Exception("Failed to load voice generator module")
            
            # 5. Producción de video
            task['steps']['video_production']['status'] = 'processing'
            video_composer = self._load_subsystem('creation', 'video_composer')
            if video_composer:
                video = video_composer.create_video(
                    visuals=visuals,
                    audio=audio,
                    script=script
                )
                task['steps']['video_production']['result'] = video
                task['steps']['video_production']['status'] = 'completed'
            else:
                task['steps']['video_production']['status'] = 'failed'
                task['steps']['video_production']['error'] = 'Video composer module not available'
                raise Exception("Failed to load video composer module")
            
            # 6. Verificación de cumplimiento
            task['steps']['compliance_check']['status'] = 'processing'
            compliance_checker = self._load_subsystem('compliance', 'compliance_checker')
            if compliance_checker:
                compliance_result = compliance_checker.check_content(
                    video=video,
                    script=script,
                    platforms=channel['platforms']
                )
                if compliance_result['compliant']:
                    task['steps']['compliance_check']['result'] = compliance_result
                    task['steps']['compliance_check']['status'] = 'completed'
                else:
                    # Si no cumple, intentar corregir
                    corrected_video = compliance_checker.fix_content(
                        video=video,
                        issues=compliance_result['issues']
                    )
                    if corrected_video:
                        video = corrected_video
                        task['steps']['compliance_check']['result'] = {'compliant': True, 'fixed': True}
                        task['steps']['compliance_check']['status'] = 'completed'
                    else:
                        task['steps']['compliance_check']['status'] = 'failed'
                        task['steps']['compliance_check']['error'] = 'Could not fix compliance issues'
                        raise Exception("Failed to fix compliance issues")
            else:
                # Si no hay verificador de cumplimiento, asumir que cumple
                task['steps']['compliance_check']['result'] = {'compliant': True, 'assumed': True}
                task['steps']['compliance_check']['status'] = 'completed'
            
            # 7. Publicación
            task['steps']['publication']['status'] = 'processing'
            for platform in channel['platforms']:
                platform_adapter = self._load_subsystem('publication', f'{platform}_adapter')
                if platform_adapter:
                    # Preparar metadatos
                    metadata = {
                        'title': script.get('title', f"Video for {channel['name']}"),
                        'description': script.get('description', ''),
                        'tags': script.get('tags', []),
                        'category': channel['niche']
                    }
                    
                    # Publicar
                    publication_result = platform_adapter.publish(
                        video=video,
                        metadata=metadata,
                        channel_id=channel_id
                    )
                    
                    # Guardar resultado
                    if 'platforms_results' not in task['steps']['publication']:
                        task['steps']['publication']['platforms_results'] = {}
                    
                    task['steps']['publication']['platforms_results'][platform] = publication_result
                    
                    if publication_result['success']:
                        logger.info(f"Contenido publicado en {platform}: {publication_result['url']}")
                    else:
                        logger.error(f"Error al publicar en {platform}: {publication_result['error']}")
                else:
                    logger.warning(f"Adaptador para {platform} no disponible")
            
            # Verificar si se publicó en al menos una plataforma
            if task['steps']['publication'].get('platforms_results') and any(
                result['success'] for result in task['steps']['publication']['platforms_results'].values()
            ):
                task['steps']['publication']['status'] = 'completed'
            else:
                task['steps']['publication']['status'] = 'failed'
                task['steps']['publication']['error'] = 'Failed to publish on any platform'
                raise Exception("Failed to publish content")
            
            # 8. Monetización
            task['steps']['monetization']['status'] = 'processing'
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
            else:
                # Si no hay optimizador de ingresos, marcar como completado sin acción
                task['steps']['monetization']['result'] = {'optimized': False, 'reason': 'Module not available'}
                task['steps']['monetization']['status'] = 'completed'
            
            # Actualizar estadísticas del canal
            self.channels[channel_id]['stats']['videos'] += 1
            
            # Finalizar tarea
            task['status'] = 'completed'
            task['completed_at'] = datetime.datetime.now().isoformat()
            
            # Notificar
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Contenido creado para {channel['name']}",
                    message=f"Se ha creado y publicado nuevo contenido: {script.get('title', 'Nuevo video')}",
                    level="info"
                )
            
            # Guardar en la base de conocimiento
            knowledge_base = self._load_component('knowledge_base')
            if knowledge_base:
                knowledge_base.save_content(task)
            
            logger.info(f"Tarea completada: {task_id}")
            
        except Exception as e:
            logger.error(f"Error en proceso de creación de contenido: {str(e)}")
            task['status'] = 'failed'
            task['error'] = str(e)
            task['failed_at'] = datetime.datetime.now().isoformat()
            
            # Notificar error
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Error en creación de contenido para {self.channels[task['channel_id']]['name']}",
                    message=f"Error: {str(e)}",
                    level="error"
                )
        
        finally:
            # Mover a historial
            self.task_history.append(task)
            del self.current_tasks[task_id]
    
    def _process_optimization(self, task_id: str):
        """Procesa una tarea de optimización de canal"""
        task = self.current_tasks.get(task_id)
        if not task:
            return
        
        # Actualizar estado
        task['status'] = 'processing'
        channel_id = task['channel_id']
        channel = self.channels[channel_id]
        
        try:
            # 1. Análisis de rendimiento
            task['steps']['performance_analysis']['status'] = 'processing'
            analytics_engine = self._load_component('analytics_engine')
            if analytics_engine:
                performance = analytics_engine.analyze_channel_performance(channel_id)
                task['steps']['performance_analysis']['result'] = performance
                task['steps']['performance_analysis']['status'] = 'completed'
            else:
                task['steps']['performance_analysis']['status'] = 'failed'
                task['steps']['performance_analysis']['error'] = 'Analytics engine not available'
                raise Exception("Failed to load analytics engine")
            
            # 2. Análisis de competencia
            task['steps']['competitor_analysis']['status'] = 'processing'
            competitor_analyzer = self._load_subsystem('analysis', 'competitor_analyzer')
            if competitor_analyzer:
                competitors = competitor_analyzer.analyze_competitors(
                    niche=channel['niche'],
                    performance=performance
                )
                task['steps']['competitor_analysis']['result'] = competitors
                task['steps']['competitor_analysis']['status'] = 'completed'
            else:
                # Si no hay analizador de competencia, continuar sin él
                task['steps']['competitor_analysis']['result'] = {'analyzed': False, 'reason': 'Module not available'}
                task['steps']['competitor_analysis']['status'] = 'completed'
            
            # 3. Actualización de estrategia de contenido
            task['steps']['content_strategy_update']['status'] = 'processing'
            decision_engine = self._load_component('decision_engine')
            if decision_engine:
                strategy_update = decision_engine.update_content_strategy(
                    channel_id=channel_id,
                    performance=performance,
                    competitors=task['steps']['competitor_analysis']['result'] if 'result' in task['steps']['competitor_analysis'] else None
                )
                task['steps']['content_strategy_update']['result'] = strategy_update
                task['steps']['content_strategy_update']['status'] = 'completed'
            else:
                task['steps']['content_strategy_update']['status'] = 'failed'
                task['steps']['content_strategy_update']['error'] = 'Decision engine not available'
                raise Exception("Failed to load decision engine")
            
            # 4. Optimización de CTAs
            task['steps']['cta_optimization']['status'] = 'processing'
            reputation_engine = self._load_subsystem('optimization', 'reputation_engine')
            if reputation_engine:
                cta_optimization = reputation_engine.optimize_ctas(
                    channel_id=channel_id,
                    performance=performance
                )
                task['steps']['cta_optimization']['result'] = cta_optimization
                task['steps']['cta_optimization']['status'] = 'completed'
            else:
                # Si no hay motor de reputación, continuar sin él
                task['steps']['cta_optimization']['result'] = {'optimized': False, 'reason': 'Module not available'}
                task['steps']['cta_optimization']['status'] = 'completed'
            
            # 5. Optimización de monetización
            task['steps']['monetization_optimization']['status'] = 'processing'
            revenue_optimizer = self._load_subsystem('monetization', 'revenue_optimizer')
            if revenue_optimizer:
                monetization_optimization = revenue_optimizer.optimize_channel(
                    channel_id=channel_id,
                    performance=performance
                )
                task['steps']['monetization_optimization']['result'] = monetization_optimization
                task['steps']['monetization_optimization']['status'] = 'completed'
            else:
                # Si no hay optimizador de ingresos, continuar sin él
                task['steps']['monetization_optimization']['result'] = {'optimized': False, 'reason': 'Module not available'}
                task['steps']['monetization_optimization']['status'] = 'completed'
            
            # Finalizar tarea
            task['status'] = 'completed'
            task['completed_at'] = datetime.datetime.now().isoformat()
            
            # Notificar
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Optimización completada para {channel['name']}",
                    message="Se ha optimizado el canal con éxito",
                    level="info"
                )
            
            # Guardar en la base de conocimiento
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
            
            # Notificar error
            notifier = self._load_component('notifier')
            if notifier:
                notifier.send_notification(
                    title=f"Error en optimización para {channel['name']}",
                    message=f"Error: {str(e)}",
                    level="error"
                )
        
        finally:
            # Mover a historial
            self.task_history.append(task)
            del self.current_tasks[task_id]

# Punto de entrada para pruebas
if __name__ == "__main__":
    # Crear directorios necesarios si no existen
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Inicializar orquestador
    orchestrator = Orchestrator()
    
    # Iniciar orquestador
    success = orchestrator.start()
    print(f"Orchestrator iniciado: {success}")
    
    # Crear contenido para un canal (simulación)
    if success and orchestrator.channels:
        channel_id = list(orchestrator.channels.keys())[0]
        result = orchestrator.create_content(channel_id)
        print(f"Tarea creada: {result}")
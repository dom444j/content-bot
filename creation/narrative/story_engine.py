"""
Story Engine - Motor de narrativas y arcos argumentales

Este módulo gestiona la creación de estructuras narrativas:
- Arcos argumentales para diferentes formatos y duraciones
- Estructuras de historia optimizadas para engagement
- Generación de narrativas adaptadas a nichos específicos
- Integración con CTAs estratégicos en momentos óptimos
- Soporte para series y narrativas multi-episodio
"""

import os
import sys
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
import datetime

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase
from brain.decision_engine import DecisionEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'story_engine.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('StoryEngine')

class StoryEngine:
    """
    Motor de narrativas que genera estructuras de historia optimizadas
    para diferentes plataformas, duraciones y nichos.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StoryEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el motor de narrativas si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Story Engine...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar motor de decisiones
        self.decision_engine = DecisionEngine()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Cargar plantillas de arcos narrativos
        self.plotlines_file = os.path.join('creation', 'narrative', 'plotlines.json')
        self.plotlines = self._load_plotlines()
        
        # Cargar plantillas de CTAs
        self.cta_templates_file = os.path.join('creation', 'narrative', 'cta_templates.json')
        self.cta_templates = self._load_cta_templates()
        
        # Configuración de duraciones por plataforma
        self.platform_durations = {
            'tiktok': {
                'short': (15, 30),    # 15-30 segundos
                'medium': (31, 60),   # 31-60 segundos
                'long': (61, 180)     # 61-180 segundos
            },
            'instagram': {
                'short': (15, 30),    # 15-30 segundos (Reels)
                'medium': (31, 60),   # 31-60 segundos (Reels)
                'long': (61, 90)      # 61-90 segundos (Reels)
            },
            'youtube_shorts': {
                'short': (15, 30),    # 15-30 segundos
                'medium': (31, 60),   # 31-60 segundos
                'long': (61, 60)      # Exactamente 60 segundos (óptimo para Shorts)
            },
            'youtube': {
                'short': (60, 180),   # 1-3 minutos
                'medium': (181, 600), # 3-10 minutos
                'long': (601, 1200)   # 10-20 minutos
            },
            # Añadimos configuración para series
            'series': {
                'short_episode': (180, 300),    # 3-5 minutos
                'medium_episode': (301, 600),   # 5-10 minutos
                'long_episode': (601, 1200),    # 10-20 minutos
                'mini_series': (5, 10),         # 5-10 episodios
                'standard_series': (10, 20),    # 10-20 episodios
                'long_series': (20, 50)         # 20-50 episodios
            }
        }
        
        # Configuración de estructuras narrativas por duración
        self.narrative_structures = {
            'short': {  # 15-60 segundos
                'hook': (0, 3),       # 0-3 segundos
                'development': (4, 8), # 4-8 segundos
                'cta_climax': (9, 15), # 9-15 segundos
                'segments': 3         # Número de segmentos narrativos
            },
            'medium': { # 1-3 minutos
                'hook': (0, 5),       # 0-5 segundos
                'development': (6, 30), # 6-30 segundos
                'cta_climax': (31, 60), # 31-60 segundos
                'segments': 5         # Número de segmentos narrativos
            },
            'long': {   # 3+ minutos
                'hook': (0, 10),      # 0-10 segundos
                'development': (11, 120), # 11-120 segundos
                'cta_climax': (121, 180), # 121-180 segundos
                'segments': 7         # Número de segmentos narrativos
            },
            # Añadimos estructura para episodios de series
            'episode': {
                'recap': (0, 30),           # 0-30 segundos para recapitulación
                'intro': (31, 60),          # 30-60 segundos para introducción
                'development': (61, 240),   # 1-4 minutos para desarrollo
                'climax': (241, 270),       # 4-4.5 minutos para clímax
                'cliffhanger': (271, 300),  # 4.5-5 minutos para cliffhanger/gancho
                'segments': 9               # Número de segmentos narrativos
            }
        }
        
        # Configuración de timing de CTAs por duración
        self.cta_timing = {
            'short': [
                {'position': 'early', 'time_range': (0, 3), 'weight': 0.2},
                {'position': 'middle', 'time_range': (4, 8), 'weight': 0.6},
                {'position': 'end', 'time_range': (9, 15), 'weight': 0.2}
            ],
            'medium': [
                {'position': 'early', 'time_range': (0, 10), 'weight': 0.1},
                {'position': 'middle', 'time_range': (11, 30), 'weight': 0.7},
                {'position': 'end', 'time_range': (31, 60), 'weight': 0.2}
            ],
            'long': [
                {'position': 'early', 'time_range': (0, 30), 'weight': 0.1},
                {'position': 'middle', 'time_range': (31, 120), 'weight': 0.6},
                {'position': 'end', 'time_range': (121, 180), 'weight': 0.3}
            ],
            # Añadimos timing para episodios de series
            'episode': [
                {'position': 'pre_intro', 'time_range': (0, 15), 'weight': 0.1},
                {'position': 'post_recap', 'time_range': (30, 45), 'weight': 0.2},
                {'position': 'mid_episode', 'time_range': (120, 180), 'weight': 0.4},
                {'position': 'pre_cliffhanger', 'time_range': (240, 270), 'weight': 0.3}
            ]
        }
        
        # Configuración de arcos narrativos para series
        self.series_arc_types = {
            'educational': {
                'arc_length': 'flexible',  # Puede ser corto o largo según el tema
                'episode_structure': 'modular',  # Cada episodio puede verse independientemente
                'continuity_level': 'low',  # Baja dependencia entre episodios
                'engagement_hooks': ['curiosity', 'practical_value', 'knowledge_gaps']
            },
            'biblical_values': {
                'arc_length': 'episodic',  # Cada historia bíblica puede ser un episodio
                'episode_structure': 'moral_lesson',  # Estructura centrada en la enseñanza moral
                'continuity_level': 'medium',  # Personajes recurrentes pero historias independientes
                'engagement_hooks': ['moral_dilemmas', 'character_growth', 'divine_intervention']
            },
            'ai_learning': {
                'arc_length': 'progressive',  # Conceptos que se construyen uno sobre otro
                'episode_structure': 'concept_application',  # Explicación + aplicación práctica
                'continuity_level': 'high',  # Alta dependencia entre episodios
                'engagement_hooks': ['future_tech', 'practical_applications', 'ethical_questions']
            },
            'mystery': {
                'arc_length': 'season',  # Un misterio principal por temporada
                'episode_structure': 'clue_revelation',  # Cada episodio revela nuevas pistas
                'continuity_level': 'very_high',  # Episodios muy dependientes entre sí
                'engagement_hooks': ['suspense', 'plot_twists', 'character_secrets']
            },
            'comedy': {
                'arc_length': 'situational',  # Situaciones cómicas que pueden ser independientes
                'episode_structure': 'setup_punchline',  # Preparación y remate cómico
                'continuity_level': 'low_to_medium',  # Algunos chistes recurrentes
                'engagement_hooks': ['relatability', 'absurdity', 'character_quirks']
            },
            'dance_trends': {
                'arc_length': 'viral',  # Basado en tendencias virales
                'episode_structure': 'tutorial_showcase',  # Tutorial + demostración
                'continuity_level': 'low',  # Poca dependencia entre episodios
                'engagement_hooks': ['novelty', 'skill_progression', 'community_participation']
            }
        }
        
        self._initialized = True
        logger.info("Story Engine inicializado correctamente")
    
    def _load_strategy(self) -> Dict:
        """Carga la configuración de estrategia desde el archivo JSON"""
        try:
            if os.path.exists(self.strategy_file):
                with open(self.strategy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estrategia no encontrado: {self.strategy_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar estrategia: {str(e)}")
            return {}
    
    def _load_plotlines(self) -> Dict:
        """Carga las plantillas de arcos narrativos desde el archivo JSON"""
        try:
            if os.path.exists(self.plotlines_file):
                with open(self.plotlines_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de plotlines no encontrado: {self.plotlines_file}")
                return self._create_default_plotlines()
        except Exception as e:
            logger.error(f"Error al cargar plotlines: {str(e)}")
            return self._create_default_plotlines()
    
    def _create_default_plotlines(self) -> Dict:
        """Crea plantillas de arcos narrativos por defecto"""
        default_plotlines = {
            # Mantenemos los plotlines existentes
            "finance": {
                "problem_solution": {
                    "hook": "Presentar un problema financiero común",
                    "development": "Explicar por qué ocurre y sus consecuencias",
                    "climax": "Revelar la solución y sus beneficios",
                    "segments": [
                        "Problema financiero",
                        "Causas del problema",
                        "Consecuencias",
                        "Solución propuesta",
                        "Beneficios inmediatos",
                        "Resultados a largo plazo",
                        "Llamada a la acción"
                    ]
                },
                # Otros plotlines de finanzas...
            },
            "health": {
                # Plotlines de salud...
            },
            "technology": {
                # Plotlines de tecnología...
            },
            "gaming": {
                # Plotlines de gaming...
            },
            "humor": {
                # Plotlines de humor...
            },
            
            # Añadimos nuevos plotlines para series
            "biblical_values": {
                "moral_lesson": {
                    "hook": "Presentar un dilema moral o situación bíblica",
                    "development": "Explorar las decisiones y consecuencias",
                    "climax": "Revelar la enseñanza moral o espiritual",
                    "segments": [
                        "Presentación del personaje bíblico",
                        "Contexto histórico y cultural",
                        "Desafío o prueba de fe",
                        "Decisiones y acciones",
                        "Intervención divina",
                        "Consecuencias y aprendizaje",
                        "Aplicación a la vida moderna",
                        "Reflexión final",
                        "Llamada a la acción"
                    ]
                },
                "parables": {
                    "hook": "Introducir una situación cotidiana con significado profundo",
                    "development": "Desarrollar la historia con elementos simbólicos",
                    "climax": "Revelar el significado espiritual oculto",
                    "segments": [
                        "Situación cotidiana",
                        "Personajes simbólicos",
                        "Desarrollo de la metáfora",
                        "Giro inesperado",
                        "Revelación del significado",
                        "Enseñanza espiritual",
                        "Aplicación práctica",
                        "Reflexión guiada",
                        "Llamada a la acción"
                    ]
                },
                "hero_journey": {
                    "hook": "Presentar un personaje bíblico en su entorno normal",
                    "development": "Mostrar su llamado, pruebas y transformación",
                    "climax": "Revelar su triunfo y legado espiritual",
                    "segments": [
                        "Vida ordinaria",
                        "Llamado divino",
                        "Resistencia inicial",
                        "Mentores y ayudantes",
                        "Pruebas de fe",
                        "Crisis espiritual",
                        "Transformación",
                        "Triunfo y legado",
                        "Llamada a la acción"
                    ]
                }
            },
            "ai_learning": {
                "concept_introduction": {
                    "hook": "Plantear un problema que la IA puede resolver",
                    "development": "Explicar el concepto de IA relevante",
                    "climax": "Mostrar la aplicación práctica y beneficios",
                    "segments": [
                        "Problema del mundo real",
                        "Limitaciones de enfoques tradicionales",
                        "Introducción al concepto de IA",
                        "Funcionamiento básico",
                        "Ejemplos prácticos",
                        "Beneficios y limitaciones",
                        "Futuro del concepto",
                        "Recursos para aprender más",
                        "Llamada a la acción"
                    ]
                },
                "ai_history": {
                    "hook": "Presentar un hito sorprendente en la historia de la IA",
                    "development": "Explorar la evolución histórica del concepto",
                    "climax": "Conectar con el estado actual y futuro de la tecnología",
                    "segments": [
                        "Momento histórico clave",
                        "Contexto tecnológico de la época",
                        "Pioneros y visionarios",
                        "Desafíos superados",
                        "Avances incrementales",
                        "Revoluciones conceptuales",
                        "Estado actual",
                        "Proyección futura",
                        "Llamada a la acción"
                    ]
                },
                "ethical_dilemma": {
                    "hook": "Presentar un dilema ético provocado por la IA",
                    "development": "Explorar diferentes perspectivas y consecuencias",
                    "climax": "Proponer marcos para abordar estos dilemas",
                    "segments": [
                        "Escenario ético problemático",
                        "Stakeholders afectados",
                        "Perspectiva tecnológica",
                        "Perspectiva humanista",
                        "Perspectiva legal",
                        "Casos de estudio reales",
                        "Marcos éticos aplicables",
                        "Soluciones potenciales",
                        "Llamada a la acción"
                    ]
                }
            },
            "mystery_series": {
                "detective_case": {
                    "hook": "Presentar un crimen o misterio intrigante",
                    "development": "Seguir la investigación y descubrimiento de pistas",
                    "climax": "Revelar una pista crucial o giro inesperado",
                    "cliffhanger": "Dejar una pregunta sin resolver para el siguiente episodio",
                    "segments": [
                        "Escena del crimen/misterio",
                        "Introducción del detective/protagonista",
                        "Primeras pistas e hipótesis",
                        "Entrevistas con testigos/sospechosos",
                        "Falsa pista o callejón sin salida",
                        "Revelación inesperada",
                        "Nueva teoría o enfoque",
                        "Descubrimiento crucial",
                        "Gancho para el siguiente episodio"
                    ]
                },
                "conspiracy_unraveling": {
                    "hook": "Sugerir que hay una verdad oculta detrás de eventos aparentemente normales",
                    "development": "Descubrir capas de la conspiración gradualmente",
                    "climax": "Revelar una conexión sorprendente entre elementos aparentemente no relacionados",
                    "cliffhanger": "Introducir una nueva amenaza o nivel de la conspiración",
                    "segments": [
                        "Evento aparentemente normal",
                        "Detalle discordante o sospechoso",
                        "Investigación inicial",
                        "Primer nivel de verdad oculta",
                        "Amenazas o obstáculos",
                        "Aliados inesperados",
                        "Revelación de motivos",
                        "Conexión sorprendente",
                        "Nueva amenaza o misterio"
                    ]
                }
            },
            "educational_kids": {
                "discovery_journey": {
                    "hook": "Plantear una pregunta intrigante sobre el mundo",
                    "development": "Explorar el tema a través de ejemplos visuales y experimentos",
                    "climax": "Revelar la explicación científica de manera accesible",
                    "segments": [
                        "Pregunta curiosa",
                        "Conexión con experiencias cotidianas",
                        "Personaje guía introduce el tema",
                        "Exploración visual del concepto",
                        "Experimento o demostración simple",
                        "Explicación adaptada a niños",
                        "Aplicaciones en el mundo real",
                        "Resumen de aprendizaje",
                        "Llamada a la acción"
                    ]
                },
                "historical_adventure": {
                    "hook": "Transportar a los espectadores a un momento histórico fascinante",
                    "development": "Explorar la vida, desafíos y logros de la época",
                    "climax": "Conectar los eventos históricos con el presente",
                    "segments": [
                        "Máquina del tiempo imaginaria",
                        "Llegada a la época histórica",
                        "Encuentro con personaje histórico",
                        "Desafío o problema de la época",
                        "Exploración de costumbres y tecnología",
                        "Evento histórico clave",
                        "Lecciones aprendidas",
                        "Conexión con el presente",
                        "Llamada a la acción"
                    ]
                }
            },
            "dance_trends": {
                "tutorial_progression": {
                    "hook": "Mostrar la versión final impresionante del baile",
                    "development": "Desglosar los movimientos paso a paso",
                    "climax": "Mostrar la secuencia completa a velocidad normal",
                    "segments": [
                        "Demostración del resultado final",
                        "Origen o contexto del baile",
                        "Desglose del primer movimiento",
                        "Práctica del primer movimiento",
                        "Desglose del segundo movimiento",
                        "Combinación de movimientos",
                        "Consejos para perfeccionar",
                        "Secuencia completa",
                        "Llamada a la acción"
                    ]
                },
                "character_dance": {
                    "hook": "Introducir un personaje carismático con estilo único",
                    "development": "El personaje enseña su baile característico",
                    "climax": "Transformación completa en el personaje a través del baile",
                    "segments": [
                        "Presentación del personaje",
                        "Historia o personalidad del personaje",
                        "Elementos distintivos del estilo",
                        "Primeros pasos característicos",
                        "Elementos de vestuario o actitud",
                        "Movimientos avanzados",
                        "Combinación de elementos",
                        "Transformación completa",
                        "Llamada a la acción"
                    ]
                }
            }
        }
        
        # Guardar plotlines por defecto
        os.makedirs(os.path.dirname(self.plotlines_file), exist_ok=True)
        with open(self.plotlines_file, 'w', encoding='utf-8') as f:
            json.dump(default_plotlines, f, indent=4)
        
        return default_plotlines
    
    def _load_cta_templates(self) -> Dict:
        """Carga las plantillas de CTAs desde el archivo JSON"""
        try:
            if os.path.exists(self.cta_templates_file):
                with open(self.cta_templates_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de CTA templates no encontrado: {self.cta_templates_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar CTA templates: {str(e)}")
            return {}
    
    def generate_story_structure(self, niche: str, platform: str, duration_type: str = 'medium', 
                                plotline_type: str = None) -> Dict:
        """
        Genera una estructura narrativa para un contenido específico
        
        Args:
            niche: Nicho de contenido (finance, health, etc.)
            platform: Plataforma (youtube, tiktok, etc.)
            duration_type: Tipo de duración (short, medium, long)
            plotline_type: Tipo de arco narrativo (si es None, se selecciona automáticamente)
            
        Returns:
            Diccionario con la estructura narrativa
        """
        # Código existente...
        
    def generate_series_arc(self, series_type: str, num_episodes: int, 
                           episode_duration: str = 'medium_episode') -> Dict:
        """
        Genera un arco narrativo para una serie completa
        
        Args:
            series_type: Tipo de serie (educational, biblical_values, ai_learning, etc.)
            num_episodes: Número de episodios en la serie
            episode_duration: Duración de cada episodio (short_episode, medium_episode, long_episode)
            
        Returns:
            Diccionario con la estructura del arco narrativo de la serie
        """
        logger.info(f"Generando arco para serie de tipo {series_type} con {num_episodes} episodios")
        
        # Validar parámetros
        if series_type not in self.series_arc_types:
            series_type = 'educational'  # Tipo por defecto
            
        if episode_duration not in self.platform_durations['series']:
            episode_duration = 'medium_episode'  # Duración por defecto
            
        # Obtener configuración del tipo de serie
        arc_config = self.series_arc_types[series_type]
        
        # Determinar estructura de la serie según el tipo
        if arc_config['arc_length'] == 'flexible':
            # Series educativas: cada episodio puede ser independiente
            episode_structures = self._generate_educational_episodes(series_type, num_episodes)
        elif arc_config['arc_length'] == 'episodic':
            # Series de valores bíblicos: historias conectadas por tema
            episode_structures = self._generate_biblical_episodes(num_episodes)
        elif arc_config['arc_length'] == 'progressive':
            # Series de aprendizaje de IA: conceptos que se construyen uno sobre otro
            episode_structures = self._generate_progressive_episodes(series_type, num_episodes)
        elif arc_config['arc_length'] == 'season':
            # Series de misterio: un arco principal para toda la temporada
            episode_structures = self._generate_mystery_season(num_episodes)
        elif arc_config['arc_length'] == 'situational':
            # Series de comedia: situaciones independientes con personajes recurrentes
            episode_structures = self._generate_comedy_episodes(num_episodes)
        elif arc_config['arc_length'] == 'viral':
            # Series de baile: basadas en tendencias virales
            episode_structures = self._generate_dance_episodes(num_episodes)
        else:
            # Estructura genérica por defecto
            episode_structures = self._generate_generic_episodes(series_type, num_episodes)
        
        # Crear estructura de la serie completa
        series_structure = {
            'series_type': series_type,
            'num_episodes': num_episodes,
            'episode_duration': episode_duration,
            'continuity_level': arc_config['continuity_level'],
            'engagement_hooks': arc_config['engagement_hooks'],
            'episodes': episode_structures,
            'series_cta_strategy': self._generate_series_cta_strategy(series_type, num_episodes),
            'character_continuity': self._generate_character_continuity(series_type, num_episodes),
            'knowledge_references': self._generate_knowledge_references(series_type, num_episodes)
        }
        
        # Guardar en la base de conocimiento para mantener continuidad
        series_id = f"{series_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.kb.store_series_structure(series_id, series_structure)
        
        return series_structure
    
    def _generate_educational_episodes(self, series_type: str, num_episodes: int) -> List[Dict]:
        """Genera estructura para episodios educativos"""
        episodes = []
        
        # Seleccionar plotline adecuado
        if series_type == 'biblical_values':
            plotline_category = 'biblical_values'
            plotline_types = list(self.plotlines.get(plotline_category, {}).keys())
        elif series_type == 'ai_learning':
            plotline_category = 'ai_learning'
            plotline_types = list(self.plotlines.get(plotline_category, {}).keys())
        elif series_type == 'educational_kids':
            plotline_category = 'educational_kids'
            plotline_types = list(self.plotlines.get(plotline_category, {}).keys())
        else:
            # Usar categoría educativa genérica
            plotline_category = 'educational_kids'
            plotline_types = ['discovery_journey', 'historical_adventure']
        
        # Si no hay plotlines disponibles, usar estructura genérica
        if not plotline_types:
            return self._generate_generic_episodes(series_type, num_episodes)
        
        # Generar cada episodio
        for i in range(num_episodes):
            # Alternar entre diferentes tipos de plotlines para variedad
            plotline_type = plotline_types[i % len(plotline_types)]
            
            # Obtener estructura base del plotline
            plotline = self.plotlines.get(plotline_category, {}).get(plotline_type, {})
            
            # Crear estructura del episodio
            episode = {
                'episode_number': i + 1,
                'title': f"Episodio {i + 1}",  # Título genérico, se personalizaría después
                'plotline_type': plotline_type,
                'hook': plotline.get('hook', "Introducir el tema del episodio"),
                'development': plotline.get('development', "Explorar el tema en detalle"),
                'climax': plotline.get('climax', "Presentar la conclusión o aprendizaje clave"),
                'segments': plotline.get('segments', ["Introducción", "Desarrollo", "Conclusión"]),
                'cta_points': self._generate_episode_cta_points(i, num_episodes),
                                'knowledge_references': [],  # Se llenarían con referencias a episodios anteriores
                'continuity_elements': self._generate_continuity_elements(series_type, i, num_episodes)
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _generate_biblical_episodes(self, num_episodes: int) -> List[Dict]:
        """Genera estructura para episodios de valores bíblicos"""
        episodes = []
        
        # Lista de historias bíblicas populares para niños
        biblical_stories = [
            {"title": "La Creación", "theme": "Dios como creador", "key_character": "Adán y Eva"},
            {"title": "El Arca de Noé", "theme": "Obediencia y fe", "key_character": "Noé"},
            {"title": "La Torre de Babel", "theme": "Orgullo y humildad", "key_character": "Constructores"},
            {"title": "Abraham y la promesa", "theme": "Fe y confianza", "key_character": "Abraham"},
            {"title": "José y sus hermanos", "theme": "Perdón y reconciliación", "key_character": "José"},
            {"title": "Moisés y la zarza ardiente", "theme": "Llamado divino", "key_character": "Moisés"},
            {"title": "Las plagas de Egipto", "theme": "Poder de Dios", "key_character": "Moisés y Faraón"},
            {"title": "El cruce del Mar Rojo", "theme": "Liberación divina", "key_character": "Moisés"},
            {"title": "Los Diez Mandamientos", "theme": "Obediencia a Dios", "key_character": "Moisés"},
            {"title": "David y Goliat", "theme": "Valentía y fe", "key_character": "David"},
            {"title": "El rey Salomón", "theme": "Sabiduría", "key_character": "Salomón"},
            {"title": "Jonás y la ballena", "theme": "Arrepentimiento", "key_character": "Jonás"},
            {"title": "Daniel en el foso de los leones", "theme": "Confianza en Dios", "key_character": "Daniel"},
            {"title": "El nacimiento de Jesús", "theme": "Esperanza", "key_character": "Jesús, María y José"},
            {"title": "Jesús y los pescadores", "theme": "Seguir a Jesús", "key_character": "Jesús y discípulos"},
            {"title": "La parábola del buen samaritano", "theme": "Amor al prójimo", "key_character": "Samaritano"},
            {"title": "La parábola del hijo pródigo", "theme": "Perdón y amor", "key_character": "Padre e hijo"},
            {"title": "Jesús calma la tormenta", "theme": "Fe en momentos difíciles", "key_character": "Jesús"},
            {"title": "La multiplicación de los panes", "theme": "Provisión divina", "key_character": "Jesús"},
            {"title": "La resurrección de Jesús", "theme": "Victoria y esperanza", "key_character": "Jesús"}
        ]
        
        # Asegurar que tenemos suficientes historias
        while len(biblical_stories) < num_episodes:
            biblical_stories.extend(biblical_stories)
        
        # Seleccionar historias para la serie
        selected_stories = biblical_stories[:num_episodes]
        
        # Generar cada episodio
        for i, story in enumerate(selected_stories):
            # Seleccionar aleatoriamente un tipo de plotline para variedad
            plotline_types = list(self.plotlines.get("biblical_values", {}).keys())
            if not plotline_types:
                plotline_types = ["moral_lesson"]  # Valor por defecto
            
            plotline_type = random.choice(plotline_types)
            plotline = self.plotlines.get("biblical_values", {}).get(plotline_type, {})
            
            # Crear estructura del episodio
            episode = {
                'episode_number': i + 1,
                'title': story["title"],
                'theme': story["theme"],
                'key_character': story["key_character"],
                'plotline_type': plotline_type,
                'hook': f"Presentar la historia de {story['key_character']} y el tema de {story['theme']}",
                'development': plotline.get('development', "Explorar las decisiones y consecuencias"),
                'climax': plotline.get('climax', "Revelar la enseñanza moral o espiritual"),
                'segments': plotline.get('segments', ["Introducción", "Desarrollo", "Conclusión"]),
                'cta_points': self._generate_episode_cta_points(i, num_episodes),
                'knowledge_references': self._generate_biblical_references(story),
                'continuity_elements': self._generate_continuity_elements("biblical_values", i, num_episodes)
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _generate_progressive_episodes(self, series_type: str, num_episodes: int) -> List[Dict]:
        """Genera estructura para episodios progresivos (como aprendizaje de IA)"""
        episodes = []
        
        # Para series de IA, definimos una progresión lógica de conceptos
        if series_type == 'ai_learning':
            ai_concepts = [
                {"title": "¿Qué es la Inteligencia Artificial?", "concept": "definición_ia", "difficulty": "básico"},
                {"title": "Historia de la IA: De los inicios a hoy", "concept": "historia_ia", "difficulty": "básico"},
                {"title": "Aprendizaje Automático: El corazón de la IA", "concept": "machine_learning", "difficulty": "básico"},
                {"title": "Redes Neuronales: Imitando el cerebro", "concept": "redes_neuronales", "difficulty": "intermedio"},
                {"title": "Aprendizaje Profundo: Capas de conocimiento", "concept": "deep_learning", "difficulty": "intermedio"},
                {"title": "Procesamiento del Lenguaje Natural", "concept": "nlp", "difficulty": "intermedio"},
                {"title": "Visión por Computadora", "concept": "computer_vision", "difficulty": "intermedio"},
                {"title": "IA Generativa: Creando contenido", "concept": "ia_generativa", "difficulty": "avanzado"},
                {"title": "Ética en la IA: Desafíos y soluciones", "concept": "etica_ia", "difficulty": "avanzado"},
                {"title": "El futuro de la IA: Tendencias y predicciones", "concept": "futuro_ia", "difficulty": "avanzado"},
                {"title": "IA en la vida cotidiana", "concept": "ia_cotidiana", "difficulty": "básico"},
                {"title": "Cómo crear tu primer modelo de IA", "concept": "crear_modelo", "difficulty": "intermedio"},
                {"title": "IA y creatividad: Arte, música y más", "concept": "ia_creatividad", "difficulty": "intermedio"},
                {"title": "IA y salud: Revolucionando la medicina", "concept": "ia_salud", "difficulty": "avanzado"},
                {"title": "IA y educación: El futuro del aprendizaje", "concept": "ia_educacion", "difficulty": "intermedio"}
            ]
            
            # Asegurar que tenemos suficientes conceptos
            while len(ai_concepts) < num_episodes:
                ai_concepts.extend(ai_concepts)
            
            # Ordenar conceptos por dificultad para una progresión lógica
            concept_order = {"básico": 0, "intermedio": 1, "avanzado": 2}
            ai_concepts.sort(key=lambda x: concept_order[x["difficulty"]])
            
            # Seleccionar conceptos para la serie
            selected_concepts = ai_concepts[:num_episodes]
            
            # Generar cada episodio
            for i, concept in enumerate(selected_concepts):
                # Seleccionar tipo de plotline según el concepto
                if concept["difficulty"] == "básico":
                    plotline_type = "concept_introduction"
                elif concept["difficulty"] == "intermedio":
                    plotline_type = random.choice(["concept_introduction", "ai_history"])
                else:  # avanzado
                    plotline_type = random.choice(["concept_introduction", "ethical_dilemma"])
                
                plotline = self.plotlines.get("ai_learning", {}).get(plotline_type, {})
                
                # Crear estructura del episodio
                episode = {
                    'episode_number': i + 1,
                    'title': concept["title"],
                    'concept': concept["concept"],
                    'difficulty': concept["difficulty"],
                    'plotline_type': plotline_type,
                    'hook': plotline.get('hook', f"Introducir el concepto de {concept['concept']}"),
                    'development': plotline.get('development', "Explorar el concepto en detalle"),
                    'climax': plotline.get('climax', "Mostrar aplicaciones prácticas"),
                    'segments': plotline.get('segments', ["Introducción", "Desarrollo", "Conclusión"]),
                    'cta_points': self._generate_episode_cta_points(i, num_episodes),
                    'knowledge_references': self._generate_concept_references(concept, i, selected_concepts),
                    'continuity_elements': self._generate_continuity_elements("ai_learning", i, num_episodes)
                }
                
                episodes.append(episode)
        else:
            # Para otros tipos de series progresivas, usar estructura genérica
            episodes = self._generate_generic_episodes(series_type, num_episodes)
        
        return episodes
    
    def _generate_mystery_season(self, num_episodes: int) -> List[Dict]:
        """Genera estructura para una temporada de misterio"""
        episodes = []
        
        # Crear un misterio principal que se desarrollará a lo largo de la temporada
        mystery_themes = [
            "El secreto de la mansión abandonada",
            "La desaparición en el pueblo costero",
            "El misterio del artefacto antiguo",
            "La conspiración corporativa",
            "El caso de las identidades robadas",
            "El enigma de los sucesos paranormales"
        ]
        
        main_mystery = random.choice(mystery_themes)
        
        # Definir la estructura de revelación del misterio
        # Dividimos la temporada en fases: introducción, desarrollo, giros, revelaciones, clímax
        total_phases = 5
        episodes_per_phase = max(1, num_episodes // total_phases)
        
        # Generar cada episodio
        for i in range(num_episodes):
            # Determinar la fase actual
            current_phase = min(i // episodes_per_phase, total_phases - 1)
            
            # Configurar el episodio según la fase
            if current_phase == 0:  # Introducción
                plotline_type = "detective_case"
                hook = f"Presentar el misterio de {main_mystery}"
                development = "Establecer personajes y primeras pistas"
                climax = "Revelar la primera pista importante"
                cliffhanger = "Sugerir que hay más de lo que parece"
            elif current_phase == 1:  # Desarrollo
                plotline_type = random.choice(["detective_case", "conspiracy_unraveling"])
                hook = "Profundizar en las implicaciones del misterio"
                development = "Seguir nuevas pistas y teorías"
                climax = "Descubrir una conexión inesperada"
                cliffhanger = "Introducir un nuevo elemento sospechoso"
            elif current_phase == 2:  # Giros
                plotline_type = "conspiracy_unraveling"
                hook = "Cuestionar las suposiciones anteriores"
                development = "Explorar teorías alternativas"
                climax = "Revelar un giro importante en la trama"
                cliffhanger = "Poner en peligro a un personaje clave"
            elif current_phase == 3:  # Revelaciones
                plotline_type = random.choice(["detective_case", "conspiracy_unraveling"])
                hook = "Comenzar a unir las piezas del rompecabezas"
                development = "Confrontar a sospechosos clave"
                climax = "Revelar una verdad importante sobre el misterio"
                cliffhanger = "Sugerir que la resolución está cerca"
            else:  # Clímax (fase 4)
                plotline_type = "detective_case"
                hook = "Preparar la resolución final"
                development = "Confrontación decisiva"
                climax = "Resolver el misterio principal"
                # El último episodio puede tener un gancho para la siguiente temporada
                cliffhanger = "Insinuar un nuevo misterio" if i == num_episodes - 1 else "Resolver un aspecto pendiente"
            
            # Obtener estructura base del plotline
            plotline = self.plotlines.get("mystery_series", {}).get(plotline_type, {})
            
            # Crear estructura del episodio
            episode = {
                'episode_number': i + 1,
                'title': f"Episodio {i + 1}: {self._generate_mystery_title(main_mystery, current_phase, i)}",
                'phase': current_phase,
                'main_mystery': main_mystery,
                'plotline_type': plotline_type,
                'hook': hook,
                'development': development,
                'climax': climax,
                'cliffhanger': cliffhanger,
                'segments': plotline.get('segments', ["Introducción", "Desarrollo", "Conclusión"]),
                'cta_points': self._generate_episode_cta_points(i, num_episodes),
                'knowledge_references': self._generate_mystery_references(i, current_phase, main_mystery),
                'continuity_elements': self._generate_continuity_elements("mystery", i, num_episodes, high_continuity=True)
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _generate_comedy_episodes(self, num_episodes: int) -> List[Dict]:
        """Genera estructura para episodios de comedia"""
        episodes = []
        
        # Definir situaciones cómicas recurrentes
        comedy_situations = [
            "Malentendido entre personajes",
            "Situación embarazosa en público",
            "Plan que sale terriblemente mal",
            "Competencia o rivalidad cómica",
            "Personaje fuera de su elemento",
            "Secreto que se vuelve cada vez más difícil de mantener",
            "Día que se repite una y otra vez",
            "Intercambio de roles o identidades",
            "Reunión familiar caótica",
            "Tecnología que causa problemas hilarantes"
        ]
        
        # Definir personajes recurrentes para mantener continuidad
        recurring_characters = [
            {"name": "Alex", "trait": "Optimista ingenuo", "catchphrase": "¡Esto va a ser increíble!"},
            {"name": "Sam", "trait": "Sarcástico realista", "catchphrase": "¿Qué podría salir mal? Oh, espera..."},
            {"name": "Jordan", "trait": "Perfeccionista estresado", "catchphrase": "¡Necesito que todo sea perfecto!"},
            {"name": "Taylor", "trait": "Despistado creativo", "catchphrase": "Tengo una idea brillante..."},
            {"name": "Casey", "trait": "Competitivo impulsivo", "catchphrase": "¡Acepto el desafío!"}
        ]
        
        # Generar cada episodio
        for i in range(num_episodes):
            # Seleccionar situación cómica y personajes para este episodio
            situation = random.choice(comedy_situations)
            main_character = random.choice(recurring_characters)
            supporting_characters = random.sample([c for c in recurring_characters if c != main_character], 
                                                 k=min(3, len(recurring_characters)-1))
            
            # Crear estructura del episodio
            episode = {
                'episode_number': i + 1,
                'title': f"Episodio {i + 1}: {situation}",
                'situation': situation,
                'main_character': main_character,
                'supporting_characters': supporting_characters,
                'plotline_type': "setup_punchline",
                'hook': f"Introducir a {main_character['name']} en una situación de {situation}",
                'development': f"Complicar la situación con la participación de {', '.join([c['name'] for c in supporting_characters])}",
                'climax': "Resolución cómica de la situación",
                'segments': [
                    "Introducción de la situación",
                    "Presentación del problema",
                    "Primer intento de solución",
                    "Complicación inesperada",
                    "Segundo intento más desesperado",
                    "Crisis cómica",
                    "Resolución inesperada",
                    "Consecuencias hilarantes",
                    "Gancho para el siguiente episodio"
                ],
                'cta_points': self._generate_episode_cta_points(i, num_episodes),
                'knowledge_references': self._generate_comedy_references(main_character, supporting_characters, i),
                'continuity_elements': self._generate_continuity_elements("comedy", i, num_episodes, 
                                                                         recurring_elements={"characters": recurring_characters})
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _generate_dance_episodes(self, num_episodes: int) -> List[Dict]:
        """Genera estructura para episodios de baile basados en tendencias"""
        episodes = []
        
        # Definir tipos de bailes y tendencias
        dance_trends = [
            {"name": "Shuffle Dance", "difficulty": "intermedio", "music_style": "EDM"},
            {"name": "Hip Hop Básico", "difficulty": "principiante", "music_style": "Hip Hop"},
            {"name": "Breakdance Intro", "difficulty": "intermedio", "music_style": "Breakbeats"},
            {"name": "K-pop Dance", "difficulty": "intermedio", "music_style": "K-pop"},
            {"name": "Salsa Básica", "difficulty": "principiante", "music_style": "Salsa"},
            {"name": "House Dance", "difficulty": "avanzado", "music_style": "House"},
            {"name": "Popping", "difficulty": "intermedio", "music_style": "Funk"},
            {"name": "Waacking", "difficulty": "intermedio", "music_style": "Disco"},
            {"name": "Dancehall", "difficulty": "intermedio", "music_style": "Dancehall"},
            {"name": "Tutting", "difficulty": "avanzado", "music_style": "Electrónica"},
            {"name": "Voguing", "difficulty": "avanzado", "music_style": "House"},
            {"name": "Locking", "difficulty": "intermedio", "music_style": "Funk"},
            {"name": "Baile Urbano", "difficulty": "principiante", "music_style": "Urbano"},
            {"name": "Baile TikTok Viral", "difficulty": "principiante", "music_style": "Pop"}
        ]
        
        # Asegurar que tenemos suficientes tendencias
        while len(dance_trends) < num_episodes:
            dance_trends.extend(dance_trends)
        
        # Ordenar por dificultad para una progresión lógica
        difficulty_order = {"principiante": 0, "intermedio": 1, "avanzado": 2}
        dance_trends.sort(key=lambda x: difficulty_order[x["difficulty"]])
        
        # Seleccionar tendencias para la serie
        selected_trends = dance_trends[:num_episodes]
        
        # Generar cada episodio
        for i, trend in enumerate(selected_trends):
            # Alternar entre tipos de plotlines para variedad
            plotline_types = list(self.plotlines.get("dance_trends", {}).keys())
            if not plotline_types:
                plotline_types = ["tutorial_progression", "character_dance"]
            
            plotline_type = plotline_types[i % len(plotline_types)]
            plotline = self.plotlines.get("dance_trends", {}).get(plotline_type, {})
            
            # Crear estructura del episodio
            episode = {
                'episode_number': i + 1,
                'title': f"Aprende {trend['name']} - Nivel {trend['difficulty'].capitalize()}",
                'dance_style': trend['name'],
                'difficulty': trend['difficulty'],
                'music_style': trend['music_style'],
                'plotline_type': plotline_type,
                'hook': plotline.get('hook', f"Mostrar la versión final del baile {trend['name']}"),
                'development': plotline.get('development', "Desglosar los movimientos paso a paso"),
                'climax': plotline.get('climax', "Mostrar la secuencia completa a velocidad normal"),
                'segments': plotline.get('segments', ["Introducción", "Desarrollo", "Conclusión"]),
                'cta_points': self._generate_episode_cta_points(i, num_episodes, 
                                                              cta_focus="community_participation"),
                'knowledge_references': self._generate_dance_references(trend, i),
                'continuity_elements': self._generate_continuity_elements("dance_trends", i, num_episodes)
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _generate_generic_episodes(self, series_type: str, num_episodes: int) -> List[Dict]:
        """Genera estructura genérica para episodios cuando no hay un tipo específico"""
        episodes = []
        
        for i in range(num_episodes):
            episode = {
                'episode_number': i + 1,
                'title': f"Episodio {i + 1}",
                'plotline_type': "generic",
                'hook': "Introducir el tema del episodio",
                'development': "Explorar el tema en detalle",
                'climax': "Presentar la conclusión o aprendizaje clave",
                'segments': ["Introducción", "Desarrollo", "Conclusión"],
                'cta_points': self._generate_episode_cta_points(i, num_episodes),
                'knowledge_references': [],
                'continuity_elements': self._generate_continuity_elements(series_type, i, num_episodes)
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _generate_episode_cta_points(self, episode_index: int, total_episodes: int, 
                                    cta_focus: str = None) -> List[Dict]:
        """Genera puntos de CTA estratégicos para un episodio"""
        cta_points = []
        
        # Determinar el tipo de CTA según la posición en la serie
        if episode_index == 0:  # Primer episodio
            cta_types = ["subscribe", "follow_series", "engagement"]
        elif episode_index == total_episodes - 1:  # Último episodio
            cta_types = ["full_series", "related_content", "community"]
        else:  # Episodios intermedios
            cta_types = ["next_episode", "engagement", "subscribe"]
        
        # Si hay un enfoque específico, priorizarlo
        if cta_focus:
            cta_types.insert(0, cta_focus)
        
        # Generar 2-3 puntos de CTA para el episodio
        num_ctas = random.randint(2, 3)
        for i in range(min(num_ctas, len(cta_types))):
            cta_type = cta_types[i]
            
            # Posición del CTA según el tipo
            if cta_type in ["subscribe", "follow_series"]:
                position = "early"
            elif cta_type in ["engagement", "next_episode"]:
                position = "middle"
            else:
                position = "end"
            
            cta_point = {
                'type': cta_type,
                'position': position,
                'message': self._generate_cta_message(cta_type, episode_index, total_episodes)
            }
            
            cta_points.append(cta_point)
        
        return cta_points
    
    def _generate_cta_message(self, cta_type: str, episode_index: int, total_episodes: int) -> str:
        """Genera un mensaje de CTA según el tipo y contexto"""
        if cta_type == "subscribe":
            return "Suscríbete para no perderte ningún episodio de esta serie"
        elif cta_type == "follow_series":
            return f"Esta es una serie de {total_episodes} episodios. ¡Síguenos para ver toda la historia!"
        elif cta_type == "next_episode":
            return "El próximo episodio estará disponible pronto. ¡Activa las notificaciones!"
        elif cta_type == "engagement":
            return "¿Qué te ha parecido este episodio? ¡Déjanos tu opinión en los comentarios!"
        elif cta_type == "full_series":
            return "¿Te has perdido algún episodio? ¡Mira la serie completa en nuestra playlist!"
        elif cta_type == "related_content":
            return "Si te ha gustado esta serie, te encantará nuestro contenido sobre temas similares"
        elif cta_type == "community":
            return "Únete a nuestra comunidad para compartir tus experiencias y aprender juntos"
        elif cta_type == "community_participation":
            return "¡Muéstranos tu versión de este baile usando nuestro hashtag!"
        else:
            return "¡No olvides dar like y compartir este video!"
    
    def _generate_continuity_elements(self, series_type: str, episode_index: int, 
                                     total_episodes: int, high_continuity: bool = False,
                                     recurring_elements: Dict = None) -> Dict:
        """Genera elementos de continuidad entre episodios"""
        continuity = {
            'recurring_themes': [],
            'callbacks': [],
            'foreshadowing': []
        }
        
        # Elementos recurrentes según el tipo de serie
        if series_type == "biblical_values":
            continuity['recurring_themes'] = ["Fe", "Obediencia", "Amor", "Perdón"]
        elif series_type == "ai_learning":
            continuity['recurring_themes'] = ["Innovación", "Ética", "Futuro", "Potencial humano"]
        elif series_type == "mystery":
            continuity['recurring_themes'] = ["Verdad oculta", "Confianza", "Secretos", "Justicia"]
        elif series_type == "comedy":
            continuity['recurring_themes'] = ["Amistad", "Superación", "Caos cotidiano"]
        elif series_type == "dance_trends":
            continuity['recurring_themes'] = ["Expresión personal", "Comunidad", "Progreso"]
        
        # Callbacks a episodios anteriores (excepto en el primer episodio)
        if episode_index > 0:
            num_callbacks = 2 if high_continuity else 1
            for _ in range(num_callbacks):
                # Referencia a un episodio anterior aleatorio
                prev_episode = random.randint(1, episode_index)
                continuity['callbacks'].append({
                    'episode_reference': prev_episode,
                    'type': 'visual' if random.random() < 0.5 else 'narrativo',
                    'description': f"Referencia al episodio {prev_episode}"
                })
        
        # Foreshadowing para episodios futuros (excepto en el último episodio)
        if episode_index < total_episodes - 1:
            # En series de alta continuidad, incluir más elementos de anticipación
            num_foreshadowing = 2 if high_continuity else 1
            for _ in range(num_foreshadowing):
                # Anticipación de un episodio futuro aleatorio
                future_episode = random.randint(episode_index + 1, total_episodes)
                continuity['foreshadowing'].append({
                    'episode_reference': future_episode,
                    'type': 'sutil' if random.random() < 0.7 else 'directo',
                    'description': f"Anticipación de evento en episodio {future_episode}"
                })
        
        # Incluir elementos recurrentes específicos si se proporcionan
        if recurring_elements:
            continuity['specific_recurring_elements'] = recurring_elements
        
        return continuity
    
    def _generate_biblical_references(self, story: Dict) -> List[Dict]:
        """Genera referencias bíblicas para un episodio"""
        references = []
        
        # Referencia al pasaje bíblico principal
        references.append({
            'type': 'scripture',
            'description': f"Pasaje bíblico relacionado con {story['title']}",
            'content': f"Referencia a la historia de {story['key_character']}"
        })
        
        # Valores o enseñanzas relacionadas
        references.append({
            'type': 'value',
            'description': f"Enseñanza sobre {story['theme']}",
            'content': f"Aplicación del valor de {story['theme']} en la vida diaria"
        })
        
        return references
    
    def _generate_concept_references(self, concept: Dict, index: int, all_concepts: List[Dict]) -> List[Dict]:
        """Genera referencias a conceptos de IA para un episodio"""
        references = []
        
        # Referencia al concepto principal
        references.append({
            'type': 'concept',
            'description': f"Explicación de {concept['concept']}",
            'content': f"Definición y características de {concept['concept']}"
        })
        
        # Referencias a conceptos previos (si existen)
        if index > 0:
            prev_concepts = all_concepts[:index]
            for prev in random.sample(prev_concepts, min(2, len(prev_concepts))):
                references.append({
                    'type': 'previous_concept',
                    'description': f"Conexión con {prev['concept']}",
                    'content': f"Cómo {concept['concept']} se relaciona con {prev['concept']}"
                })
        
        return references
    
        def _generate_mystery_title(self, main_mystery: str, phase: int, episode_index: int) -> str:
        """Genera un título para un episodio de misterio basado en la fase y el misterio principal"""
        # Títulos según la fase de la temporada
        if phase == 0:  # Introducción
            prefixes = ["El comienzo de", "Primeras pistas sobre", "El descubrimiento de"]
        elif phase == 1:  # Desarrollo
            prefixes = ["Profundizando en", "Nuevas pistas sobre", "El enigma de"]
        elif phase == 2:  # Giros
            prefixes = ["La verdad detrás de", "El giro inesperado en", "Revelaciones sobre"]
        elif phase == 3:  # Revelaciones
            prefixes = ["Desenmascarando", "La verdad emerge sobre", "Confrontando"]
        else:  # Clímax
            prefixes = ["La resolución de", "El final de", "Desenlace:"]
        
        # Seleccionar un prefijo aleatorio
        prefix = random.choice(prefixes)
        
        # Extraer una parte clave del misterio principal para el título
        mystery_parts = main_mystery.split()
        if len(mystery_parts) > 3:
            key_part = " ".join(mystery_parts[1:3])
        else:
            key_part = main_mystery
        
        # Añadir un elemento específico del episodio
        episode_elements = [
            "la sombra", "el testigo", "la pista", "el sospechoso", 
            "la evidencia", "el secreto", "la conexión", "la trampa"
        ]
        
        # Combinar elementos para crear el título
        if episode_index % 3 == 0:
            return f"{prefix} {key_part}"
        else:
            element = random.choice(episode_elements)
            return f"{element.capitalize()} en {key_part}"
    
    def _generate_mystery_references(self, episode_index: int, phase: int, main_mystery: str) -> List[Dict]:
        """Genera referencias para mantener la continuidad en una serie de misterio"""
        references = []
        
        # Pistas acumuladas hasta este punto
        if episode_index > 0:
            references.append({
                'type': 'clues_recap',
                'description': f"Resumen de pistas clave hasta el episodio {episode_index}",
                'content': f"Recapitulación de elementos importantes para resolver {main_mystery}"
            })
        
        # Personajes relevantes según la fase
        if phase == 0:
            references.append({
                'type': 'character_introduction',
                'description': "Presentación de personajes clave",
                'content': "Introducción de protagonistas y posibles sospechosos"
            })
        elif phase in [1, 2]:
            references.append({
                'type': 'character_development',
                'description': "Desarrollo de motivaciones",
                'content': "Profundización en las motivaciones de personajes clave"
            })
        else:
            references.append({
                'type': 'character_revelation',
                'description': "Revelaciones sobre personajes",
                'content': "Verdades ocultas sobre los personajes principales"
            })
        
        # Elementos del misterio según la fase
        mystery_elements = {
            0: "primeras pistas",
            1: "conexiones entre pistas",
            2: "teorías alternativas",
            3: "revelaciones importantes",
            4: "resolución final"
        }
        
        references.append({
            'type': 'mystery_element',
            'description': f"Elementos de {mystery_elements[phase]}",
            'content': f"Información crucial para entender {main_mystery} en esta fase"
        })
        
        return references
    
    def _generate_comedy_references(self, main_character: Dict, supporting_characters: List[Dict], 
                                   episode_index: int) -> List[Dict]:
        """Genera referencias para mantener la continuidad en una serie de comedia"""
        references = []
        
        # Referencia al personaje principal
        references.append({
            'type': 'character_trait',
            'description': f"Rasgos característicos de {main_character['name']}",
            'content': f"{main_character['name']} es {main_character['trait']} y suele decir '{main_character['catchphrase']}'"
        })
        
        # Referencias a personajes secundarios
        for character in supporting_characters[:2]:  # Limitar a 2 para no sobrecargar
            references.append({
                'type': 'supporting_character',
                'description': f"Dinámica con {character['name']}",
                'content': f"Interacción entre {main_character['name']} y {character['name']}, quien es {character['trait']}"
            })
        
        # Referencias a episodios anteriores (si aplica)
        if episode_index > 0:
            references.append({
                'type': 'previous_situation',
                'description': f"Referencia a situación cómica anterior",
                'content': f"Mención a un evento divertido del episodio {random.randint(1, episode_index)}"
            })
        
        return references
    
    def _generate_dance_references(self, trend: Dict, episode_index: int) -> List[Dict]:
        """Genera referencias para mantener la continuidad en una serie de baile"""
        references = []
        
        # Referencia al estilo de baile
        references.append({
            'type': 'dance_style',
            'description': f"Características del {trend['name']}",
            'content': f"Explicación del origen y elementos distintivos del {trend['name']}"
        })
        
        # Referencia al estilo musical
        references.append({
            'type': 'music_style',
            'description': f"Música para {trend['name']}",
            'content': f"Explicación de por qué {trend['music_style']} funciona bien con este estilo de baile"
        })
        
        # Referencias a movimientos básicos (para principiantes)
        if trend['difficulty'] == 'principiante':
            references.append({
                'type': 'basic_movements',
                'description': "Movimientos fundamentales",
                'content': "Explicación detallada de la postura y pasos básicos"
            })
        
        # Referencias a técnicas más avanzadas
        elif trend['difficulty'] in ['intermedio', 'avanzado']:
            references.append({
                'type': 'advanced_technique',
                'description': "Técnicas especializadas",
                'content': f"Consejos para dominar aspectos técnicos del {trend['name']}"
            })
        
        # Referencias a bailes anteriores (si aplica)
        if episode_index > 0:
            references.append({
                'type': 'previous_dance',
                'description': "Conexión con bailes anteriores",
                'content': "Cómo aplicar técnicas aprendidas en episodios previos"
            })
        
        return references
    
    def _generate_series_cta_strategy(self, series_type: str, num_episodes: int) -> Dict:
        """Genera una estrategia de CTA para toda la serie"""
        strategy = {
            'primary_cta': '',
            'secondary_cta': '',
            'progression': [],
            'community_building': []
        }
        
        # Definir CTA primario según el tipo de serie
        if series_type == 'biblical_values':
            strategy['primary_cta'] = "Suscríbete para más historias bíblicas para niños"
            strategy['secondary_cta'] = "Comparte estos valores con otros padres y niños"
        elif series_type == 'ai_learning':
            strategy['primary_cta'] = "Suscríbete para aprender más sobre IA"
            strategy['secondary_cta'] = "Únete a nuestra comunidad de entusiastas de la tecnología"
        elif series_type == 'mystery':
            strategy['primary_cta'] = "Suscríbete para no perderte ninguna pista"
            strategy['secondary_cta'] = "Comparte tus teorías en los comentarios"
        elif series_type == 'comedy':
            strategy['primary_cta'] = "Suscríbete para más risas garantizadas"
            strategy['secondary_cta'] = "Comparte tus momentos favoritos"
        elif series_type == 'dance_trends':
            strategy['primary_cta'] = "Suscríbete para aprender más bailes"
            strategy['secondary_cta'] = "Comparte tu versión con nuestro hashtag"
        else:
            strategy['primary_cta'] = "Suscríbete para no perderte ningún episodio"
            strategy['secondary_cta'] = "Activa las notificaciones para nuevos contenidos"
        
        # Definir progresión de CTAs a lo largo de la serie
        progression_points = [
            {'position': 'inicio', 'focus': 'awareness', 'message': "Descubre nuestra serie completa"},
            {'position': '25%', 'focus': 'engagement', 'message': "Participa en nuestra comunidad"},
            {'position': '50%', 'focus': 'sharing', 'message': "Comparte este contenido con amigos"},
            {'position': '75%', 'focus': 'conversion', 'message': "Suscríbete para contenido exclusivo"},
            {'position': 'final', 'focus': 'retention', 'message': "Explora nuestras otras series"}
        ]
        
        # Seleccionar puntos de progresión según el número de episodios
        if num_episodes <= 3:
            strategy['progression'] = [progression_points[0], progression_points[-1]]
        elif num_episodes <= 6:
            strategy['progression'] = [progression_points[0], progression_points[2], progression_points[-1]]
        else:
            strategy['progression'] = progression_points
        
        # Estrategias de construcción de comunidad
        community_strategies = [
            {'type': 'question', 'description': "Preguntar a la audiencia sobre su experiencia"},
            {'type': 'challenge', 'description': "Proponer un reto relacionado con el contenido"},
            {'type': 'poll', 'description': "Realizar una encuesta sobre preferencias"},
            {'type': 'user_content', 'description': "Solicitar contenido generado por usuarios"},
            {'type': 'discussion', 'description': "Fomentar debate sobre un tema específico"}
        ]
        
        # Seleccionar estrategias de comunidad según el tipo de serie
        if series_type in ['biblical_values', 'educational_kids']:
            strategy['community_building'] = [s for s in community_strategies if s['type'] in ['question', 'challenge']]
        elif series_type in ['ai_learning', 'mystery']:
            strategy['community_building'] = [s for s in community_strategies if s['type'] in ['discussion', 'poll']]
        elif series_type == 'dance_trends':
            strategy['community_building'] = [s for s in community_strategies if s['type'] in ['challenge', 'user_content']]
        else:
            # Seleccionar 3 estrategias aleatorias
            strategy['community_building'] = random.sample(community_strategies, min(3, len(community_strategies)))
        
        return strategy
    
    def _generate_character_continuity(self, series_type: str, num_episodes: int) -> Dict:
        """Genera estructura para mantener continuidad de personajes en la serie"""
        continuity = {
            'main_characters': [],
            'supporting_characters': [],
            'character_arcs': []
        }
        
        # Definir personajes según el tipo de serie
        if series_type == 'biblical_values':
            continuity['main_characters'] = [
                {'name': 'Narrador', 'role': 'Guía', 'traits': ['Sabio', 'Amigable', 'Inspirador']},
                {'name': 'Niño/a Curioso/a', 'role': 'Aprendiz', 'traits': ['Curioso', 'Entusiasta', 'Reflexivo']}
            ]
            continuity['supporting_characters'] = [
                {'name': 'Personajes Bíblicos', 'role': 'Ejemplos', 'appearances': 'Según la historia'}
            ]
        
        elif series_type == 'ai_learning':
            continuity['main_characters'] = [
                {'name': 'Profesor/a Tech', 'role': 'Experto/a', 'traits': ['Conocedor', 'Entusiasta', 'Accesible']},
                {'name': 'IA Asistente', 'role': 'Ayudante', 'traits': ['Servicial', 'Preciso', 'Amigable']}
            ]
            continuity['supporting_characters'] = [
                {'name': 'Estudiantes', 'role': 'Aprendices', 'appearances': 'Recurrentes'},
                {'name': 'Expertos Invitados', 'role': 'Especialistas', 'appearances': 'Ocasionales'}
            ]
        
        elif series_type == 'mystery':
            continuity['main_characters'] = [
                {'name': 'Detective', 'role': 'Protagonista', 'traits': ['Observador', 'Inteligente', 'Persistente']},
                {'name': 'Asistente', 'role': 'Apoyo', 'traits': ['Leal', 'Práctico', 'Cuestionador']}
            ]
            continuity['supporting_characters'] = [
                {'name': 'Sospechosos', 'role': 'Antagonistas potenciales', 'appearances': 'Recurrentes'},
                {'name': 'Testigos', 'role': 'Fuentes de información', 'appearances': 'Episódicos'},
                {'name': 'Autoridades', 'role': 'Obstáculos o ayudas', 'appearances': 'Ocasionales'}
            ]
        
        elif series_type == 'comedy':
            # Ya definidos en _generate_comedy_episodes
            continuity['main_characters'] = [
                {'name': 'Alex', 'role': 'Protagonista optimista', 'traits': ['Optimista', 'Ingenuo', 'Entusiasta']},
                {'name': 'Sam', 'role': 'Contrapunto realista', 'traits': ['Sarcástico', 'Realista', 'Leal']}
            ]
            continuity['supporting_characters'] = [
                {'name': 'Jordan', 'role': 'Amigo perfeccionista', 'traits': ['Estresado', 'Organizado', 'Leal']},
                {'name': 'Taylor', 'role': 'Amigo creativo', 'traits': ['Despistado', 'Creativo', 'Soñador']},
                {'name': 'Casey', 'role': 'Amigo competitivo', 'traits': ['Impulsivo', 'Competitivo', 'Divertido']}
            ]
        
        elif series_type == 'dance_trends':
            continuity['main_characters'] = [
                {'name': 'Instructor/a', 'role': 'Maestro/a de baile', 'traits': ['Energético', 'Paciente', 'Talentoso']},
                {'name': 'Aprendiz', 'role': 'Estudiante', 'traits': ['Motivado', 'Progresivo', 'Relatable']}
            ]
            continuity['supporting_characters'] = [
                {'name': 'Bailarines de fondo', 'role': 'Demostración', 'appearances': 'Recurrentes'},
                {'name': 'Invitados especiales', 'role': 'Expertos en estilos', 'appearances': 'Ocasionales'}
            ]
        
        else:
            # Personajes genéricos para otros tipos de series
            continuity['main_characters'] = [
                {'name': 'Presentador/a', 'role': 'Guía', 'traits': ['Carismático', 'Conocedor', 'Accesible']}
            ]
            continuity['supporting_characters'] = [
                {'name': 'Invitados', 'role': 'Expertos', 'appearances': 'Episódicos'}
            ]
        
        # Generar arcos de personajes para series más largas
        if num_episodes >= 5:
            # Crear arcos para los personajes principales
            for character in continuity['main_characters']:
                character_arc = {
                    'character': character['name'],
                    'arc_type': random.choice(['growth', 'challenge', 'revelation']),
                    'progression': []
                }
                
                # Definir progresión del arco a lo largo de la serie
                arc_stages = min(num_episodes, 5)  # Máximo 5 etapas
                for i in range(arc_stages):
                    stage_position = (i + 1) / arc_stages
                    
                    if character_arc['arc_type'] == 'growth':
                        if stage_position < 0.3:
                            stage = "Introducción de la necesidad de crecimiento"
                        elif stage_position < 0.6:
                            stage = "Enfrentamiento a desafíos"
                        elif stage_position < 0.9:
                            stage = "Desarrollo de nuevas habilidades"
                        else:
                            stage = "Demostración de crecimiento completo"
                    
                    elif character_arc['arc_type'] == 'challenge':
                        if stage_position < 0.3:
                            stage = "Presentación del desafío principal"
                        elif stage_position < 0.6:
                            stage = "Intentos fallidos de superación"
                        elif stage_position < 0.9:
                            stage = "Momento de verdad y decisión"
                        else:
                            stage = "Superación del desafío"
                    
                    else:  # revelation
                        if stage_position < 0.3:
                            stage = "Pistas sutiles sobre una verdad oculta"
                        elif stage_position < 0.6:
                            stage = "Cuestionamiento de creencias previas"
                        elif stage_position < 0.9:
                            stage = "Descubrimiento de la verdad"
                        else:
                            stage = "Adaptación a la nueva realidad"
                    
                    character_arc['progression'].append({
                        'stage': i + 1,
                        'description': stage,
                        'episode_range': self._calculate_episode_range(i, arc_stages, num_episodes)
                    })
                
                continuity['character_arcs'].append(character_arc)
        
        return continuity
    
    def _calculate_episode_range(self, stage_index: int, total_stages: int, num_episodes: int) -> Dict:
        """Calcula el rango de episodios para una etapa de un arco de personaje"""
        # Distribuir las etapas a lo largo de todos los episodios
        episodes_per_stage = max(1, num_episodes // total_stages)
        
        start_episode = (stage_index * episodes_per_stage) + 1
        end_episode = min(start_episode + episodes_per_stage - 1, num_episodes)
        
        return {'start': start_episode, 'end': end_episode}
    
    def _generate_knowledge_references(self, series_type: str, num_episodes: int) -> Dict:
        """Genera referencias de conocimiento para toda la serie"""
        knowledge = {
            'core_concepts': [],
            'external_references': [],
            'cross_episode_connections': []
        }
        
        # Definir conceptos centrales según el tipo de serie
        if series_type == 'biblical_values':
            knowledge['core_concepts'] = [
                {'name': 'Fe', 'description': 'Confianza en Dios y sus promesas'},
                {'name': 'Amor', 'description': 'Amor a Dios y al prójimo'},
                {'name': 'Obediencia', 'description': 'Seguir los mandamientos de Dios'},
                {'name': 'Perdón', 'description': 'Perdonar a otros como Dios nos perdona'},
                {'name': 'Gratitud', 'description': 'Ser agradecidos por las bendiciones'}
            ]
            knowledge['external_references'] = [
                {'type': 'scripture', 'source': 'Biblia', 'relevance': 'Fuente principal de historias y valores'}
            ]
        
        elif series_type == 'ai_learning':
            knowledge['core_concepts'] = [
                {'name': 'Inteligencia Artificial', 'description': 'Sistemas que pueden realizar tareas que requieren inteligencia humana'},
                {'name': 'Machine Learning', 'description': 'Sistemas que aprenden de datos sin ser programados explícitamente'},
                {'name': 'Redes Neuronales', 'description': 'Modelos inspirados en el cerebro humano'},
                {'name': 'Ética en IA', 'description': 'Consideraciones éticas en el desarrollo y uso de IA'},
                {'name': 'Aplicaciones de IA', 'description': 'Usos prácticos de la IA en diferentes campos'}
            ]
            knowledge['external_references'] = [
                {'type': 'research', 'source': 'Papers académicos', 'relevance': 'Fundamentos teóricos'},
                {'type': 'industry', 'source': 'Desarrollos recientes', 'relevance': 'Aplicaciones prácticas'}
            ]
        
        elif series_type == 'mystery':
            knowledge['core_concepts'] = [
                {'name': 'Investigación', 'description': 'Métodos para recopilar y analizar pistas'},
                {'name': 'Deducción', 'description': 'Proceso de razonamiento para llegar a conclusiones'},
                {'name': 'Motivo', 'description': 'Razones que impulsan las acciones de los personajes'},
                {'name': 'Evidencia', 'description': 'Pruebas que apoyan o refutan teorías'},
                {'name': 'Resolución', 'description': 'Proceso de resolver el misterio central'}
            ]
            knowledge['external_references'] = [
                {'type': 'genre', 'source': 'Convenciones del género', 'relevance': 'Estructura narrativa'},
                {'type': 'techniques', 'source': 'Técnicas de investigación', 'relevance': 'Verosimilitud'}
            ]
        
        elif series_type == 'dance_trends':
            knowledge['core_concepts'] = [
                {'name': 'Técnica', 'description': 'Fundamentos técnicos de los movimientos'},
                {'name': 'Ritmo', 'description': 'Sincronización con la música'},
                {'name': 'Expresión', 'description': 'Comunicación de emociones a través del baile'},
                {'name': 'Estilo', 'description': 'Características distintivas de cada tipo de baile'},
                {'name': 'Progresión', 'description': 'Desarrollo de habilidades de básico a avanzado'}
            ]
            knowledge['external_references'] = [
                {'type': 'culture', 'source': 'Orígenes culturales', 'relevance': 'Contexto histórico y cultural'},
                {'type': 'trends', 'source': 'Tendencias actuales', 'relevance': 'Relevancia contemporánea'}
            ]
        
        # Generar conexiones entre episodios
        if num_episodes > 1:
            # Número de conexiones a generar
            num_connections = min(num_episodes - 1, 5)
            
            for i in range(num_connections):
                # Seleccionar dos episodios aleatorios diferentes
                episode1 = random.randint(1, num_episodes)
                episode2 = random.randint(1, num_episodes)
                while episode2 == episode1:
                    episode2 = random.randint(1, num_episodes)
                
                # Asegurar que el episodio con número menor va primero
                if episode2 < episode1:
                    episode1, episode2 = episode2, episode1
                
                # Generar una conexión temática
                connection = {
                    'from_episode': episode1,
                    'to_episode': episode2,
                    'type': random.choice(['theme', 'character', 'concept', 'event']),
                    'description': f"Conexión entre episodios {episode1} y {episode2}"
                }
                
                knowledge['cross_episode_connections'].append(connection)
        
        return knowledge
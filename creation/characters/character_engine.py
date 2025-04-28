"""
Character Engine - Gestor de personajes para contenido multimedia

Este módulo gestiona la creación y administración de personajes:
- Creación de personajes con personalidades definidas
- Gestión de identidades visuales y voces
- Adaptación de personajes según feedback y sentimiento
- Integración de CTAs naturales en el estilo del personaje
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
from creation.characters.personality_model import PersonalityModel
from creation.characters.visual_identity import VisualIdentity

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'character_engine.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('CharacterEngine')

class CharacterEngine:
    """
    Clase principal para la gestión de personajes.
    Implementa el patrón Singleton para asegurar una única instancia.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CharacterEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.knowledge_base = KnowledgeBase()
        self.personality_model = PersonalityModel()
        self.visual_identity = VisualIdentity()
        self.characters = {}
        self.load_characters()
        
        self._initialized = True
        logger.info("CharacterEngine initialized")

    def load_characters(self) -> None:
        """Carga los personajes existentes desde la base de conocimiento"""
        try:
            self.characters = self.knowledge_base.get_characters()
            logger.info(f"Loaded {len(self.characters)} characters from knowledge base")
        except Exception as e:
            logger.error(f"Error loading characters: {str(e)}")
            self.characters = {}

    def create_character(self, name: str, niche: str, traits: List[str], 
                         voice_style: str, visual_style: Dict[str, Any],
                         backstory: str = "", age: Optional[int] = None,
                         gender: Optional[str] = None) -> Dict[str, Any]:
        """
        Crea un nuevo personaje con los atributos especificados
        
        Args:
            name: Nombre del personaje
            niche: Nicho principal (finanzas, salud, gaming, etc.)
            traits: Lista de rasgos de personalidad
            voice_style: Estilo de voz (energético, calmado, profesional, etc.)
            visual_style: Diccionario con estilos visuales
            backstory: Historia de fondo del personaje
            age: Edad del personaje (opcional)
            gender: Género del personaje (opcional)
            
        Returns:
            Diccionario con la información del personaje creado
        """
        try:
            # Generar personalidad basada en los rasgos
            personality = self.personality_model.generate_personality(traits, niche)
            
            # Generar identidad visual
            visual_identity = self.visual_identity.generate_identity(
                name, niche, visual_style, personality
            )
            
            # Crear el personaje
            character_id = f"{name.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            character = {
                "id": character_id,
                "name": name,
                "niche": niche,
                "traits": traits,
                "personality": personality,
                "voice_style": voice_style,
                "visual_identity": visual_identity,
                "backstory": backstory,
                "age": age,
                "gender": gender,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "performance_metrics": {
                    "engagement_rate": 0,
                    "retention_rate": 0,
                    "conversion_rate": 0,
                    "sentiment_score": 0,
                    "videos_count": 0
                },
                "cta_style": self.generate_cta_style(personality, niche)
            }
            
            # Guardar en la base de conocimiento
            self.knowledge_base.save_character(character)
            
            # Actualizar caché local
            self.characters[character_id] = character
            
            logger.info(f"Created new character: {name} for niche: {niche}")
            return character
            
        except Exception as e:
            logger.error(f"Error creating character: {str(e)}")
            raise

    def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un personaje por su ID"""
        if character_id in self.characters:
            return self.characters[character_id]
        
        # Intentar cargar desde la base de conocimiento
        character = self.knowledge_base.get_character(character_id)
        if character:
            self.characters[character_id] = character
            return character
            
        logger.warning(f"Character not found: {character_id}")
        return None

    def update_character(self, character_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Actualiza los atributos de un personaje existente"""
        character = self.get_character(character_id)
        if not character:
            logger.warning(f"Cannot update non-existent character: {character_id}")
            return None
            
        # Actualizar atributos
        for key, value in updates.items():
            if key in character and key not in ["id", "created_at"]:
                character[key] = value
                
        character["updated_at"] = datetime.datetime.now().isoformat()
        
        # Si se actualizaron rasgos, regenerar personalidad
        if "traits" in updates:
            character["personality"] = self.personality_model.generate_personality(
                character["traits"], character["niche"]
            )
            character["cta_style"] = self.generate_cta_style(
                character["personality"], character["niche"]
            )
            
        # Si se actualizó el estilo visual, regenerar identidad visual
        if "visual_style" in updates:
            character["visual_identity"] = self.visual_identity.generate_identity(
                character["name"], character["niche"], 
                updates["visual_style"], character["personality"]
            )
            
        # Guardar en la base de conocimiento
        self.knowledge_base.save_character(character)
        
        # Actualizar caché local
        self.characters[character_id] = character
        
        logger.info(f"Updated character: {character_id}")
        return character

    def delete_character(self, character_id: str) -> bool:
        """Elimina un personaje"""
        if character_id not in self.characters:
            character = self.knowledge_base.get_character(character_id)
            if not character:
                logger.warning(f"Cannot delete non-existent character: {character_id}")
                return False
                
        # Eliminar de la base de conocimiento
        result = self.knowledge_base.delete_character(character_id)
        
        # Eliminar de la caché local
        if character_id in self.characters:
            del self.characters[character_id]
            
        logger.info(f"Deleted character: {character_id}")
        return result

    def get_characters_by_niche(self, niche: str) -> List[Dict[str, Any]]:
        """Obtiene todos los personajes de un nicho específico"""
        return [
            character for character in self.characters.values()
            if character["niche"] == niche
        ]

    def get_best_performing_character(self, niche: str, metric: str = "engagement_rate") -> Optional[Dict[str, Any]]:
        """Obtiene el personaje con mejor rendimiento en un nicho según una métrica"""
        characters = self.get_characters_by_niche(niche)
        if not characters:
            logger.warning(f"No characters found for niche: {niche}")
            return None
            
        return max(characters, key=lambda x: x["performance_metrics"].get(metric, 0))

    def update_performance_metrics(self, character_id: str, metrics: Dict[str, float]) -> bool:
        """Actualiza las métricas de rendimiento de un personaje"""
        character = self.get_character(character_id)
        if not character:
            logger.warning(f"Cannot update metrics for non-existent character: {character_id}")
            return False
            
        # Actualizar métricas
        for key, value in metrics.items():
            if key in character["performance_metrics"]:
                character["performance_metrics"][key] = value
                
        character["updated_at"] = datetime.datetime.now().isoformat()
        
        # Guardar en la base de conocimiento
        self.knowledge_base.save_character(character)
        
        # Actualizar caché local
        self.characters[character_id] = character
        
        logger.info(f"Updated performance metrics for character: {character_id}")
        return True

    def adjust_character_by_sentiment(self, character_id: str, sentiment_score: float) -> Optional[Dict[str, Any]]:
        """
        Ajusta la personalidad del personaje según el sentimiento de la audiencia
        
        Args:
            character_id: ID del personaje
            sentiment_score: Puntuación de sentimiento (-1 a 1)
            
        Returns:
            Personaje actualizado o None si no existe
        """
        character = self.get_character(character_id)
        if not character:
            logger.warning(f"Cannot adjust non-existent character: {character_id}")
            return None
            
        # Actualizar puntuación de sentimiento
        character["performance_metrics"]["sentiment_score"] = sentiment_score
        
        # Ajustar personalidad según sentimiento
        adjusted_personality = self.personality_model.adjust_by_sentiment(
            character["personality"], sentiment_score
        )
        
        # Actualizar personalidad
        character["personality"] = adjusted_personality
        character["updated_at"] = datetime.datetime.now().isoformat()
        
        # Actualizar estilo de CTA
        character["cta_style"] = self.generate_cta_style(adjusted_personality, character["niche"])
        
        # Guardar en la base de conocimiento
        self.knowledge_base.save_character(character)
        
        # Actualizar caché local
        self.characters[character_id] = character
        
        logger.info(f"Adjusted character by sentiment: {character_id}, score: {sentiment_score}")
        return character

    def generate_cta_style(self, personality: Dict[str, Any], niche: str) -> Dict[str, Any]:
        """
        Genera un estilo de CTA basado en la personalidad y nicho del personaje
        
        Args:
            personality: Diccionario con atributos de personalidad
            niche: Nicho del personaje
            
        Returns:
            Diccionario con el estilo de CTA
        """
        # Extraer atributos relevantes de la personalidad
        formality = personality.get("formality", 0.5)  # 0=informal, 1=formal
        enthusiasm = personality.get("enthusiasm", 0.5)  # 0=calmado, 1=entusiasta
        directness = personality.get("directness", 0.5)  # 0=indirecto, 1=directo
        humor = personality.get("humor", 0.3)  # 0=serio, 1=humorístico
        
        # Generar estilo de CTA
        cta_style = {
            "tone": "formal" if formality > 0.7 else "casual" if formality < 0.3 else "balanced",
            "intensity": "high" if enthusiasm > 0.7 else "low" if enthusiasm < 0.3 else "medium",
            "approach": "direct" if directness > 0.7 else "indirect" if directness < 0.3 else "balanced",
            "use_humor": humor > 0.5,
            "use_emojis": formality < 0.6 and enthusiasm > 0.4,
            "preferred_timing": "early" if directness > 0.7 else "late" if directness < 0.3 else "middle",
            "call_to_actions": self._generate_personalized_ctas(personality, niche)
        }
        
        return cta_style
        
    def _generate_personalized_ctas(self, personality: Dict[str, Any], niche: str) -> Dict[str, List[str]]:
        """Genera CTAs personalizados según la personalidad y nicho"""
        # Cargar plantillas de CTA
        cta_templates = self.knowledge_base.get_cta_templates()
        if not cta_templates:
            logger.warning("No CTA templates found, using defaults")
            return {
                "follow": ["¡Sígueme para más contenido!"],
                "like": ["¡Dale like si te ha gustado!"],
                "comment": ["¡Déjame tu opinión en los comentarios!"],
                "share": ["¡Comparte con tus amigos!"]
            }
            
        # Seleccionar plantillas según nicho
        niche_templates = cta_templates.get(niche, cta_templates.get("general", {}))
        
        # Personalizar según personalidad
        personalized_ctas = {}
        
        # Extraer atributos relevantes
        formality = personality.get("formality", 0.5)
        enthusiasm = personality.get("enthusiasm", 0.5)
        humor = personality.get("humor", 0.3)
        
        # Personalizar cada tipo de CTA
        for cta_type, templates in niche_templates.items():
            if not templates:
                continue
                
            # Seleccionar plantillas según personalidad
            if formality > 0.7:  # Formal
                filtered_templates = [t for t in templates if "!" not in t and "?" not in t]
            elif formality < 0.3:  # Informal
                filtered_templates = [t for t in templates if "!" in t or "?" in t]
            else:  # Balanceado
                filtered_templates = templates
                
            # Si no hay plantillas después del filtro, usar todas
            if not filtered_templates:
                filtered_templates = templates
                
            # Añadir emojis si corresponde
            if personality.get("use_emojis", False) and enthusiasm > 0.5:
                emojis = self._get_niche_emojis(niche)
                personalized_ctas[cta_type] = [
                    f"{template} {random.choice(emojis)}" for template in filtered_templates
                ]
            else:
                personalized_ctas[cta_type] = filtered_templates
                
        return personalized_ctas
        
    def _get_niche_emojis(self, niche: str) -> List[str]:
        """Retorna emojis relevantes para un nicho"""
        emoji_map = {
            "finance": ["💰", "💸", "📈", "🚀", "💎", "🤑"],
            "health": ["💪", "🥗", "🧘", "🏃", "❤️", "🧠"],
            "gaming": ["🎮", "🕹️", "🏆", "🔥", "⚔️", "🛡️"],
            "technology": ["💻", "📱", "🤖", "⚙️", "🔋", "🚀"],
            "humor": ["😂", "🤣", "😆", "😜", "🤪", "😎"],
            "education": ["📚", "🧠", "✏️", "🎓", "💡", "🔍"]
        }
        
        return emoji_map.get(niche, ["✨", "👍", "🙌", "🔥", "💯"])

    def get_character_for_content(self, niche: str, content_type: str, 
                                 sentiment_score: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Selecciona el personaje más adecuado para un tipo de contenido
        
        Args:
            niche: Nicho del contenido
            content_type: Tipo de contenido (tutorial, opinión, etc.)
            sentiment_score: Puntuación de sentimiento de la audiencia (opcional)
            
        Returns:
            Personaje seleccionado o None si no hay personajes disponibles
        """
        # Obtener personajes del nicho
        characters = self.get_characters_by_niche(niche)
        if not characters:
            logger.warning(f"No characters found for niche: {niche}")
            return None
            
        # Si hay puntuación de sentimiento, priorizar personajes con sentimiento similar
        if sentiment_score is not None:
            characters.sort(key=lambda x: abs(x["performance_metrics"].get("sentiment_score", 0) - sentiment_score))
            
        # Filtrar por tipo de contenido si es relevante
        content_type_map = {
            "tutorial": ["helpful", "clear", "methodical"],
            "opinion": ["opinionated", "passionate", "direct"],
            "news": ["objective", "serious", "informative"],
            "entertainment": ["funny", "energetic", "creative"],
            "educational": ["knowledgeable", "patient", "clear"]
        }
        
        relevant_traits = content_type_map.get(content_type, [])
        if relevant_traits:
            # Puntuar personajes según cuántos rasgos relevantes tienen
            scored_characters = []
            for character in characters:
                score = sum(1 for trait in character["traits"] if trait.lower() in relevant_traits)
                scored_characters.append((character, score))
                
            # Ordenar por puntuación
            scored_characters.sort(key=lambda x: x[1], reverse=True)
            
            # Retornar el mejor
            if scored_characters and scored_characters[0][1] > 0:
                return scored_characters[0][0]
        
        # Si no hay personajes con rasgos relevantes, usar el de mejor rendimiento
        return self.get_best_performing_character(niche)

    def generate_character_script(self, character_id: str, script_template: str, 
                                 variables: Dict[str, str]) -> str:
        """
        Adapta un guion al estilo del personaje
        
        Args:
            character_id: ID del personaje
            script_template: Plantilla del guion
            variables: Variables para reemplazar en la plantilla
            
        Returns:
            Guion adaptado al personaje
        """
        character = self.get_character(character_id)
        if not character:
            logger.warning(f"Cannot generate script for non-existent character: {character_id}")
            return script_template
            
        # Obtener personalidad
        personality = character["personality"]
        
        # Adaptar guion según personalidad
        adapted_script = self.personality_model.adapt_script(
            script_template, personality, variables
        )
        
        logger.info(f"Generated character script for: {character_id}")
        return adapted_script

    def get_character_cta(self, character_id: str, cta_type: str, 
                         platform: str = "tiktok") -> Optional[str]:
        """
        Obtiene un CTA personalizado para el personaje
        
        Args:
            character_id: ID del personaje
            cta_type: Tipo de CTA (follow, like, comment, etc.)
            platform: Plataforma (tiktok, youtube, etc.)
            
        Returns:
            CTA personalizado o None si no hay disponibles
        """
        character = self.get_character(character_id)
        if not character:
            logger.warning(f"Cannot get CTA for non-existent character: {character_id}")
            return None
            
        # Obtener estilo de CTA
        cta_style = character.get("cta_style", {})
        
        # Obtener CTAs personalizados
        personalized_ctas = cta_style.get("call_to_actions", {})
        
        # Si hay CTAs personalizados para este tipo, usar uno aleatorio
        if cta_type in personalized_ctas and personalized_ctas[cta_type]:
            return random.choice(personalized_ctas[cta_type])
            
        # Si no hay personalizados, usar plantillas generales
        cta_templates = self.knowledge_base.get_cta_templates()
        if not cta_templates:
            logger.warning("No CTA templates found")
            return None
            
        # Buscar en la plataforma específica
        platform_templates = cta_templates.get(platform, {})
        if cta_type in platform_templates and platform_templates[cta_type]:
            return random.choice(platform_templates[cta_type])
            
        # Buscar en plantillas generales
        general_templates = cta_templates.get("general", {})
        if cta_type in general_templates and general_templates[cta_type]:
            return random.choice(general_templates[cta_type])
            
        logger.warning(f"No CTA found for type: {cta_type}, platform: {platform}")
        return None
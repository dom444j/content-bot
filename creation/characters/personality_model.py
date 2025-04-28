"""
Personality Model - Modelado de personalidades para personajes

Este módulo gestiona la creación y adaptación de personalidades:
- Generación de personalidades basadas en rasgos
- Adaptación de guiones al estilo de personalidad
- Ajuste de personalidades según feedback y sentimiento
- Modelado de comportamientos y expresiones verbales
"""

import os
import sys
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
import re

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'personality_model.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PersonalityModel')

class PersonalityModel:
    """
    Clase para modelar y gestionar personalidades de personajes.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.trait_map = self._load_trait_map()
        logger.info("PersonalityModel initialized")

    def _load_trait_map(self) -> Dict[str, Dict[str, float]]:
        """Carga el mapa de rasgos de personalidad desde la base de conocimiento"""
        try:
            trait_map = self.knowledge_base.get_personality_traits()
            if not trait_map:
                # Crear mapa de rasgos por defecto
                trait_map = {
                    "extrovertido": {"formality": 0.3, "enthusiasm": 0.8, "directness": 0.7, "humor": 0.6},
                    "introvertido": {"formality": 0.6, "enthusiasm": 0.3, "directness": 0.4, "humor": 0.3},
                    "analítico": {"formality": 0.8, "enthusiasm": 0.4, "directness": 0.7, "humor": 0.2},
                    "creativo": {"formality": 0.4, "enthusiasm": 0.7, "directness": 0.5, "humor": 0.7},
                    "serio": {"formality": 0.9, "enthusiasm": 0.3, "directness": 0.8, "humor": 0.1},
                    "divertido": {"formality": 0.2, "enthusiasm": 0.9, "directness": 0.6, "humor": 0.9},
                    "profesional": {"formality": 0.8, "enthusiasm": 0.5, "directness": 0.7, "humor": 0.3},
                    "casual": {"formality": 0.2, "enthusiasm": 0.7, "directness": 0.5, "humor": 0.7},
                    "directo": {"formality": 0.5, "enthusiasm": 0.6, "directness": 0.9, "humor": 0.4},
                    "indirecto": {"formality": 0.6, "enthusiasm": 0.4, "directness": 0.2, "humor": 0.5},
                    "entusiasta": {"formality": 0.3, "enthusiasm": 0.9, "directness": 0.7, "humor": 0.6},
                    "calmado": {"formality": 0.6, "enthusiasm": 0.2, "directness": 0.5, "humor": 0.3},
                    "autoritario": {"formality": 0.7, "enthusiasm": 0.6, "directness": 0.9, "humor": 0.2},
                    "amigable": {"formality": 0.3, "enthusiasm": 0.7, "directness": 0.5, "humor": 0.7},
                    "motivador": {"formality": 0.4, "enthusiasm": 0.9, "directness": 0.8, "humor": 0.5},
                    "informativo": {"formality": 0.7, "enthusiasm": 0.5, "directness": 0.7, "humor": 0.3},
                    "humorístico": {"formality": 0.3, "enthusiasm": 0.8, "directness": 0.6, "humor": 0.9},
                    "sarcástico": {"formality": 0.4, "enthusiasm": 0.6, "directness": 0.7, "humor": 0.8},
                    "empático": {"formality": 0.5, "enthusiasm": 0.6, "directness": 0.4, "humor": 0.5},
                    "objetivo": {"formality": 0.7, "enthusiasm": 0.4, "directness": 0.8, "humor": 0.2}
                }
                # Guardar en la base de conocimiento
                self.knowledge_base.save_personality_traits(trait_map)
            return trait_map
        except Exception as e:
            logger.error(f"Error loading trait map: {str(e)}")
            # Retornar mapa vacío
            return {}

    def generate_personality(self, traits: List[str], niche: str) -> Dict[str, Any]:
        """
        Genera una personalidad basada en rasgos y nicho
        
        Args:
            traits: Lista de rasgos de personalidad
            niche: Nicho del personaje
            
        Returns:
            Diccionario con atributos de personalidad
        """
        # Inicializar atributos base
        personality = {
            "formality": 0.5,  # 0=informal, 1=formal
            "enthusiasm": 0.5,  # 0=calmado, 1=entusiasta
            "directness": 0.5,  # 0=indirecto, 1=directo
            "humor": 0.3,  # 0=serio, 1=humorístico
            "vocabulary_level": 0.5,  # 0=simple, 1=complejo
            "sentence_length": 0.5,  # 0=cortas, 1=largas
            "use_emojis": False,
            "use_slang": False,
            "use_technical_terms": False,
            "traits": traits,
            "niche": niche
        }
        
        # Ajustar según rasgos
        trait_count = 0
        for trait in traits:
            trait_lower = trait.lower()
            if trait_lower in self.trait_map:
                trait_count += 1
                for attr, value in self.trait_map[trait_lower].items():
                    if attr in personality:
                        personality[attr] = (personality[attr] + value) / 2
        
        # Si no se encontraron rasgos, usar valores por defecto según nicho
        if trait_count == 0:
            self._adjust_by_niche(personality, niche)
        
        # Ajustar atributos derivados
        self._derive_secondary_attributes(personality)
        
        # Generar expresiones verbales
        personality["verbal_expressions"] = self._generate_verbal_expressions(personality)
        
        logger.info(f"Generated personality with traits: {traits} for niche: {niche}")
        return personality

    def _adjust_by_niche(self, personality: Dict[str, Any], niche: str) -> None:
        """Ajusta la personalidad según el nicho"""
        niche_adjustments = {
            "finance": {
                "formality": 0.7,
                "enthusiasm": 0.6,
                "directness": 0.8,
                "humor": 0.3,
                "vocabulary_level": 0.7,
                "use_technical_terms": True
            },
            "health": {
                "formality": 0.6,
                "enthusiasm": 0.7,
                "directness": 0.6,
                "humor": 0.4,
                "vocabulary_level": 0.6,
                "use_technical_terms": True
            },
            "gaming": {
                "formality": 0.3,
                "enthusiasm": 0.8,
                "directness": 0.7,
                "humor": 0.7,
                "vocabulary_level": 0.5,
                "use_slang": True,
                "use_emojis": True
            },
            "technology": {
                "formality": 0.6,
                "enthusiasm": 0.6,
                "directness": 0.7,
                "humor": 0.4,
                "vocabulary_level": 0.8,
                "use_technical_terms": True
            },
            "humor": {
                "formality": 0.2,
                "enthusiasm": 0.9,
                "directness": 0.6,
                "humor": 0.9,
                "vocabulary_level": 0.4,
                "use_slang": True,
                "use_emojis": True
            },
            "education": {
                "formality": 0.7,
                "enthusiasm": 0.6,
                                "directness": 0.7,
                "humor": 0.4,
                "vocabulary_level": 0.8,
                "use_technical_terms": True
            }
        }
        
        # Aplicar ajustes del nicho si existe
        if niche.lower() in niche_adjustments:
            for attr, value in niche_adjustments[niche.lower()].items():
                personality[attr] = value

    def _derive_secondary_attributes(self, personality: Dict[str, Any]) -> None:
        """Deriva atributos secundarios basados en los primarios"""
        # Determinar uso de emojis
        if not "use_emojis" in personality or personality["use_emojis"] is None:
            personality["use_emojis"] = personality["formality"] < 0.5 and personality["enthusiasm"] > 0.6
            
        # Determinar uso de jerga
        if not "use_slang" in personality or personality["use_slang"] is None:
            personality["use_slang"] = personality["formality"] < 0.4
            
        # Determinar uso de términos técnicos
        if not "use_technical_terms" in personality or personality["use_technical_terms"] is None:
            personality["use_technical_terms"] = personality["vocabulary_level"] > 0.6
            
        # Determinar longitud de frases
        if not "sentence_length" in personality or personality["sentence_length"] is None:
            personality["sentence_length"] = (personality["formality"] + personality["vocabulary_level"]) / 2

    def _generate_verbal_expressions(self, personality: Dict[str, Any]) -> Dict[str, List[str]]:
        """Genera expresiones verbales características basadas en la personalidad"""
        expressions = {
            "greetings": [],
            "transitions": [],
            "emphasis": [],
            "conclusions": [],
            "reactions": []
        }
        
        # Saludos
        if personality["formality"] > 0.7:
            expressions["greetings"].extend([
                "Saludos a todos", 
                "Bienvenidos", 
                "Es un placer estar con ustedes"
            ])
        elif personality["formality"] < 0.3:
            expressions["greetings"].extend([
                "¡Hola a todos!", 
                "¡Qué tal gente!", 
                "¡Hey! ¿Cómo están?"
            ])
        else:
            expressions["greetings"].extend([
                "Hola a todos", 
                "Bienvenidos a un nuevo video", 
                "¿Cómo están?"
            ])
            
        # Transiciones
        if personality["directness"] > 0.7:
            expressions["transitions"].extend([
                "Vamos al punto", 
                "Ahora veamos", 
                "Pasemos a lo importante"
            ])
        elif personality["directness"] < 0.3:
            expressions["transitions"].extend([
                "Podríamos considerar que", 
                "Tal vez sea interesante ver", 
                "¿Y si exploramos?"
            ])
        else:
            expressions["transitions"].extend([
                "Ahora veremos", 
                "Continuemos con", 
                "El siguiente punto es"
            ])
            
        # Énfasis
        if personality["enthusiasm"] > 0.7:
            expressions["emphasis"].extend([
                "¡Esto es increíble!", 
                "¡No te lo vas a creer!", 
                "¡Es impresionante!"
            ])
        elif personality["enthusiasm"] < 0.3:
            expressions["emphasis"].extend([
                "Es interesante notar", 
                "Vale la pena mencionar", 
                "Consideren esto"
            ])
        else:
            expressions["emphasis"].extend([
                "Esto es importante", 
                "Presten atención a esto", 
                "No pasen por alto"
            ])
            
        # Conclusiones
        if personality["formality"] > 0.7:
            expressions["conclusions"].extend([
                "En conclusión", 
                "Para finalizar", 
                "En resumen"
            ])
        elif personality["formality"] < 0.3:
            expressions["conclusions"].extend([
                "Y eso es todo", 
                "¡Y listo!", 
                "Así que ya saben"
            ])
        else:
            expressions["conclusions"].extend([
                "Para terminar", 
                "Con esto concluimos", 
                "Finalmente"
            ])
            
        # Reacciones
        if personality["humor"] > 0.7:
            expressions["reactions"].extend([
                "¡Vaya, eso no me lo esperaba!", 
                "¡Qué locura!", 
                "¡No me digas!"
            ])
        elif personality["humor"] < 0.3:
            expressions["reactions"].extend([
                "Interesante", 
                "Comprendo", 
                "Entiendo"
            ])
        else:
            expressions["reactions"].extend([
                "Eso es sorprendente", 
                "Vaya, no lo sabía", 
                "Qué interesante"
            ])
            
        # Añadir emojis si corresponde
        if personality.get("use_emojis", False):
            for category in expressions:
                expressions[category] = [f"{expr} 😊" if random.random() > 0.5 else expr 
                                        for expr in expressions[category]]
                
        return expressions

    def adjust_by_sentiment(self, personality: Dict[str, Any], sentiment_score: float) -> Dict[str, Any]:
        """
        Ajusta la personalidad según el sentimiento de la audiencia
        
        Args:
            personality: Diccionario con atributos de personalidad
            sentiment_score: Puntuación de sentimiento (-1 a 1)
            
        Returns:
            Personalidad ajustada
        """
        # Crear copia para no modificar el original
        adjusted = personality.copy()
        
        # Ajustar según sentimiento
        if sentiment_score > 0.5:  # Muy positivo
            # Aumentar entusiasmo y humor
            adjusted["enthusiasm"] = min(1.0, adjusted["enthusiasm"] + 0.1)
            adjusted["humor"] = min(1.0, adjusted["humor"] + 0.1)
            # Reducir formalidad
            adjusted["formality"] = max(0.0, adjusted["formality"] - 0.1)
            
        elif sentiment_score > 0.2:  # Positivo
            # Ligero aumento de entusiasmo
            adjusted["enthusiasm"] = min(1.0, adjusted["enthusiasm"] + 0.05)
            
        elif sentiment_score < -0.5:  # Muy negativo
            # Aumentar formalidad y reducir humor
            adjusted["formality"] = min(1.0, adjusted["formality"] + 0.1)
            adjusted["humor"] = max(0.0, adjusted["humor"] - 0.1)
            # Ajustar entusiasmo según el contexto
            if adjusted["enthusiasm"] > 0.7:
                adjusted["enthusiasm"] = max(0.5, adjusted["enthusiasm"] - 0.1)
                
        elif sentiment_score < -0.2:  # Negativo
            # Ligero aumento de formalidad
            adjusted["formality"] = min(1.0, adjusted["formality"] + 0.05)
            
        # Recalcular atributos derivados
        self._derive_secondary_attributes(adjusted)
        
        # Regenerar expresiones verbales
        adjusted["verbal_expressions"] = self._generate_verbal_expressions(adjusted)
        
        logger.info(f"Adjusted personality by sentiment score: {sentiment_score}")
        return adjusted

    def adapt_script(self, script_template: str, personality: Dict[str, Any], 
                    variables: Dict[str, str]) -> str:
        """
        Adapta un guion al estilo de personalidad
        
        Args:
            script_template: Plantilla del guion
            personality: Diccionario con atributos de personalidad
            variables: Variables para reemplazar en la plantilla
            
        Returns:
            Guion adaptado al personaje
        """
        # Reemplazar variables en la plantilla
        script = script_template
        for key, value in variables.items():
            script = script.replace(f"{{{key}}}", value)
            
        # Dividir en párrafos
        paragraphs = script.split('\n\n')
        adapted_paragraphs = []
        
        # Obtener expresiones verbales
        expressions = personality.get("verbal_expressions", {})
        greetings = expressions.get("greetings", ["Hola a todos"])
        transitions = expressions.get("transitions", ["Ahora veamos"])
        emphasis = expressions.get("emphasis", ["Esto es importante"])
        conclusions = expressions.get("conclusions", ["Para terminar"])
        
        # Adaptar primer párrafo (introducción)
        if paragraphs:
            first_paragraph = paragraphs[0]
            # Añadir saludo si no hay uno
            if not any(g.lower() in first_paragraph.lower() for g in ["hola", "saludos", "bienvenidos"]):
                greeting = random.choice(greetings)
                first_paragraph = f"{greeting}. {first_paragraph}"
            adapted_paragraphs.append(first_paragraph)
            
        # Adaptar párrafos intermedios
        for i, paragraph in enumerate(paragraphs[1:-1] if len(paragraphs) > 2 else []):
            # Añadir transiciones ocasionalmente
            if random.random() < 0.3:
                transition = random.choice(transitions)
                paragraph = f"{transition}. {paragraph}"
                
            # Añadir énfasis ocasionalmente
            if random.random() < 0.3 and personality["enthusiasm"] > 0.5:
                # Buscar una frase importante para enfatizar
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                if len(sentences) > 1:
                    emphasis_idx = random.randint(0, len(sentences) - 1)
                    emphasis_phrase = random.choice(emphasis)
                    sentences[emphasis_idx] = f"{emphasis_phrase} {sentences[emphasis_idx]}"
                    paragraph = " ".join(sentences)
                    
            adapted_paragraphs.append(paragraph)
            
        # Adaptar último párrafo (conclusión)
        if len(paragraphs) > 1:
            last_paragraph = paragraphs[-1]
            # Añadir conclusión si no hay una
            if not any(c.lower() in last_paragraph.lower() for c in ["conclusión", "finalmente", "para terminar"]):
                conclusion = random.choice(conclusions)
                last_paragraph = f"{conclusion}, {last_paragraph}"
            adapted_paragraphs.append(last_paragraph)
            
        # Unir párrafos adaptados
        adapted_script = "\n\n".join(adapted_paragraphs)
        
        # Ajustar según formalidad
        if personality["formality"] > 0.7:
            # Más formal: eliminar contracciones, usar lenguaje más formal
            adapted_script = adapted_script.replace("no puedo", "no es posible")
            adapted_script = adapted_script.replace("no sé", "desconozco")
            adapted_script = adapted_script.replace("creo que", "considero que")
        elif personality["formality"] < 0.3:
            # Menos formal: usar contracciones, lenguaje más casual
            adapted_script = adapted_script.replace("vamos a", "vamos a")
            adapted_script = adapted_script.replace("estoy", "estoy")
            
        # Ajustar según entusiasmo
        if personality["enthusiasm"] > 0.7:
            # Más entusiasta: añadir signos de exclamación
            adapted_script = re.sub(r'(?<=[a-zA-Z])\. ', '! ', adapted_script)
        elif personality["enthusiasm"] < 0.3:
            # Menos entusiasta: eliminar signos de exclamación
            adapted_script = adapted_script.replace("!", ".")
            
        # Añadir emojis si corresponde
        if personality.get("use_emojis", False):
            # Añadir emojis al final de algunas frases
            sentences = re.split(r'(?<=[.!?])\s+', adapted_script)
            for i in range(len(sentences)):
                if random.random() < 0.2:  # 20% de probabilidad
                    emoji = random.choice(["😊", "👍", "💯", "🔥", "✨", "💪", "👏", "🙌"])
                    sentences[i] = f"{sentences[i]} {emoji}"
            adapted_script = " ".join(sentences)
            
        # Añadir jerga si corresponde
        if personality.get("use_slang", False) and personality["niche"] == "gaming":
            # Reemplazar términos con jerga de gaming
            adapted_script = adapted_script.replace("muy bueno", "OP")
            adapted_script = adapted_script.replace("impresionante", "épico")
            adapted_script = adapted_script.replace("difícil", "hardcore")
            
        # Añadir términos técnicos si corresponde
        if personality.get("use_technical_terms", False):
            if personality["niche"] == "finance":
                # Reemplazar términos con jerga financiera
                adapted_script = adapted_script.replace("dinero", "capital")
                adapted_script = adapted_script.replace("ganancias", "rendimientos")
            elif personality["niche"] == "technology":
                # Reemplazar términos con jerga tecnológica
                adapted_script = adapted_script.replace("programa", "software")
                adapted_script = adapted_script.replace("teléfono", "dispositivo móvil")
                
        logger.info("Adapted script to personality")
        return adapted_script

    def generate_character_response(self, personality: Dict[str, Any], 
                                   prompt: str, context: str = "") -> str:
        """
        Genera una respuesta de personaje a un comentario o pregunta
        
        Args:
            personality: Diccionario con atributos de personalidad
            prompt: Comentario o pregunta a responder
            context: Contexto adicional (opcional)
            
        Returns:
            Respuesta generada
        """
        # Obtener expresiones verbales
        expressions = personality.get("verbal_expressions", {})
        reactions = expressions.get("reactions", ["Interesante"])
        
        # Determinar tono de respuesta según personalidad
        if personality["formality"] > 0.7:
            # Formal
            response_templates = [
                "Gracias por su comentario. {reaction}",
                "Agradezco su participación. {reaction}",
                "Valoro mucho su opinión. {reaction}"
            ]
        elif personality["formality"] < 0.3:
            # Informal
            response_templates = [
                "¡Gracias por comentar! {reaction}",
                "¡Me encanta leer tus comentarios! {reaction}",
                "¡Qué bueno verte por aquí! {reaction}"
            ]
        else:
            # Neutral
            response_templates = [
                "Gracias por tu comentario. {reaction}",
                "Aprecio tu participación. {reaction}",
                "Gracias por compartir tu opinión. {reaction}"
            ]
            
        # Seleccionar plantilla y reacción
        template = random.choice(response_templates)
        reaction = random.choice(reactions)
        
        # Generar respuesta base
        response = template.format(reaction=reaction)
        
        # Añadir emojis si corresponde
        if personality.get("use_emojis", False):
            emoji = random.choice(["😊", "👍", "💯", "🔥", "✨", "💪", "👏", "🙌"])
            response = f"{response} {emoji}"
            
        logger.info("Generated character response")
        return response

    def save_personality(self, personality_id: str, personality: Dict[str, Any]) -> bool:
        """
        Guarda una personalidad en la base de conocimiento
        
        Args:
            personality_id: Identificador único de la personalidad
            personality: Diccionario con atributos de personalidad
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            self.knowledge_base.save_personality(personality_id, personality)
            logger.info(f"Saved personality: {personality_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving personality: {str(e)}")
            return False

    def load_personality(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """
        Carga una personalidad desde la base de conocimiento
        
        Args:
            personality_id: Identificador único de la personalidad
            
        Returns:
            Diccionario con atributos de personalidad o None si no existe
        """
        try:
            personality = self.knowledge_base.get_personality(personality_id)
            if personality:
                logger.info(f"Loaded personality: {personality_id}")
            else:
                logger.warning(f"Personality not found: {personality_id}")
            return personality
        except Exception as e:
            logger.error(f"Error loading personality: {str(e)}")
            return None
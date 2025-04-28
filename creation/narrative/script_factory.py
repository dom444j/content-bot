"""
Script Factory - Generador de guiones para contenido multimedia

Este módulo genera guiones optimizados para diferentes plataformas:
- Guiones para videos cortos (TikTok, Reels, Shorts)
- Guiones para videos largos (YouTube)
- Integración de CTAs estratégicos en momentos óptimos
- Adaptación a diferentes nichos y personajes
"""

import os
import sys
import json
import logging
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple
import re

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase
from creation.narrative.story_engine import StoryEngine
from creation.narrative.cta_generator import CTAGenerator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'script_factory.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ScriptFactory')

class ScriptFactory:
    """
    Generador de guiones que crea scripts optimizados para diferentes
    plataformas, nichos y personajes.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ScriptFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el generador de guiones si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Script Factory...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar motor de narrativas
        self.story_engine = StoryEngine()
        
        # Cargar generador de CTAs
        self.cta_generator = CTAGenerator()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Cargar plantillas de frases virales
        self.viral_phrases_file = os.path.join('datasets', 'viral_phrases.json')
        self.viral_phrases = self._load_viral_phrases()
        
        # Configuración de estilos de guion por plataforma
        self.script_styles = {
            'tiktok': {
                'tone': 'casual',
                'pacing': 'fast',
                'sentence_length': 'short',
                'hook_intensity': 'high',
                'cta_frequency': 'high',
                'emoji_usage': 'high',
                'trending_references': 'high'
            },
            'instagram': {
                'tone': 'aspirational',
                'pacing': 'medium',
                                'sentence_length': 'medium',
                'hook_intensity': 'medium',
                'cta_frequency': 'medium',
                'emoji_usage': 'medium',
                'trending_references': 'medium'
            },
            'youtube_shorts': {
                'tone': 'energetic',
                'pacing': 'fast',
                'sentence_length': 'short',
                'hook_intensity': 'very_high',
                'cta_frequency': 'high',
                'emoji_usage': 'medium',
                'trending_references': 'high'
            },
            'youtube': {
                'tone': 'informative',
                'pacing': 'slow',
                'sentence_length': 'long',
                'hook_intensity': 'medium',
                'cta_frequency': 'low',
                'emoji_usage': 'low',
                'trending_references': 'low'
            }
        }
        
        # Configuración de hooks por nicho
        self.hook_templates = {
            'finance': [
                "¿Sabías que el {porcentaje}% de las personas pierden dinero por este error?",
                "El secreto que los bancos no quieren que sepas sobre {tema}",
                "Cómo gané {cantidad} euros en solo {tiempo} usando esta estrategia",
                "La regla del {numero} que cambió mi situación financiera",
                "El método {nombre} que está ayudando a miles a {beneficio}"
            ],
            'health': [
                "Descubrí por qué no {problema_salud} a pesar de {acción}",
                "El ingrediente que deberías evitar para {beneficio_salud}",
                "Haz esto por {tiempo} días y notarás {cambio_positivo}",
                "Lo que tu {profesional_salud} no te dice sobre {tema_salud}",
                "La rutina de {minutos} minutos que transformó mi {aspecto_salud}"
            ],
            'technology': [
                "Este truco de {dispositivo} que nadie conoce...",
                "La función oculta de {app} que deberías estar usando",
                "Probé {producto_tech} durante {tiempo} y esto es lo que descubrí",
                "La configuración que mejora {porcentaje}% el rendimiento de tu {dispositivo}",
                "El error de {tecnología} que todos cometen y cómo solucionarlo"
            ],
            'gaming': [
                "Cómo vencer a {porcentaje}% de jugadores con este truco en {juego}",
                "La estrategia secreta que usan los pros de {juego}",
                "Este ajuste de configuración te dará ventaja en {juego}",
                "El arma/personaje más infravalorado en {juego} y por qué deberías usarlo",
                "Descubrí un glitch en {juego} que te permite {ventaja}"
            ],
            'humor': [
                "Cuando intentas {acción_cotidiana} pero {situación_cómica}",
                "POV: {situación_humorística}",
                "Nadie: / Absolutamente nadie: / Yo: {acción_inesperada}",
                "Mi reacción cuando {situación}",
                "Lo que realmente pasa cuando {situación_vs_expectativa}"
            ]
        }
        
        self._initialized = True
        logger.info("Script Factory inicializado correctamente")
    
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
    
    def _load_viral_phrases(self) -> Dict:
        """Carga las plantillas de frases virales desde el archivo JSON"""
        try:
            if os.path.exists(self.viral_phrases_file):
                with open(self.viral_phrases_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de frases virales no encontrado: {self.viral_phrases_file}")
                return self._create_default_viral_phrases()
        except Exception as e:
            logger.error(f"Error al cargar frases virales: {str(e)}")
            return self._create_default_viral_phrases()
    
    def _create_default_viral_phrases(self) -> Dict:
        """Crea plantillas de frases virales por defecto"""
        default_phrases = {
            "hooks": {
                "general": [
                    "No vas a creer lo que descubrí...",
                    "Esto cambiará la forma en que ves {tema}",
                    "El secreto que nadie te cuenta sobre {tema}",
                    "Deja de hacer {acción_común} hasta que veas esto",
                    "Lo que deberías saber antes de {acción}"
                ],
                "finance": [
                    "Así es como puedes {beneficio_financiero} sin {sacrificio}",
                    "El error financiero que está costándote {cantidad}",
                    "Cómo {acción_financiera} como un profesional",
                    "La estrategia que los ricos usan para {beneficio}",
                    "Esto es lo que los bancos no quieren que sepas"
                ]
            },
            "transitions": [
                "Pero eso no es todo...",
                "Aquí viene lo interesante...",
                "Y ahora, lo más importante...",
                "Esto te va a sorprender...",
                "Pero espera, hay más..."
            ],
            "ctas": [
                "Si quieres saber más, sigue mi cuenta",
                "Dale like si te ha servido",
                "Comenta qué te pareció",
                "Guarda este video para más tarde",
                "Comparte con alguien que necesite ver esto"
            ],
            "closings": [
                "Y así es como puedes {beneficio_final}",
                "Ahora ya sabes cómo {resultado_positivo}",
                "Pruébalo y cuéntame en los comentarios",
                "¿Tú qué opinas? Déjamelo saber",
                "Nos vemos en el próximo video"
            ]
        }
        
        # Guardar frases virales por defecto
        os.makedirs(os.path.dirname(self.viral_phrases_file), exist_ok=True)
        with open(self.viral_phrases_file, 'w', encoding='utf-8') as f:
            json.dump(default_phrases, f, indent=4)
        
        return default_phrases
    
    def generate_script(self, topic: str, niche: str, platform: str, 
                       duration_type: str = 'medium', keywords: List[str] = None,
                       persona: str = None, story_structure_id: str = None) -> Dict:
        """
        Genera un guion completo para un contenido específico
        
        Args:
            topic: Tema principal del contenido
            niche: Nicho de contenido (finance, health, etc.)
            platform: Plataforma (youtube, tiktok, etc.)
            duration_type: Tipo de duración (short, medium, long)
            keywords: Palabras clave a incluir
            persona: Personalidad del creador (opcional)
            story_structure_id: ID de estructura narrativa existente (opcional)
            
        Returns:
            Diccionario con el guion completo
        """
        # Obtener o generar estructura narrativa
        if story_structure_id:
            story_structure = self.story_engine.get_story_structure(story_structure_id)
            if not story_structure:
                logger.warning(f"Estructura narrativa {story_structure_id} no encontrada, generando nueva")
                story_structure = self.story_engine.generate_story_structure(niche, platform, duration_type)
        else:
            story_structure = self.story_engine.generate_story_structure(niche, platform, duration_type)
        
        # Obtener segmentos de la historia
        segments = self.story_engine.get_story_segments(story_structure)
        
        # Obtener estilo de guion para la plataforma
        script_style = self.script_styles.get(platform, self.script_styles['tiktok'])
        
        # Generar hook basado en el nicho
        hook = self._generate_hook(topic, niche)
        
        # Generar CTA
        cta_time = story_structure.get('cta', {}).get('time', 0)
        cta = self.cta_generator.generate_cta(niche, platform, topic)
        
        # Generar contenido para cada segmento
        script_segments = []
        
        for segment in segments:
            segment_content = self._generate_segment_content(
                segment, topic, niche, platform, script_style, keywords
            )
            
            # Añadir CTA si corresponde a este segmento
            if segment.get('contains_cta', False):
                segment_content['cta'] = cta
                segment_content['cta_time'] = cta_time
            
            script_segments.append(segment_content)
        
        # Generar cierre
        closing = self._generate_closing(topic, niche)
        
        # Crear guion completo
        script = {
            'id': f"script_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}",
            'topic': topic,
            'niche': niche,
            'platform': platform,
            'duration_type': duration_type,
            'target_duration': story_structure.get('target_duration', 60),
            'hook': hook,
            'segments': script_segments,
            'closing': closing,
            'cta': cta,
            'cta_time': cta_time,
            'keywords': keywords or [],
            'persona': persona,
            'story_structure_id': story_structure.get('id', None),
            'created_at': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Guion generado para {topic} en {platform}, duración {duration_type}")
        
        return script
    
    def _generate_hook(self, topic: str, niche: str) -> str:
        """Genera un hook atractivo basado en el nicho y tema"""
        # Obtener plantillas de hook para el nicho
        hook_templates = self.hook_templates.get(niche, self.hook_templates.get('finance', []))
        
        if not hook_templates:
            # Usar plantillas generales si no hay específicas
            hook_templates = self.viral_phrases.get('hooks', {}).get('general', [
                "No vas a creer lo que descubrí sobre {tema}...",
                "Esto cambiará la forma en que ves {tema}"
            ])
        
        # Seleccionar plantilla aleatoria
        template = random.choice(hook_templates)
        
        # Reemplazar variables en la plantilla
        hook = template.replace('{tema}', topic)
        
        # Reemplazar otras variables comunes
        replacements = {
            '{porcentaje}': str(random.randint(70, 95)),
            '{cantidad}': f"{random.randint(1, 5)},{random.randint(100, 999)}",
            '{tiempo}': f"{random.randint(1, 30)} días",
            '{numero}': str(random.randint(1, 10)),
            '{nombre}': random.choice(['80/20', '4%', 'Warren Buffett', 'japonés']),
            '{beneficio}': random.choice(['ahorrar más', 'invertir mejor', 'ganar dinero extra']),
            '{problema_salud}': random.choice(['pierdes peso', 'duermes bien', 'tienes energía']),
            '{acción}': random.choice(['intentarlo todo', 'seguir consejos comunes', 'gastar dinero']),
            '{beneficio_salud}': random.choice(['perder peso', 'dormir mejor', 'tener más energía']),
            '{cambio_positivo}': random.choice(['la diferencia', 'resultados increíbles', 'una transformación']),
            '{profesional_salud}': random.choice(['médico', 'nutricionista', 'entrenador']),
            '{tema_salud}': random.choice(['nutrición', 'ejercicio', 'suplementos']),
            '{minutos}': str(random.randint(5, 30)),
            '{aspecto_salud}': random.choice(['salud', 'físico', 'energía']),
            '{dispositivo}': random.choice(['iPhone', 'Android', 'Windows', 'Mac']),
            '{app}': random.choice(['WhatsApp', 'Instagram', 'TikTok', 'YouTube']),
            '{producto_tech}': random.choice(['este gadget', 'esta app', 'este dispositivo']),
            '{tecnología}': random.choice(['WiFi', 'smartphone', 'computadora']),
            '{juego}': random.choice(['Fortnite', 'Minecraft', 'Roblox', 'Call of Duty']),
            '{ventaja}': random.choice(['ganar siempre', 'conseguir recursos infinitos', 'subir de nivel rápido']),
            '{acción_cotidiana}': random.choice(['cocinar', 'estudiar', 'trabajar', 'hacer ejercicio']),
            '{situación_cómica}': random.choice(['todo sale mal', 'tu perro te juzga', 'tu madre aparece']),
            '{situación_humorística}': random.choice(['eres el último en enterarte', 'nadie te avisó', 'todos te miran']),
            '{acción_inesperada}': random.choice(['bailando en pijama', 'comiendo a las 3am', 'hablando solo']),
            '{situación}': random.choice(['el profesor me pregunta', 'mi jefe me llama', 'veo mi ex']),
            '{situación_vs_expectativa}': random.choice(['trabajas desde casa', 'cocinas algo nuevo', 'haces ejercicio'])
        }
        
        for placeholder, value in replacements.items():
            hook = hook.replace(placeholder, value)
        
        return hook
    
    def _generate_segment_content(self, segment: Dict, topic: str, niche: str, 
                                 platform: str, style: Dict, keywords: List[str] = None) -> Dict:
        """Genera contenido para un segmento específico del guion"""
        # Crear estructura base del segmento
        segment_content = {
            'index': segment['index'],
            'description': segment['description'],
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['duration'],
            'content': ""
        }
        
        # Generar contenido basado en la descripción del segmento
        description = segment['description'].lower()
        
        # Determinar tipo de contenido basado en la descripción
        if any(word in description for word in ['introducción', 'problema', 'situación']):
            segment_content['content'] = self._generate_introduction_content(topic, niche, style)
        elif any(word in description for word in ['desarrollo', 'explicación', 'causas']):
            segment_content['content'] = self._generate_development_content(topic, niche, style)
        elif any(word in description for word in ['solución', 'método', 'estrategia']):
            segment_content['content'] = self._generate_solution_content(topic, niche, style)
        elif any(word in description for word in ['beneficios', 'resultados', 'ventajas']):
            segment_content['content'] = self._generate_benefits_content(topic, niche, style)
        elif any(word in description for word in ['conclusión', 'resumen', 'cierre']):
            segment_content['content'] = self._generate_conclusion_content(topic, niche, style)
        else:
            # Contenido genérico si no coincide con ninguna categoría
            segment_content['content'] = f"En este segmento hablamos sobre {topic} y cómo se relaciona con {segment['description']}."
        
        # Añadir transición si no es el último segmento
        if segment['index'] < 6:  # Asumiendo máximo 7 segmentos (0-6)
            transition = random.choice(self.viral_phrases.get('transitions', [
                "Pero eso no es todo...",
                "Aquí viene lo interesante...",
                "Y ahora, lo más importante..."
            ]))
            segment_content['content'] += f" {transition}"
        
        # Insertar keywords si están disponibles
        if keywords:
            for keyword in keywords:
                # Solo insertar si no está ya presente
                if keyword.lower() not in segment_content['content'].lower():
                    # Encontrar un punto o coma para insertar después
                    match = re.search(r'[.,]', segment_content['content'])
                    if match:
                        pos = match.start() + 1
                        segment_content['content'] = (
                            segment_content['content'][:pos] + 
                            f" Hablando de {keyword}," + 
                            segment_content['content'][pos:]
                        )
                        break
        
        # Ajustar estilo según la plataforma
        segment_content['content'] = self._adjust_content_style(segment_content['content'], style)
        
        return segment_content
    
    def _generate_introduction_content(self, topic: str, niche: str, style: Dict) -> str:
        """Genera contenido para segmentos de introducción"""
        intros = [
            f"Hoy vamos a hablar de {topic}, algo que muchos encuentran difícil de entender.",
            f"¿Alguna vez te has preguntado sobre {topic}? No estás solo.",
            f"El {topic} es uno de los temas más importantes en {niche} actualmente.",
            f"Muchas personas cometen errores con {topic} que les cuestan tiempo y dinero.",
            f"Descubrí algo sorprendente sobre {topic} que cambió mi perspectiva."
        ]
        return random.choice(intros)
    
    def _generate_development_content(self, topic: str, niche: str, style: Dict) -> str:
        """Genera contenido para segmentos de desarrollo"""
        developments = [
            f"Lo que poca gente sabe es que {topic} funciona de manera diferente a lo que nos han enseñado.",
            f"Existen tres factores clave que determinan el éxito en {topic}.",
            f"Según los expertos, el {topic} se basa en principios que cualquiera puede aplicar.",
            f"La mayoría comete el error de pensar que {topic} es complicado, pero la realidad es más simple.",
            f"Después de investigar durante meses sobre {topic}, descubrí un patrón interesante."
        ]
        return random.choice(developments)
    
    def _generate_solution_content(self, topic: str, niche: str, style: Dict) -> str:
        """Genera contenido para segmentos de solución"""
        solutions = [
            f"La solución para dominar {topic} se resume en tres pasos sencillos.",
            f"He desarrollado un método que simplifica {topic} y lo hace accesible para todos.",
            f"El secreto para resolver los problemas de {topic} está en cambiar tu enfoque.",
            f"Aplicando esta técnica, podrás mejorar tus resultados con {topic} en poco tiempo.",
            f"La estrategia que voy a compartir ha ayudado a miles de personas con {topic}."
        ]
        return random.choice(solutions)
    
    def _generate_benefits_content(self, topic: str, niche: str, style: Dict) -> str:
        """Genera contenido para segmentos de beneficios"""
        benefits = [
            f"Al implementar esto, notarás mejoras inmediatas en tus resultados con {topic}.",
            f"Los beneficios de este enfoque incluyen ahorro de tiempo, mejores resultados y menos estrés.",
            f"Quienes han aplicado este método han visto un cambio radical en su experiencia con {topic}.",
            f"No solo mejorarás en {topic}, sino que también notarás beneficios en otras áreas relacionadas.",
            f"El impacto positivo que esto tendrá en tu dominio de {topic} es difícil de exagerar."
        ]
        return random.choice(benefits)
    
    def _generate_conclusion_content(self, topic: str, niche: str, style: Dict) -> str:
        """Genera contenido para segmentos de conclusión"""
        conclusions = [
            f"En resumen, {topic} no tiene que ser complicado si sigues estos consejos.",
            f"Ahora tienes las herramientas para dominar {topic} como un profesional.",
            f"Recuerda que la clave para el éxito con {topic} es la consistencia y la práctica.",
            f"Con estos conocimientos, estás mejor preparado que el 90% de las personas en {topic}.",
            f"Implementa estos cambios y verás la diferencia en tu experiencia con {topic}."
        ]
        return random.choice(conclusions)
    
    def _generate_closing(self, topic: str, niche: str) -> str:
        """Genera un cierre para el guion"""
        closings = self.viral_phrases.get('closings', [
            "Y así es como puedes dominar {tema}",
            "Ahora ya sabes cómo mejorar en {tema}",
            "Pruébalo y cuéntame en los comentarios",
            "¿Tú qué opinas? Déjamelo saber",
            "Nos vemos en el próximo video"
        ])
        
        closing = random.choice(closings).replace('{tema}', topic)
        
        # Reemplazar otras variables comunes
        replacements = {
            '{beneficio_final}': random.choice(['mejorar en este tema', 'obtener mejores resultados', 'dominar esta habilidad']),
            '{resultado_positivo}': random.choice(['resolver este problema', 'mejorar en este aspecto', 'dominar esta técnica'])
        }
        
        for placeholder, value in replacements.items():
            closing = closing.replace(placeholder, value)
        
        return closing
    
    def _adjust_content_style(self, content: str, style: Dict) -> str:
        """Ajusta el contenido según el estilo de la plataforma"""
        adjusted_content = content
        
        # Ajustar longitud de frases
        if style['sentence_length'] == 'short':
            # Dividir frases largas
            adjusted_content = re.sub(r'([.!?]) ([A-Z])', r'\1\n\2', adjusted_content)
        
        # Ajustar tono
        if style['tone'] == 'casual':
            # Hacer más casual
            adjusted_content = adjusted_content.replace('Adicionalmente', 'Además')
            adjusted_content = adjusted_content.replace('Sin embargo', 'Pero')
            adjusted_content = adjusted_content.replace('Por lo tanto', 'Así que')
        
        # Añadir emojis según nivel de uso
        if style['emoji_usage'] == 'high':
            emojis = ['🔥', '💯', '⚡', '🤯', '👀', '💪', '🚀', '✅', '💰', '🎯']
            # Añadir emoji al final de cada frase
            adjusted_content = re.sub(r'([.!?]) ', lambda m: f"{m.group(1)} {random.choice(emojis)} ", adjusted_content)
        elif style['emoji_usage'] == 'medium':
            emojis = ['🔥', '💯', '⚡', '🤯', '👀']
            # Añadir emoji ocasionalmente
            if random.random() > 0.5:
                adjusted_content += f" {random.choice(emojis)}"
        
        return adjusted_content
    
    def save_script(self, script: Dict) -> str:
        """
        Guarda un guion en la base de conocimiento
        
        Args:
            script: Guion a guardar
            
        Returns:
            ID del guion guardado
        """
        # Generar ID si no existe
        if 'id' not in script:
            script['id'] = f"script_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Guardar en la base de conocimiento
        self.kb.save_script(script)
        
        logger.info(f"Guion guardado con ID: {script['id']}")
        
        return script['id']
    
    def get_script(self, script_id: str) -> Optional[Dict]:
        """
        Recupera un guion de la base de conocimiento
        
        Args:
            script_id: ID del guion
            
        Returns:
            Guion o None si no se encuentra
        """
        return self.kb.get_script(script_id)
    
    def adapt_script_for_platform(self, script: Dict, target_platform: str) -> Dict:
        """
        Adapta un guion para una plataforma diferente
        
        Args:
            script: Guion original
            target_platform: Plataforma destino
            
        Returns:
            Guion adaptado
        """
        # Copiar guion original
        adapted_script = script.copy()
        
        # Actualizar plataforma
        adapted_script['platform'] = target_platform
        
        # Obtener estilo para la nueva plataforma
        target_style = self.script_styles.get(target_platform, self.script_styles['tiktok'])
        
        # Adaptar contenido de segmentos
        for i, segment in enumerate(adapted_script['segments']):
            segment['content'] = self._adjust_content_style(segment['content'], target_style)
        
        # Adaptar CTA según plataforma
        adapted_script['cta'] = self.cta_generator.generate_cta(
            adapted_script['niche'], 
            target_platform, 
            adapted_script['topic']
        )
        
        # Actualizar timestamp
        adapted_script['adapted_at'] = datetime.datetime.now().isoformat()
        
        logger.info(f"Guion adaptado de {script['platform']} a {target_platform}")
        
        return adapted_script
    
    def generate_script_variations(self, script_id: str, num_variations: int = 3) -> List[Dict]:
        """
        Genera variaciones de un guion existente
        
        Args:
            script_id: ID del guion original
            num_variations: Número de variaciones a generar
            
        Returns:
            Lista de guiones variados
        """
        # Obtener guion original
        original_script = self.get_script(script_id)
        
        if not original_script:
            logger.error(f"Guion {script_id} no encontrado")
            return []
        
        variations = []
        
        for i in range(num_variations):
            # Crear variación basada en el original
            variation = original_script.copy()
            
            # Generar nuevo ID
            variation['id'] = f"{original_script['id']}_var{i+1}"
            
            # Variar hook
            variation['hook'] = self._generate_hook(original_script['topic'], original_script['niche'])
            
            # Variar contenido de segmentos (manteniendo estructura)
            for j, segment in enumerate(variation['segments']):
                # Solo variar el contenido, mantener tiempos y descripción
                segment_copy = segment.copy()
                segment_copy['content'] = self._generate_segment_content(
                    segment, 
                    original_script['topic'], 
                    original_script['niche'],
                    original_script['platform'],
                    self.script_styles.get(original_script['platform'], self.script_styles['tiktok']),
                    original_script.get('keywords', [])
                )['content']
                
                variation['segments'][j] = segment_copy
            
            # Variar cierre
            variation['closing'] = self._generate_closing(original_script['topic'], original_script['niche'])
            
            # Añadir metadatos de variación
            variation['original_script_id'] = original_script['id']
            variation['variation_number'] = i + 1
            variation['created_at'] = datetime.datetime.now().isoformat()
            
            # Guardar variación
            self.save_script(variation)
            
            variations.append(variation)
        
        logger.info(f"Generadas {num_variations} variaciones del guion {script_id}")
        
        return variations
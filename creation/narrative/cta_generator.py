"""
CTA Generator - Generador de llamadas a la acción optimizadas

Este módulo se encarga de generar CTAs (Call To Action) estratégicos:
- Optimizados para diferentes momentos del video (0-3s, 4-8s, finales)
- Personalizados según audiencia y plataforma
- Gamificados para aumentar engagement
- Con seguimiento de reputación y efectividad
"""

import os
import json
import logging
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar base de conocimiento
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'cta_generator.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('CTAGenerator')

class CTAGenerator:
    """
    Generador de llamadas a la acción (CTAs) optimizadas para diferentes
    plataformas, nichos y momentos del video.
    """
    
    def __init__(self):
        """Inicializa el generador de CTAs"""
        logger.info("Inicializando CTA Generator...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar plantillas de CTAs
        self.templates_file = os.path.join('creation', 'narrative', 'cta_templates.json')
        self.templates = self._load_templates()
        
        # Configuración de timing de CTAs por plataforma (en segundos)
        self.cta_timing = {
            'early': {'start': 0, 'end': 3, 'optimal': 2},  # 0-3s
            'middle': {'start': 4, 'end': 8, 'optimal': 6},  # 4-8s
            'end': {'start': -5, 'end': 0, 'optimal': -2}    # Últimos 5s
        }
        
        # Configuración de tipos de CTAs por plataforma
        self.platform_cta_types = {
            'tiktok': ['follow', 'comment', 'like', 'share', 'profile'],
            'instagram': ['follow', 'save', 'share', 'profile', 'dm'],
            'youtube_shorts': ['subscribe', 'like', 'comment', 'notification', 'more'],
            'youtube': ['subscribe', 'like', 'comment', 'notification', 'playlist', 'join'],
            'threads': ['follow', 'like', 'repost', 'quote'],
            'bluesky': ['follow', 'like', 'repost', 'mute']
        }
        
        logger.info("CTA Generator inicializado correctamente")
    
    def _load_templates(self) -> Dict:
        """Carga las plantillas de CTAs desde el archivo JSON"""
        try:
            if os.path.exists(self.templates_file):
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Crear plantillas por defecto si no existe el archivo
                default_templates = self._create_default_templates()
                
                # Guardar plantillas por defecto
                os.makedirs(os.path.dirname(self.templates_file), exist_ok=True)
                with open(self.templates_file, 'w', encoding='utf-8') as f:
                    json.dump(default_templates, f, indent=4, ensure_ascii=False)
                
                return default_templates
        except Exception as e:
            logger.error(f"Error al cargar plantillas de CTAs: {str(e)}")
            return self._create_default_templates()
    
    def _create_default_templates(self) -> Dict:
        """Crea plantillas de CTAs por defecto"""
        return {
            "follow": {
                "early": [
                    "¡No te pierdas nada! Sígueme ahora para más {content_type}",
                    "¿Te gusta este contenido? ¡Sígueme para no perderte nada!",
                    "Sígueme ahora y descubre {benefit} cada día"
                ],
                "middle": [
                    "Si quieres más contenido como este, ¡sígueme ahora!",
                    "¡Dale a seguir para más {content_type} como este!",
                    "Sígueme para {benefit} - ¡Nuevo contenido cada {frequency}!"
                ],
                "end": [
                    "¡No olvides seguirme para más {content_type}!",
                    "¿Te ha gustado? ¡Sígueme para no perderte lo próximo!",
                    "Sígueme ahora y sé el primero en ver mi próximo {content_type}"
                ]
            },
            "like": {
                "early": [
                    "¡Dale like si estás de acuerdo con esto!",
                    "Si te gusta {topic}, dale like ahora",
                    "¡Like si quieres más contenido sobre {topic}!"
                ],
                "middle": [
                    "¿Te está gustando? ¡Dale like ahora!",
                    "Si te está ayudando, ¡apóyame con un like!",
                    "¡Dale like si quieres la parte 2!"
                ],
                "end": [
                    "¡No olvides darle like si te ha gustado!",
                    "¡Like y comparte si te ha servido!",
                    "¡Dale like para que más personas vean este {content_type}!"
                ]
            },
            "comment": {
                "early": [
                    "Comenta '{emoji}' si quieres ver la parte 2",
                    "¿Qué opinas sobre {topic}? ¡Comenta ahora!",
                    "Comenta tu experiencia con {topic} abajo"
                ],
                "middle": [
                    "Comenta '{emoji}' si quieres que haga un video sobre {related_topic}",
                    "¿Cuál es tu opinión? ¡Déjala en los comentarios!",
                    "Comenta '{keyword}' y te responderé personalmente"
                ],
                "end": [
                    "¡Déjame tu pregunta en los comentarios!",
                    "Comenta qué tema quieres que trate en el próximo video",
                    "¡Etiqueta a alguien que necesite ver esto!"
                ]
            },
            "share": {
                "early": [
                    "¡Comparte ahora con alguien que necesite ver esto!",
                    "¡Esta información es demasiado importante para no compartirla!",
                    "Comparte con tus amigos que les interese {topic}"
                ],
                "middle": [
                    "¡Comparte este video antes de que termine!",
                    "¿Conoces a alguien que necesite esta información? ¡Compártela ahora!",
                    "¡Ayúdame a llegar a más personas compartiendo este video!"
                ],
                "end": [
                    "¡Comparte este video con alguien que lo necesite!",
                    "¡No guardes este secreto! Compártelo ahora",
                    "¡Comparte para ayudar a más personas con {topic}!"
                ]
            },
            "subscribe": {
                "early": [
                    "¡Suscríbete ahora para no perderte ningún video!",
                    "¡Suscríbete para más contenido sobre {topic}!",
                    "¡Únete a la familia! Suscríbete ahora"
                ],
                "middle": [
                    "¿Te está gustando? ¡Suscríbete para más!",
                    "¡Suscríbete y activa las notificaciones para no perderte nada!",
                    "¡Suscríbete ahora! Nuevo contenido cada {frequency}"
                ],
                "end": [
                    "¡No olvides suscribirte antes de irte!",
                    "¡Suscríbete para más contenido como este!",
                    "¡Suscríbete ahora y sé parte de la comunidad!"
                ]
            },
            "notification": {
                "early": [
                    "¡Activa la campanita para ser el primero en ver mis videos!",
                    "¡No te pierdas ningún video! Activa las notificaciones",
                    "¡Activa las notificaciones para contenido exclusivo!"
                ],
                "middle": [
                    "¡Activa la campanita para no perderte el próximo video!",
                    "¡Notificaciones activadas = contenido garantizado!",
                    "¡Activa las notificaciones y entérate antes que nadie!"
                ],
                "end": [
                    "¡Activa las notificaciones antes de irte!",
                    "¡No olvides activar la campanita para más contenido!",
                    "¡Campanita activada = próximo video asegurado!"
                ]
            },
            "profile": {
                "early": [
                    "¡Visita mi perfil para más contenido como este!",
                    "¡Tengo más videos sobre {topic} en mi perfil!",
                    "¡Descubre más contenido exclusivo en mi perfil!"
                ],
                "middle": [
                    "¡Visita mi perfil después de este video!",
                    "¡Más contenido sobre {topic} en mi perfil!",
                    "¡Contenido exclusivo esperándote en mi perfil!"
                ],
                "end": [
                    "¡Visita mi perfil para más videos como este!",
                    "¡No te vayas sin revisar mi perfil!",
                    "¡Más sorpresas te esperan en mi perfil!"
                ]
            },
            "join": {
                "early": [
                    "¡Únete a mi membresía para contenido exclusivo!",
                    "¡Conviértete en miembro y accede a {exclusive_content}!",
                    "¡Únete ahora y obtén {benefit} exclusivo!"
                ],
                "middle": [
                    "¡Los miembros del canal ya están disfrutando de {exclusive_content}!",
                    "¡Únete a la membresía y accede a todo mi contenido premium!",
                    "¡Hazte miembro ahora y obtén {benefit} al instante!"
                ],
                "end": [
                    "¡Únete a la membresía antes de irte!",
                    "¡No te pierdas el contenido exclusivo para miembros!",
                    "¡Conviértete en miembro y lleva tu experiencia al siguiente nivel!"
                ]
            },
            "gamified": {
                "early": [
                    "Comenta '{emoji}' si quieres ganar {prize}",
                    "¡Los primeros {number} comentarios recibirán {benefit}!",
                    "¡Adivina el final en los comentarios y gana {prize}!"
                ],
                "middle": [
                    "¡Pausa el video en 3, 2, 1 y comenta lo que ves para ganar {prize}!",
                    "¡Etiqueta a {number} amigos y participa por {prize}!",
                    "¡Comenta la respuesta correcta y podrías ganar {prize}!"
                ],
                "end": [
                    "¡Comenta '{keyword}' en los próximos {timeframe} y participa por {prize}!",
                    "¡Los mejores {number} comentarios recibirán {benefit}!",
                    "¡Comparte este video y envíame captura para participar por {prize}!"
                ]
            },
            "affiliate": {
                "early": [
                    "¡Usa mi código '{code}' para obtener {discount}% de descuento!",
                    "¡Enlace en mi bio para conseguir {product} con {discount}% de descuento!",
                    "¡Oferta especial! {product} con {discount}% usando mi enlace"
                ],
                "middle": [
                    "¡No te pierdas esta oferta! {product} con {discount}% de descuento",
                    "¡Usa mi código '{code}' antes de que termine la oferta!",
                    "¡Consigue {product} ahora con mi enlace en la bio!"
                ],
                "end": [
                    "¡Recuerda usar mi código '{code}' para tu descuento!",
                    "¡Enlace en la bio para {product} con {discount}% de descuento!",
                    "¡Última oportunidad! {product} con descuento en mi bio"
                ]
            },
            "product": {
                "early": [
                    "¡Mi {product_type} ya está disponible! Link en bio",
                    "¡Por fin he lanzado mi {product_type}! Consíguelo ahora",
                    "¡Oferta de lanzamiento de mi nuevo {product_type}!"
                ],
                "middle": [
                    "¡Mi {product_type} te ayudará con {benefit}!",
                    "¡Descubre cómo mi {product_type} ha ayudado a {number} personas!",
                    "¡Mi {product_type} incluye {feature_1}, {feature_2} y más!"
                ],
                "end": [
                    "¡No olvides revisar mi {product_type} en el link de la bio!",
                    "¡Oferta por tiempo limitado en mi {product_type}!",
                    "¡Consigue mi {product_type} antes de que se agote!"
                ]
            }
        }
    
    def generate_cta(self, 
                     platform: str, 
                     niche: str, 
                     content_type: str, 
                     duration: int, 
                     timing: str = 'middle', 
                     cta_type: str = None, 
                     audience_data: Dict = None,
                     character_data: Dict = None) -> Dict:
        """
        Genera un CTA optimizado para una plataforma y momento específicos
        
        Args:
            platform: Plataforma de destino (tiktok, youtube, etc.)
            niche: Nicho de contenido (finance, health, etc.)
            content_type: Tipo de contenido (educational, entertainment, etc.)
            duration: Duración del video en segundos
            timing: Momento del CTA (early, middle, end)
            cta_type: Tipo de CTA (follow, like, comment, etc.)
            audience_data: Datos de audiencia para personalización
            character_data: Datos del personaje para adaptar el tono
            
        Returns:
            Dict: CTA generado con texto, tiempo y metadatos
        """
        logger.info(f"Generando CTA para {platform} en timing {timing}")
        
        # Normalizar plataforma
        if platform == 'reels':
            platform = 'instagram'
        elif platform == 'shorts':
            platform = 'youtube_shorts'
        
        # Si no se especifica tipo de CTA, seleccionar uno apropiado para la plataforma
        if not cta_type:
            available_types = self.platform_cta_types.get(platform, ['follow', 'like', 'comment'])
            
            # Priorizar tipos según el timing
            if timing == 'early':
                priority_types = ['follow', 'subscribe', 'notification']
            elif timing == 'middle':
                priority_types = ['comment', 'like', 'gamified']
            else:  # end
                priority_types = ['profile', 'share', 'product', 'affiliate']
            
            # Filtrar tipos disponibles según prioridad
            priority_available = [t for t in priority_types if t in available_types]
            if priority_available:
                cta_type = random.choice(priority_available)
            else:
                cta_type = random.choice(available_types)
        
        # Determinar el tiempo exacto para el CTA
        if timing == 'end' and self.cta_timing[timing]['start'] < 0:
            # Para CTAs al final, calculamos desde el final del video
            start_time = max(0, duration + self.cta_timing[timing]['start'])
            end_time = duration
        else:
            # Para otros CTAs, usamos los tiempos absolutos
            start_time = self.cta_timing[timing]['start']
            end_time = min(duration, self.cta_timing[timing]['end'])
        
        # Si el video es muy corto, ajustar tiempos
        if duration < 10:
            if timing == 'middle':
                start_time = max(3, int(duration * 0.3))
                end_time = min(start_time + 2, duration - 1)
            elif timing == 'end':
                start_time = max(0, duration - 3)
                end_time = duration
        
        # Seleccionar una plantilla para el tipo y timing
        templates = self.templates.get(cta_type, {}).get(timing, [])
        if not templates:
            # Si no hay plantillas específicas, usar plantillas genéricas
            templates = [
                f"¡{cta_type.capitalize()} ahora para más contenido!",
                f"¡No olvides {cta_type} si te ha gustado!",
                f"¡{cta_type.capitalize()} para apoyar este contenido!"
            ]
        
        template = random.choice(templates)
        
        # Preparar variables para la plantilla
        variables = {
            "content_type": content_type,
            "topic": niche,
            "emoji": self._get_emoji_for_niche(niche),
            "frequency": "semana",
            "benefit": self._get_benefit_for_niche(niche),
            "related_topic": self._get_related_topic(niche),
            "keyword": niche.upper(),
            "code": f"{niche.upper()}10",
            "discount": str(random.randint(10, 30)),
            "product": self._get_product_for_niche(niche),
            "product_type": self._get_product_type_for_niche(niche),
            "number": str(random.randint(3, 10)),
            "prize": self._get_prize_for_niche(niche),
            "timeframe": f"{random.randint(1, 24)} horas",
            "exclusive_content": self._get_exclusive_content(niche),
            "feature_1": f"guía de {niche}",
            "feature_2": "tutoriales exclusivos"
        }
        
        # Añadir variables de audiencia si están disponibles
        if audience_data:
            variables.update({
                "age_range": f"{audience_data.get('age_range', [18, 35])[0]}-{audience_data.get('age_range', [18, 35])[1]}",
                "gender": audience_data.get('gender_skew', 'balanced'),
                "interests": ", ".join(audience_data.get('interests', ['general'])[:2])
            })
        
        # Añadir variables de personaje si están disponibles
        if character_data:
            variables.update({
                "character_name": character_data.get('name', ''),
                "character_catchphrase": character_data.get('catchphrase', ''),
                "character_style": character_data.get('style', 'casual')
            })
            
            # Adaptar el CTA al estilo del personaje
            if 'cta_style' in character_data:
                if character_data['cta_style'] == 'educational' and '!' in template:
                    template = template.replace('!', '.')
                elif character_data['cta_style'] == 'motivational' and '!' not in template:
                    template = template.replace('.', '!')
        
        # Rellenar la plantilla con las variables
        cta_text = self._fill_template(template, variables)
        
        # Calcular tiempo exacto para el CTA
        cta_time = random.randint(start_time, end_time) if start_time < end_time else start_time
        
        # Crear objeto CTA
        cta_id = f"cta_{int(datetime.datetime.now().timestamp())}_{platform}_{timing}"
        cta = {
            "id": cta_id,
            "text": cta_text,
            "platform": platform,
            "niche": niche,
            "content_type": content_type,
            "timing": timing,
            "cta_type": cta_type,
            "time": cta_time,
            "duration": min(5, duration - cta_time) if cta_time < duration else 2,
            "variables": variables,
            "template": template,
            "reputation": 50,  # Reputación inicial neutral
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Guardar CTA en la base de conocimiento
        self.kb.save_cta(cta)
        
        logger.info(f"CTA generado: {cta_text} (tiempo: {cta_time}s)")
        return cta
    
    def generate_multiple_ctas(self, 
                              platform: str, 
                              niche: str, 
                              content_type: str, 
                              duration: int,
                              count: int = 3,
                              audience_data: Dict = None,
                              character_data: Dict = None) -> List[Dict]:
        """
        Genera múltiples CTAs para diferentes momentos del video
        
        Args:
            platform: Plataforma de destino
            niche: Nicho de contenido
            content_type: Tipo de contenido
            duration: Duración del video en segundos
            count: Número de CTAs a generar
            audience_data: Datos de audiencia para personalización
            character_data: Datos del personaje para adaptar el tono
            
        Returns:
            List[Dict]: Lista de CTAs generados
        """
        ctas = []
        
        # Determinar los timings según la duración
        if duration <= 15:
            # Para videos muy cortos, solo un CTA en el medio
            timings = ['middle']
        elif duration <= 30:
            # Para videos cortos, CTA al inicio y al final
            timings = ['early', 'end']
        elif duration <= 60:
            # Para videos de hasta 1 minuto, CTAs al inicio, medio y final
            timings = ['early', 'middle', 'end']
        else:
            # Para videos largos, múltiples CTAs
            timings = ['early', 'middle', 'middle', 'end']
            # Añadir CTAs adicionales para videos muy largos
            if duration > 180 and count > 4:
                extra_count = min(count - 4, duration // 60 - 2)
                timings.extend(['middle'] * extra_count)
        
        # Limitar al número solicitado
        timings = timings[:count]
        
        # Generar CTAs para cada timing
        for timing in timings:
            # Seleccionar tipo de CTA apropiado para el timing
            if timing == 'early':
                cta_types = ['follow', 'subscribe'] if platform in ['youtube', 'youtube_shorts'] else ['follow']
            elif timing == 'middle':
                if random.random() < 0.3:  # 30% de probabilidad de CTA gamificado
                    cta_types = ['gamified']
                else:
                    cta_types = ['comment', 'like']
            else:  # end
                if random.random() < 0.2 and duration > 30:  # 20% de probabilidad de CTA de producto/afiliado
                    cta_types = ['affiliate', 'product']
                else:
                    cta_types = ['profile', 'share']
            
            cta_type = random.choice(cta_types)
            
            # Generar CTA
            cta = self.generate_cta(
                platform=platform,
                niche=niche,
                content_type=content_type,
                duration=duration,
                timing=timing,
                cta_type=cta_type,
                audience_data=audience_data,
                character_data=character_data
            )
            
            ctas.append(cta)
        
        # Asegurar que los CTAs no se superpongan en tiempo
        ctas.sort(key=lambda x: x['time'])
        for i in range(1, len(ctas)):
            if ctas[i]['time'] - ctas[i-1]['time'] < ctas[i-1]['duration']:
                ctas[i]['time'] = ctas[i-1]['time'] + ctas[i-1]['duration'] + 1
        
        return ctas
    
    def get_best_performing_ctas(self, platform: str, niche: str, count: int = 5) -> List[Dict]:
        """
        Obtiene los CTAs con mejor rendimiento para una plataforma y nicho
        
        Args:
            platform: Plataforma de destino
            niche: Nicho de contenido
            count: Número de CTAs a obtener
            
        Returns:
            List[Dict]: Lista de CTAs con mejor rendimiento
        """
        # Obtener CTAs de la base de conocimiento
        ctas = self.kb.get_ctas_by_criteria({
            'platform': platform,
            'niche': niche
        })
        
        # Ordenar por reputación (rendimiento)
        ctas.sort(key=lambda x: x.get('reputation', 0), reverse=True)
        
        return ctas[:count]
    
    def update_cta_reputation(self, cta_id: str, performance_data: Dict) -> bool:
        """
        Actualiza la reputación de un CTA basado en su rendimiento
        
        Args:
            cta_id: ID del CTA
            performance_data: Datos de rendimiento (engagement, conversión, etc.)
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario
        """
        # Obtener CTA actual
        cta = self.kb.get_cta(cta_id)
        if not cta:
            logger.error(f"No se encontró CTA con ID: {cta_id}")
            return False
        
        # Calcular nueva reputación basada en métricas de rendimiento
        current_reputation = cta.get('reputation', 50)
        
        # Factores que afectan la reputación
        engagement_factor = performance_data.get('engagement_rate', 0) * 100  # 0-100%
        conversion_factor = performance_data.get('conversion_rate', 0) * 100  # 0-100%
        retention_factor = performance_data.get('retention_after_cta', 0) * 100  # 0-100%
        
        # Pesos para cada factor
        weights = {
            'engagement': 0.3,
            'conversion': 0.5,
            'retention': 0.2
        }
        
        # Calcular nueva reputación
        new_reputation = (
            current_reputation * 0.7 +  # 70% reputación actual
            (engagement_factor * weights['engagement'] +
             conversion_factor * weights['conversion'] +
             retention_factor * weights['retention']) * 0.3  # 30% nuevo rendimiento
        )
        
        # Limitar a 0-100
        new_reputation = max(0, min(100, new_reputation))
        
        # Actualizar CTA
        cta['reputation'] = new_reputation
        cta['last_performance'] = performance_data
        cta['updated_at'] = datetime.datetime.now().isoformat()
        
        # Guardar CTA actualizado
        success = self.kb.save_cta(cta)
        
        if success:
            logger.info(f"Reputación de CTA {cta_id} actualizada: {current_reputation} -> {new_reputation}")
        
        return success
    
    def save_custom_template(self, cta_type: str, timing: str, template: str) -> bool:
        """
        Guarda una plantilla personalizada de CTA
        
        Args:
            cta_type: Tipo de CTA (follow, like, comment, etc.)
            timing: Momento del CTA (early, middle, end)
            template: Texto de la plantilla
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        # Verificar que el tipo y timing sean válidos
        if cta_type not in self.templates:
            self.templates[cta_type] = {'early': [], 'middle': [], 'end': []}
        
        if timing not in self.templates[cta_type]:
            self.templates[cta_type][timing] = []
        
        # Añadir plantilla si no existe
        if template not in self.templates[cta_type][timing]:
            self.templates[cta_type][timing].append(template)
            
            # Guardar plantillas actualizadas
            try:
                with open(self.templates_file, 'w', encoding='utf-8') as f:
                    json.dump(self.templates, f, indent=4, ensure_ascii=False)
                logger.info(f"Plantilla personalizada guardada: {cta_type}/{timing}")
                return True
            except Exception as e:
                logger.error(f"Error al guardar plantilla personalizada: {str(e)}")
                return False
        
        return True  # La plantilla ya existía
    
    def _fill_template(self, template: str, variables: Dict[str, str]) -> str:
        """
        Rellena una plantilla con variables
        
        Args:
            template: Plantilla con marcadores {variable}
            variables: Diccionario de variables para sustituir
            
        Returns:
            str: Plantilla con variables sustituidas
        """
        result = template
        
        # Sustituir variables conocidas
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, value)
        
        # Identificar variables faltantes
        import re
        missing_vars = re.findall(r'\{([^}]+)\}', result)
        
        # Generar valores para variables faltantes
        for var in missing_vars:
                        # Generar un valor genérico basado en el nombre de la variable
            if 'emoji' in var:
                emojis = ["🔥", "💯", "👍", "✅", "⭐", "🚀", "💪", "👏", "🙌", "😍"]
                value = random.choice(emojis)
            elif 'benefit' in var:
                benefits = ["mejorar tus habilidades", "aprender algo nuevo", "estar al día", 
                           "optimizar tus resultados", "descubrir secretos", "dominar el tema"]
                value = random.choice(benefits)
            elif 'related_topic' in var:
                related = [f"otro aspecto de {niche}", f"consejos avanzados", 
                          f"errores comunes en {niche}", f"novedades en {niche}"]
                value = random.choice(related)
            elif 'product' in var:
                products = ["este producto", "esta herramienta", "este curso", 
                           "esta guía", "este recurso", "esta solución"]
                value = random.choice(products)
            elif 'product_type' in var:
                types = ["curso", "guía", "ebook", "membresía", "herramienta", "plantilla"]
                value = random.choice(types)
            elif 'prize' in var:
                prizes = ["una consulta gratis", "acceso a contenido exclusivo", 
                         "un descuento especial", "una mención en mi próximo video"]
                value = random.choice(prizes)
            elif 'exclusive_content' in var:
                content = ["tutoriales avanzados", "sesiones en vivo", 
                          "recursos exclusivos", "acceso anticipado"]
                value = random.choice(content)
            elif 'frequency' in var:
                value = random.choice(["semana", "lunes", "día"])
            else:
                value = f"[{var}]"  # Marcar como pendiente
            
            result = result.replace("{" + var + "}", value)
        
        return result
    
    def _get_emoji_for_niche(self, niche: str) -> str:
        """Devuelve un emoji apropiado para el nicho"""
        emoji_mapping = {
            "finance": ["💰", "💸", "📈", "💹", "🏦"],
            "crypto": ["🪙", "📊", "🚀", "💎", "🔐"],
            "health": ["💪", "🥗", "🧘", "❤️", "🏃"],
            "fitness": ["🏋️", "🤸", "🏃", "💪", "🥇"],
            "technology": ["💻", "🔌", "📱", "🤖", "⚙️"],
            "gaming": ["🎮", "🕹️", "🎯", "🏆", "👾"],
            "education": ["📚", "🎓", "✏️", "🧠", "📝"],
            "beauty": ["💄", "💅", "👗", "✨", "💫"],
            "food": ["🍔", "🍕", "🍰", "🍗", "🥗"],
            "travel": ["✈️", "🏝️", "🧳", "🗺️", "🏞️"],
            "fashion": ["👔", "👗", "👠", "👜", "🕶️"],
            "music": ["🎵", "🎸", "🎹", "🎤", "🎧"],
            "art": ["🎨", "🖌️", "🖼️", "✏️", "🎭"],
            "business": ["💼", "📊", "📈", "🤝", "💡"],
            "mindfulness": ["🧘", "🌿", "🧠", "✨", "🌈"],
            "humor": ["😂", "🤣", "😆", "😜", "🎭"]
        }
        
        # Buscar emojis para el nicho específico
        if niche in emoji_mapping:
            return random.choice(emoji_mapping[niche])
        
        # Si no se encuentra, usar emojis genéricos
        generic_emojis = ["✅", "⭐", "🔥", "💯", "👍"]
        return random.choice(generic_emojis)
    
    def _get_benefit_for_niche(self, niche: str) -> str:
        """Devuelve un beneficio apropiado para el nicho"""
        benefit_mapping = {
            "finance": ["mejorar tus finanzas", "aumentar tus ingresos", "optimizar tus inversiones", 
                       "ahorrar más dinero", "alcanzar libertad financiera"],
            "crypto": ["maximizar tus ganancias", "evitar errores comunes", "identificar oportunidades", 
                      "entender el mercado crypto", "diversificar tu portafolio"],
            "health": ["mejorar tu salud", "aumentar tu energía", "optimizar tu bienestar", 
                      "sentirte mejor cada día", "transformar tu cuerpo"],
            "fitness": ["conseguir resultados más rápido", "optimizar tu entrenamiento", 
                       "desarrollar músculo", "perder grasa", "mejorar tu rendimiento"],
            "technology": ["dominar la tecnología", "estar al día con las novedades", 
                          "optimizar tus dispositivos", "resolver problemas técnicos", "automatizar tareas"],
            "gaming": ["mejorar tus habilidades", "dominar estrategias avanzadas", 
                      "conseguir más victorias", "descubrir secretos", "subir de nivel más rápido"],
            "education": ["aprender más rápido", "dominar nuevas habilidades", 
                         "mejorar tus resultados", "optimizar tu estudio", "destacar académicamente"],
            "beauty": ["lucir tu mejor versión", "dominar técnicas profesionales", 
                      "conocer los mejores productos", "ahorrar en tu rutina", "transformar tu imagen"],
            "business": ["hacer crecer tu negocio", "aumentar tus ventas", 
                        "optimizar tus procesos", "liderar con éxito", "destacar en tu mercado"]
        }
        
        # Buscar beneficios para el nicho específico
        if niche in benefit_mapping:
            return random.choice(benefit_mapping[niche])
        
        # Si no se encuentra, usar beneficios genéricos
        generic_benefits = ["mejorar tus resultados", "aprender algo nuevo cada día", 
                           "estar al día con las últimas tendencias", "optimizar tu experiencia", 
                           "descubrir contenido exclusivo"]
        return random.choice(generic_benefits)
    
    def _get_related_topic(self, niche: str) -> str:
        """Devuelve un tema relacionado con el nicho"""
        related_mapping = {
            "finance": ["inversiones", "ahorro", "presupuesto", "deudas", "ingresos pasivos"],
            "crypto": ["bitcoin", "ethereum", "trading", "NFTs", "DeFi"],
            "health": ["nutrición", "ejercicio", "sueño", "estrés", "suplementos"],
            "fitness": ["entrenamiento de fuerza", "cardio", "nutrición deportiva", "recuperación", "rutinas"],
            "technology": ["inteligencia artificial", "programación", "gadgets", "apps", "seguridad"],
            "gaming": ["estrategias", "nuevos lanzamientos", "configuraciones", "competitivo", "trucos"],
            "education": ["técnicas de estudio", "productividad", "idiomas", "certificaciones", "carrera"],
            "beauty": ["skincare", "maquillaje", "cabello", "uñas", "tratamientos"],
            "business": ["marketing", "ventas", "liderazgo", "productividad", "emprendimiento"]
        }
        
        # Buscar temas relacionados para el nicho específico
        if niche in related_mapping:
            return random.choice(related_mapping[niche])
        
        # Si no se encuentra, usar temas genéricos
        return f"temas relacionados con {niche}"
    
    def _get_product_for_niche(self, niche: str) -> str:
        """Devuelve un producto apropiado para el nicho"""
        product_mapping = {
            "finance": ["curso de inversiones", "calculadora financiera", "plantilla de presupuesto", 
                       "ebook de finanzas personales", "consultoría financiera"],
            "crypto": ["curso de trading", "señales premium", "guía de inversión en crypto", 
                      "herramienta de análisis", "membresía VIP"],
            "health": ["plan de nutrición", "guía de suplementos", "programa de bienestar", 
                      "app de seguimiento", "consulta personalizada"],
            "fitness": ["programa de entrenamiento", "plan de nutrición", "suplementos", 
                       "equipamiento fitness", "coaching personalizado"],
            "technology": ["curso de programación", "herramienta de productividad", 
                          "guía de optimización", "software premium", "consultoría tech"],
            "gaming": ["guía de estrategias", "coaching gaming", "equipamiento pro", 
                      "acceso a servidor VIP", "contenido exclusivo"],
            "education": ["curso completo", "material de estudio", "mentoría personalizada", 
                         "recursos premium", "certificación"],
            "beauty": ["kit de productos", "tutorial avanzado", "consulta personalizada", 
                      "herramientas profesionales", "rutina completa"]
        }
        
        # Buscar productos para el nicho específico
        if niche in product_mapping:
            return random.choice(product_mapping[niche])
        
        # Si no se encuentra, usar productos genéricos
        generic_products = ["curso completo", "guía definitiva", "consultoría personalizada", 
                           "herramienta especializada", "contenido premium"]
        return random.choice(generic_products)
    
    def _get_product_type_for_niche(self, niche: str) -> str:
        """Devuelve un tipo de producto apropiado para el nicho"""
        # Tipos de productos comunes por nicho
        if niche in ["finance", "crypto", "business"]:
            types = ["curso", "mentoría", "membresía", "ebook", "calculadora", "plantilla"]
        elif niche in ["health", "fitness"]:
            types = ["programa", "plan", "guía", "consultoría", "app", "suplemento"]
        elif niche in ["technology", "gaming"]:
            types = ["curso", "herramienta", "guía", "software", "membresía", "coaching"]
        elif niche in ["education"]:
            types = ["curso", "material", "mentoría", "certificación", "programa", "recursos"]
        elif niche in ["beauty", "fashion"]:
            types = ["tutorial", "kit", "guía", "consultoría", "colección", "rutina"]
        else:
            types = ["curso", "guía", "ebook", "membresía", "consultoría", "herramienta"]
        
        return random.choice(types)
    
    def _get_prize_for_niche(self, niche: str) -> str:
        """Devuelve un premio apropiado para el nicho"""
        prize_mapping = {
            "finance": ["una consultoría financiera gratuita", "acceso a mi curso de inversiones", 
                       "mi plantilla de presupuesto premium", "una sesión de planificación financiera"],
            "crypto": ["señales de trading por 1 mes", "mi análisis de portafolio", 
                      "acceso a mi grupo VIP", "mi guía de inversión en altcoins"],
            "health": ["una consulta personalizada", "mi plan de bienestar", 
                      "acceso a mi programa completo", "una rutina personalizada"],
            "fitness": ["un plan de entrenamiento personalizado", "una consulta nutricional", 
                       "acceso a mi programa premium", "una sesión de coaching"],
            "technology": ["una consultoría tech", "acceso a mi curso", 
                          "una revisión de tu setup", "mis recursos premium"],
            "gaming": ["una sesión de coaching", "acceso a mi servidor privado", 
                      "mi guía de estrategias avanzadas", "equipamiento gaming"],
            "education": ["una sesión de tutoría", "acceso a mi curso completo", 
                         "mis materiales premium", "una revisión personalizada"],
            "beauty": ["una consulta de imagen", "mi kit de productos recomendados", 
                      "acceso a mis tutoriales premium", "una rutina personalizada"]
        }
        
        # Buscar premios para el nicho específico
        if niche in prize_mapping:
            return random.choice(prize_mapping[niche])
        
        # Si no se encuentra, usar premios genéricos
        generic_prizes = ["acceso a mi contenido exclusivo", "una consulta personalizada", 
                         "mi guía completa", "un descuento especial", "una mención en mi próximo video"]
        return random.choice(generic_prizes)
    
    def _get_exclusive_content(self, niche: str) -> str:
        """Devuelve contenido exclusivo apropiado para el nicho"""
        content_mapping = {
            "finance": ["análisis de mercado semanales", "alertas de inversión", 
                       "webinars exclusivos", "plantillas premium", "consultas personalizadas"],
            "crypto": ["señales de trading", "análisis de proyectos", 
                      "alertas de mercado", "estrategias avanzadas", "acceso a ICOs"],
            "health": ["planes personalizados", "seguimiento de progreso", 
                      "recetas exclusivas", "sesiones en vivo", "consultas privadas"],
            "fitness": ["rutinas avanzadas", "planes nutricionales", 
                       "técnicas profesionales", "seguimiento personalizado", "sesiones en vivo"],
            "technology": ["tutoriales avanzados", "código fuente", 
                          "herramientas premium", "soporte prioritario", "proyectos completos"],
            "gaming": ["estrategias pro", "sesiones de juego privadas", 
                      "análisis de partidas", "acceso a servidor VIP", "contenido anticipado"],
            "education": ["material de estudio avanzado", "correcciones personalizadas", 
                         "clases privadas", "recursos premium", "certificaciones"],
            "business": ["plantillas de negocio", "análisis de mercado", 
                        "estrategias de crecimiento", "mentoría personalizada", "networking exclusivo"]
        }
        
        # Buscar contenido exclusivo para el nicho específico
        if niche in content_mapping:
            return random.choice(content_mapping[niche])
        
        # Si no se encuentra, usar contenido genérico
        generic_content = ["tutoriales avanzados", "contenido anticipado", 
                          "sesiones en vivo", "recursos premium", "soporte personalizado"]
        return random.choice(generic_content)


# Punto de entrada para pruebas
if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs('logs', exist_ok=True)
    os.makedirs(os.path.join('creation', 'narrative'), exist_ok=True)
    
    # Inicializar generador de CTAs
    cta_generator = CTAGenerator()
    
    # Generar un CTA para TikTok sobre finanzas
    cta = cta_generator.generate_cta(
        platform="tiktok",
        niche="finance",
        content_type="consejos financieros",
        duration=30,
        timing="middle"
    )
    print("CTA generado:")
    print(json.dumps(cta, indent=2, ensure_ascii=False))
    
    # Generar múltiples CTAs para un video de YouTube
    ctas = cta_generator.generate_multiple_ctas(
        platform="youtube",
        niche="technology",
        content_type="tutorial",
        duration=300,
        count=4
    )
    print("\nMúltiples CTAs generados:")
    for i, cta in enumerate(ctas):
        print(f"\n{i+1}. {cta['text']} (tiempo: {cta['time']}s)")
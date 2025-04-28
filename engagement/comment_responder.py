import os
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/engagement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CommentResponder")

class CommentResponder:
    """
    Sistema para responder automáticamente a comentarios en diferentes plataformas.
    Utiliza plantillas personalizables, análisis de sentimiento y priorización
    para mantener engagement con la audiencia.
    """
    
    def __init__(self, config_path: str = "config/engagement_config.json"):
        """
        Inicializa el sistema de respuesta a comentarios.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.templates = {}
        self.response_history = []
        self.user_interaction_history = {}
        self.blacklisted_phrases = []
        self.priority_keywords = []
        
        # Crear directorio de logs si no existe
        os.makedirs("logs", exist_ok=True)
        
        # Cargar configuración
        self._load_config()
        
        logger.info("Sistema de respuesta a comentarios inicializado")
    
    def _load_config(self) -> None:
        """Carga la configuración desde el archivo JSON."""
        try:
            # Crear directorio de configuración si no existe
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Verificar si existe el archivo de configuración
            if not os.path.exists(self.config_path):
                # Crear configuración por defecto
                default_config = {
                    "templates": {
                        "positive": [
                            "¡Gracias por tu comentario! 😊 ¿Te gustaría ver más contenido como este?",
                            "¡Me alegra que te haya gustado! 🙌 Sigue para más contenido similar.",
                            "¡Aprecio mucho tu apoyo! 💯 ¿Qué otro tema te gustaría ver?"
                        ],
                        "negative": [
                            "Gracias por compartir tu opinión. ¿Hay algo específico que podamos mejorar?",
                            "Valoramos tu feedback. ¿Qué tipo de contenido prefieres?",
                            "Lamentamos que no haya cumplido tus expectativas. ¿Qué te gustaría ver en el futuro?"
                        ],
                        "question": [
                            "¡Buena pregunta! 🤔 {answer}",
                            "Excelente pregunta. {answer} ¿Hay algo más que quieras saber?",
                            "¡Gracias por preguntar! {answer} Déjame saber si tienes más dudas."
                        ],
                        "generic": [
                            "¡Gracias por comentar! 👍 ¿Qué te pareció el video?",
                            "¡Apreciamos tu interacción! ¿Qué otro contenido te gustaría ver?",
                            "¡Gracias por pasar por aquí! 🙏 No olvides seguirnos para más."
                        ]
                    },
                    "blacklisted_phrases": [
                        "spam", "publicidad", "estafa", "fake", "clickbait"
                    ],
                    "priority_keywords": [
                        "pregunta", "duda", "cómo", "ayuda", "información"
                    ],
                    "response_rate": 0.8,  # Responder al 80% de los comentarios
                    "max_daily_responses": 100,
                    "response_delay": {
                        "min_minutes": 5,
                        "max_minutes": 120
                    }
                }
                
                # Guardar configuración por defecto
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4)
                
                self.templates = default_config["templates"]
                self.blacklisted_phrases = default_config["blacklisted_phrases"]
                self.priority_keywords = default_config["priority_keywords"]
                self.response_rate = default_config["response_rate"]
                self.max_daily_responses = default_config["max_daily_responses"]
                self.response_delay = default_config["response_delay"]
                
                logger.info("Configuración por defecto creada")
            else:
                # Cargar configuración existente
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                self.templates = config.get("templates", {})
                self.blacklisted_phrases = config.get("blacklisted_phrases", [])
                self.priority_keywords = config.get("priority_keywords", [])
                self.response_rate = config.get("response_rate", 0.8)
                self.max_daily_responses = config.get("max_daily_responses", 100)
                self.response_delay = config.get("response_delay", {"min_minutes": 5, "max_minutes": 120})
                
                logger.info("Configuración cargada correctamente")
        
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {str(e)}")
            # Usar valores por defecto
            self.templates = {
                "positive": ["¡Gracias por tu comentario! 😊"],
                "negative": ["Gracias por compartir tu opinión."],
                "question": ["¡Buena pregunta! 🤔 {answer}"],
                "generic": ["¡Gracias por comentar! 👍"]
            }
            self.blacklisted_phrases = ["spam", "publicidad"]
            self.priority_keywords = ["pregunta", "duda"]
            self.response_rate = 0.8
            self.max_daily_responses = 100
            self.response_delay = {"min_minutes": 5, "max_minutes": 120}
    
    def _load_response_history(self) -> None:
        """Carga el historial de respuestas desde el archivo JSON."""
        history_path = "data/engagement/response_history.json"
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    self.response_history = json.load(f)
                logger.info(f"Historial de respuestas cargado: {len(self.response_history)} entradas")
            else:
                self.response_history = []
                logger.info("No se encontró historial de respuestas previo")
        
        except Exception as e:
            logger.error(f"Error al cargar historial de respuestas: {str(e)}")
            self.response_history = []
    
    def _save_response_history(self) -> None:
        """Guarda el historial de respuestas en un archivo JSON."""
        history_path = "data/engagement/response_history.json"
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.response_history, f, indent=4)
            
            logger.info(f"Historial de respuestas guardado: {len(self.response_history)} entradas")
        
        except Exception as e:
            logger.error(f"Error al guardar historial de respuestas: {str(e)}")
    
    def _load_user_interaction_history(self) -> None:
        """Carga el historial de interacciones de usuarios desde el archivo JSON."""
        history_path = "data/engagement/user_interaction_history.json"
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    self.user_interaction_history = json.load(f)
                logger.info(f"Historial de interacciones cargado: {len(self.user_interaction_history)} usuarios")
            else:
                self.user_interaction_history = {}
                logger.info("No se encontró historial de interacciones previo")
        
        except Exception as e:
            logger.error(f"Error al cargar historial de interacciones: {str(e)}")
            self.user_interaction_history = {}
    
    def _save_user_interaction_history(self) -> None:
        """Guarda el historial de interacciones de usuarios en un archivo JSON."""
        history_path = "data/engagement/user_interaction_history.json"
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.user_interaction_history, f, indent=4)
            
            logger.info(f"Historial de interacciones guardado: {len(self.user_interaction_history)} usuarios")
        
        except Exception as e:
            logger.error(f"Error al guardar historial de interacciones: {str(e)}")
    
    def should_respond(self, comment: Dict[str, Any]) -> bool:
        """
        Determina si se debe responder a un comentario basado en la configuración
        y el análisis del comentario.
        
        Args:
            comment: Datos del comentario
            
        Returns:
            True si se debe responder, False en caso contrario
        """
        # Verificar si ya se ha respondido al comentario
        if any(response["comment_id"] == comment["comment_id"] for response in self.response_history):
            return False
        
        # Verificar si contiene frases en la lista negra
        comment_text = comment.get("text", "").lower()
        if any(phrase in comment_text for phrase in self.blacklisted_phrases):
            logger.info(f"Comentario contiene frase en lista negra: {comment['comment_id']}")
            return False
        
        # Verificar si contiene palabras clave prioritarias
        has_priority = any(keyword in comment_text for keyword in self.priority_keywords)
        
        # Aplicar tasa de respuesta si no es prioritario
        if not has_priority and random.random() > self.response_rate:
            return False
        
        # Verificar límite diario de respuestas
        today = datetime.now().strftime("%Y-%m-%d")
        responses_today = sum(1 for response in self.response_history 
                             if response.get("response_date", "").startswith(today))
        
        if responses_today >= self.max_daily_responses:
            logger.info(f"Límite diario de respuestas alcanzado: {self.max_daily_responses}")
            return False
        
        return True
    
    def classify_comment(self, comment: Dict[str, Any]) -> str:
        """
        Clasifica un comentario en una categoría para seleccionar la plantilla adecuada.
        
        Args:
            comment: Datos del comentario
            
        Returns:
            Categoría del comentario (positive, negative, question, generic)
        """
        # Obtener texto del comentario
        comment_text = comment.get("text", "").lower()
        
        # Verificar si es una pregunta
        if "?" in comment_text or any(q in comment_text for q in ["cómo", "qué", "cuándo", "dónde", "por qué", "quién"]):
            return "question"
        
        # Verificar sentimiento (simplificado, idealmente usar sentiment_analyzer.py)
        positive_words = ["gracias", "genial", "excelente", "bueno", "me gusta", "increíble", "wow", "amor", "❤️", "👍", "😍"]
        negative_words = ["malo", "terrible", "odio", "no me gusta", "aburrido", "decepcionado", "👎", "😡", "🤮"]
        
        positive_count = sum(1 for word in positive_words if word in comment_text)
        negative_count = sum(1 for word in negative_words if word in comment_text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "generic"
    
    def generate_response(self, comment: Dict[str, Any], sentiment_data: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Genera una respuesta personalizada para un comentario.
        
        Args:
            comment: Datos del comentario
            sentiment_data: Datos de análisis de sentimiento (opcional)
            
        Returns:
            Respuesta generada y metadatos
        """
        # Clasificar comentario
        category = self.classify_comment(comment)
        
        # Usar datos de sentimiento si están disponibles
        if sentiment_data and "sentiment" in sentiment_data:
            if sentiment_data["sentiment"] == "positive":
                category = "positive"
            elif sentiment_data["sentiment"] == "negative":
                category = "negative"
        
        # Obtener plantillas para la categoría
        templates = self.templates.get(category, self.templates.get("generic", ["¡Gracias por comentar!"]))
        
        # Seleccionar plantilla aleatoria
        template = random.choice(templates)
        
        # Personalizar respuesta
        response = template
        
        # Reemplazar variables en la plantilla
        if "{username}" in response:
            response = response.replace("{username}", comment.get("username", ""))
        
        if "{answer}" in response and category == "question":
            # Aquí se podría integrar con un sistema de respuestas automáticas
            # Por ahora, usamos una respuesta genérica
            response = response.replace("{answer}", "Pronto responderemos a tu pregunta.")
        
        # Registrar usuario en historial de interacciones
        user_id = comment.get("user_id", "unknown")
        if user_id not in self.user_interaction_history:
            self.user_interaction_history[user_id] = {
                "first_interaction": datetime.now().isoformat(),
                "interaction_count": 0,
                "last_interaction": None,
                "sentiment_history": []
            }
        
        # Actualizar historial de interacciones
        self.user_interaction_history[user_id]["interaction_count"] += 1
        self.user_interaction_history[user_id]["last_interaction"] = datetime.now().isoformat()
        
        if sentiment_data and "sentiment" in sentiment_data:
            self.user_interaction_history[user_id]["sentiment_history"].append({
                "date": datetime.now().isoformat(),
                "sentiment": sentiment_data["sentiment"],
                "score": sentiment_data.get("score", 0)
            })
        
        # Guardar historial de interacciones
        self._save_user_interaction_history()
        
        # Crear metadatos de respuesta
        response_metadata = {
            "comment_id": comment.get("comment_id", ""),
            "user_id": user_id,
            "content_id": comment.get("content_id", ""),
            "platform": comment.get("platform", ""),
            "category": category,
            "template_used": template,
            "response_date": datetime.now().isoformat(),
            "scheduled_time": None  # Se establecerá al programar la respuesta
        }
        
        return response, response_metadata
    
    def schedule_response(self, response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Programa una respuesta para ser enviada con un retraso natural.
        
        Args:
            response: Texto de la respuesta
            metadata: Metadatos de la respuesta
            
        Returns:
            Datos de la respuesta programada
        """
        # Calcular tiempo de retraso aleatorio
        min_minutes = self.response_delay.get("min_minutes", 5)
        max_minutes = self.response_delay.get("max_minutes", 120)
        delay_minutes = random.randint(min_minutes, max_minutes)
        
        # Calcular tiempo programado
        now = datetime.now()
        scheduled_time = now.timestamp() + (delay_minutes * 60)
        scheduled_time_iso = datetime.fromtimestamp(scheduled_time).isoformat()
        
        # Actualizar metadatos
        metadata["scheduled_time"] = scheduled_time_iso
        
        # Crear entrada de respuesta
        response_entry = {
            "response_id": f"resp_{len(self.response_history) + 1}",
            "response_text": response,
            "comment_id": metadata["comment_id"],
            "user_id": metadata["user_id"],
            "content_id": metadata["content_id"],
            "platform": metadata["platform"],
            "category": metadata["category"],
            "created_at": datetime.now().isoformat(),
            "scheduled_time": scheduled_time_iso,
            "sent_at": None,
            "status": "scheduled"
        }
        
        # Añadir a historial
        self.response_history.append(response_entry)
        self._save_response_history()
        
        logger.info(f"Respuesta programada para {scheduled_time_iso} ({delay_minutes} minutos de retraso)")
        
        return response_entry
    
    def process_comment(self, comment: Dict[str, Any], sentiment_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Procesa un comentario y genera una respuesta si es apropiado.
        
        Args:
            comment: Datos del comentario
            sentiment_data: Datos de análisis de sentimiento (opcional)
            
        Returns:
            Datos de la respuesta programada o None si no se debe responder
        """
        # Verificar si se debe responder
        if not self.should_respond(comment):
            return None
        
        # Generar respuesta
        response_text, response_metadata = self.generate_response(comment, sentiment_data)
        
        # Programar respuesta
        response_entry = self.schedule_response(response_text, response_metadata)
        
        return response_entry
    
    def process_comments_batch(self, comments: List[Dict[str, Any]], sentiment_data: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Procesa un lote de comentarios y genera respuestas.
        
        Args:
            comments: Lista de comentarios
            sentiment_data: Diccionario de datos de sentimiento por comment_id (opcional)
            
        Returns:
            Lista de respuestas programadas
        """
        responses = []
        
        for comment in comments:
            comment_id = comment.get("comment_id")
            comment_sentiment = sentiment_data.get(comment_id) if sentiment_data else None
            
            response = self.process_comment(comment, comment_sentiment)
            if response:
                responses.append(response)
        
        return responses
    
    def get_pending_responses(self) -> List[Dict[str, Any]]:
        """
        Obtiene las respuestas pendientes que están programadas para ser enviadas.
        
        Returns:
            Lista de respuestas pendientes
        """
        now = datetime.now().timestamp()
        
        pending_responses = []
        for response in self.response_history:
            if response["status"] == "scheduled" and response["scheduled_time"]:
                scheduled_time = datetime.fromisoformat(response["scheduled_time"]).timestamp()
                if scheduled_time <= now:
                    pending_responses.append(response)
        
        return pending_responses
    
    def mark_response_as_sent(self, response_id: str) -> bool:
        """
        Marca una respuesta como enviada.
        
        Args:
            response_id: ID de la respuesta
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        for i, response in enumerate(self.response_history):
            if response["response_id"] == response_id:
                self.response_history[i]["status"] = "sent"
                self.response_history[i]["sent_at"] = datetime.now().isoformat()
                self._save_response_history()
                logger.info(f"Respuesta marcada como enviada: {response_id}")
                return True
        
        logger.warning(f"No se encontró la respuesta con ID: {response_id}")
        return False
    
    def add_response_template(self, category: str, template: str) -> bool:
        """
        Añade una nueva plantilla de respuesta.
        
        Args:
            category: Categoría de la plantilla
            template: Texto de la plantilla
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        try:
            if category not in self.templates:
                self.templates[category] = []
            
            self.templates[category].append(template)
            
            # Actualizar configuración
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            config["templates"] = self.templates
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Plantilla añadida a categoría {category}")
            return True
        
        except Exception as e:
            logger.error(f"Error al añadir plantilla: {str(e)}")
            return False
    
    def update_blacklist(self, phrases: List[str], remove: bool = False) -> bool:
        """
        Actualiza la lista negra de frases.
        
        Args:
            phrases: Lista de frases
            remove: True para eliminar, False para añadir
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            if remove:
                self.blacklisted_phrases = [p for p in self.blacklisted_phrases if p not in phrases]
            else:
                self.blacklisted_phrases.extend([p for p in phrases if p not in self.blacklisted_phrases])
            
            # Actualizar configuración
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            config["blacklisted_phrases"] = self.blacklisted_phrases
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Lista negra actualizada: {len(self.blacklisted_phrases)} frases")
            return True
        
        except Exception as e:
            logger.error(f"Error al actualizar lista negra: {str(e)}")
            return False
    
    def get_user_interaction_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de interacción con usuarios.
        
        Returns:
            Estadísticas de interacción
        """
        total_users = len(self.user_interaction_history)
        total_interactions = sum(user["interaction_count"] for user in self.user_interaction_history.values())
        
        # Calcular usuarios activos (interacción en los últimos 30 días)
        now = datetime.now()
        active_users = 0
        for user in self.user_interaction_history.values():
            if user.get("last_interaction"):
                last_interaction = datetime.fromisoformat(user["last_interaction"])
                days_since_last = (now - last_interaction).days
                if days_since_last <= 30:
                    active_users += 1
        
        # Calcular sentimiento promedio
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for user in self.user_interaction_history.values():
            for sentiment_entry in user.get("sentiment_history", []):
                if sentiment_entry.get("sentiment") == "positive":
                    positive_count += 1
                elif sentiment_entry.get("sentiment") == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1
        
        total_sentiment = positive_count + negative_count + neutral_count
        
        sentiment_stats = {
            "positive_percent": (positive_count / total_sentiment) * 100 if total_sentiment > 0 else 0,
            "negative_percent": (negative_count / total_sentiment) * 100 if total_sentiment > 0 else 0,
            "neutral_percent": (neutral_count / total_sentiment) * 100 if total_sentiment > 0 else 0
        }
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_interactions": total_interactions,
            "avg_interactions_per_user": total_interactions / total_users if total_users > 0 else 0,
            "sentiment_stats": sentiment_stats,
            "response_count": len(self.response_history),
            "response_rate": self.response_rate
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Crear directorio de datos
    os.makedirs("data/engagement", exist_ok=True)
    
    # Inicializar respondedor
    responder = CommentResponder()
    
    # Ejemplo de comentario
    example_comment = {
        "comment_id": "comment123",
        "user_id": "user456",
        "username": "FanEnthusiast",
        "content_id": "video789",
        "platform": "youtube",
        "text": "¡Me encantó este video! ¿Cuándo subirás más contenido como este?",
        "created_at": datetime.now().isoformat()
    }
    
    # Procesar comentario
    response = responder.process_comment(example_comment)
    
    if response:
        print(f"Respuesta programada: {response['response_text']}")
        print(f"Programada para: {response['scheduled_time']}")
    else:
        print("No se generó respuesta para este comentario")
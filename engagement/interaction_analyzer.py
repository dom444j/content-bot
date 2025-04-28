import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
logger = logging.getLogger("InteractionAnalyzer")

class InteractionAnalyzer:
    """
    Sistema para analizar interacciones de usuarios con el contenido.
    Mide engagement diferido, patrones de interacción, y proporciona
    insights para optimizar estrategias de contenido y CTAs.
    """
    
    def __init__(self, data_path: str = "data/engagement/interactions"):
        """
        Inicializa el analizador de interacciones.
        
        Args:
            data_path: Ruta al directorio de datos de interacciones
        """
        self.data_path = data_path
        self.interactions = []
        self.content_metrics = {}
        self.user_metrics = {}
        self.platform_metrics = {}
        
        # Crear directorios necesarios
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Cargar datos existentes
        self._load_data()
        
        logger.info("Analizador de interacciones inicializado")
    
    def _load_data(self) -> None:
        """Carga los datos de interacciones desde archivos JSON."""
        try:
            # Cargar interacciones
            interactions_path = os.path.join(self.data_path, "interactions.json")
            if os.path.exists(interactions_path):
                with open(interactions_path, "r", encoding="utf-8") as f:
                    self.interactions = json.load(f)
                logger.info(f"Datos de interacciones cargados: {len(self.interactions)} registros")
            
            # Cargar métricas de contenido
            content_metrics_path = os.path.join(self.data_path, "content_metrics.json")
            if os.path.exists(content_metrics_path):
                with open(content_metrics_path, "r", encoding="utf-8") as f:
                    self.content_metrics = json.load(f)
                logger.info(f"Métricas de contenido cargadas: {len(self.content_metrics)} elementos")
            
            # Cargar métricas de usuarios
            user_metrics_path = os.path.join(self.data_path, "user_metrics.json")
            if os.path.exists(user_metrics_path):
                with open(user_metrics_path, "r", encoding="utf-8") as f:
                    self.user_metrics = json.load(f)
                logger.info(f"Métricas de usuarios cargadas: {len(self.user_metrics)} usuarios")
            
            # Cargar métricas de plataformas
            platform_metrics_path = os.path.join(self.data_path, "platform_metrics.json")
            if os.path.exists(platform_metrics_path):
                with open(platform_metrics_path, "r", encoding="utf-8") as f:
                    self.platform_metrics = json.load(f)
                logger.info(f"Métricas de plataformas cargadas: {len(self.platform_metrics)} plataformas")
        
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            self.interactions = []
            self.content_metrics = {}
            self.user_metrics = {}
            self.platform_metrics = {}
    
    def _save_data(self) -> None:
        """Guarda los datos de interacciones en archivos JSON."""
        try:
            # Guardar interacciones
            interactions_path = os.path.join(self.data_path, "interactions.json")
            with open(interactions_path, "w", encoding="utf-8") as f:
                json.dump(self.interactions, f, indent=4)
            
                        # Guardar métricas de contenido
            content_metrics_path = os.path.join(self.data_path, "content_metrics.json")
            with open(content_metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.content_metrics, f, indent=4)
            
            # Guardar métricas de usuarios
            user_metrics_path = os.path.join(self.data_path, "user_metrics.json")
            with open(user_metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.user_metrics, f, indent=4)
            
            # Guardar métricas de plataformas
            platform_metrics_path = os.path.join(self.data_path, "platform_metrics.json")
            with open(platform_metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.platform_metrics, f, indent=4)
            
            logger.info("Datos guardados correctamente")
        
        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
    
    def record_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """
        Registra una nueva interacción de usuario.
        
        Args:
            interaction_data: Datos de la interacción
            
        Returns:
            ID de la interacción registrada
        """
        # Validar datos mínimos requeridos
        required_fields = ["user_id", "content_id", "platform", "interaction_type"]
        for field in required_fields:
            if field not in interaction_data:
                logger.error(f"Falta campo requerido: {field}")
                return ""
        
        # Generar ID único para la interacción
        interaction_id = f"int_{len(self.interactions) + 1}_{int(datetime.now().timestamp())}"
        
        # Añadir timestamp si no existe
        if "timestamp" not in interaction_data:
            interaction_data["timestamp"] = datetime.now().isoformat()
        
        # Añadir ID de interacción
        interaction_data["interaction_id"] = interaction_id
        
        # Registrar interacción
        self.interactions.append(interaction_data)
        
        # Actualizar métricas
        self._update_metrics(interaction_data)
        
        # Guardar datos
        self._save_data()
        
        logger.info(f"Interacción registrada: {interaction_id} - {interaction_data['interaction_type']}")
        
        return interaction_id
    
    def _update_metrics(self, interaction: Dict[str, Any]) -> None:
        """
        Actualiza las métricas basadas en una nueva interacción.
        
        Args:
            interaction: Datos de la interacción
        """
        content_id = interaction["content_id"]
        user_id = interaction["user_id"]
        platform = interaction["platform"]
        interaction_type = interaction["interaction_type"]
        timestamp = interaction["timestamp"]
        
        # Actualizar métricas de contenido
        if content_id not in self.content_metrics:
            self.content_metrics[content_id] = {
                "total_interactions": 0,
                "interaction_types": {},
                "unique_users": set(),
                "first_interaction": timestamp,
                "last_interaction": timestamp,
                "interaction_timeline": {}
            }
        
        content_metric = self.content_metrics[content_id]
        content_metric["total_interactions"] += 1
        content_metric["last_interaction"] = timestamp
        
        if interaction_type not in content_metric["interaction_types"]:
            content_metric["interaction_types"][interaction_type] = 0
        content_metric["interaction_types"][interaction_type] += 1
        
        content_metric["unique_users"].add(user_id)
        
        # Convertir set a lista para serialización JSON
        content_metric["unique_users"] = list(content_metric["unique_users"])
        
        # Actualizar timeline de interacciones (por día)
        interaction_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
        if interaction_date not in content_metric["interaction_timeline"]:
            content_metric["interaction_timeline"][interaction_date] = 0
        content_metric["interaction_timeline"][interaction_date] += 1
        
        # Actualizar métricas de usuario
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = {
                "total_interactions": 0,
                "interaction_types": {},
                "content_interactions": {},
                "platforms": set(),
                "first_interaction": timestamp,
                "last_interaction": timestamp
            }
        
        user_metric = self.user_metrics[user_id]
        user_metric["total_interactions"] += 1
        user_metric["last_interaction"] = timestamp
        
        if interaction_type not in user_metric["interaction_types"]:
            user_metric["interaction_types"][interaction_type] = 0
        user_metric["interaction_types"][interaction_type] += 1
        
        if content_id not in user_metric["content_interactions"]:
            user_metric["content_interactions"][content_id] = 0
        user_metric["content_interactions"][content_id] += 1
        
        user_metric["platforms"].add(platform)
        
        # Convertir set a lista para serialización JSON
        user_metric["platforms"] = list(user_metric["platforms"])
        
        # Actualizar métricas de plataforma
        if platform not in self.platform_metrics:
            self.platform_metrics[platform] = {
                "total_interactions": 0,
                "interaction_types": {},
                "unique_users": set(),
                "unique_content": set(),
                "first_interaction": timestamp,
                "last_interaction": timestamp,
                "daily_interactions": {}
            }
        
        platform_metric = self.platform_metrics[platform]
        platform_metric["total_interactions"] += 1
        platform_metric["last_interaction"] = timestamp
        
        if interaction_type not in platform_metric["interaction_types"]:
            platform_metric["interaction_types"][interaction_type] = 0
        platform_metric["interaction_types"][interaction_type] += 1
        
        platform_metric["unique_users"].add(user_id)
        platform_metric["unique_content"].add(content_id)
        
        # Convertir sets a listas para serialización JSON
        platform_metric["unique_users"] = list(platform_metric["unique_users"])
        platform_metric["unique_content"] = list(platform_metric["unique_content"])
        
        # Actualizar interacciones diarias
        if interaction_date not in platform_metric["daily_interactions"]:
            platform_metric["daily_interactions"][interaction_date] = 0
        platform_metric["daily_interactions"][interaction_date] += 1
    
    def get_content_engagement(self, content_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas de engagement para un contenido específico.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            Métricas de engagement del contenido
        """
        if content_id not in self.content_metrics:
            logger.warning(f"No hay datos para el contenido: {content_id}")
            return {
                "content_id": content_id,
                "engagement_score": 0,
                "total_interactions": 0,
                "unique_users": 0,
                "interaction_types": {},
                "timeline": {}
            }
        
        metrics = self.content_metrics[content_id]
        
        # Calcular puntuación de engagement (fórmula personalizable)
        engagement_score = self._calculate_engagement_score(metrics)
        
        # Obtener interacciones por tipo
        interaction_types = metrics["interaction_types"]
        
        # Obtener timeline de interacciones
        timeline = metrics["interaction_timeline"]
        
        # Calcular usuarios únicos
        unique_users = len(metrics["unique_users"])
        
        return {
            "content_id": content_id,
            "engagement_score": engagement_score,
            "total_interactions": metrics["total_interactions"],
            "unique_users": unique_users,
            "interaction_types": interaction_types,
            "timeline": timeline,
            "first_interaction": metrics["first_interaction"],
            "last_interaction": metrics["last_interaction"]
        }
    
    def _calculate_engagement_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calcula una puntuación de engagement basada en las métricas.
        
        Args:
            metrics: Métricas de contenido
            
        Returns:
            Puntuación de engagement
        """
        # Pesos para diferentes tipos de interacción
        interaction_weights = {
            "view": 1,
            "like": 3,
            "comment": 5,
            "share": 10,
            "save": 7,
            "click": 2,
            "follow": 8,
            "subscribe": 15
        }
        
        # Calcular puntuación ponderada
        score = 0
        for interaction_type, count in metrics["interaction_types"].items():
            weight = interaction_weights.get(interaction_type, 1)
            score += count * weight
        
        # Normalizar por número de usuarios únicos
        unique_users = len(metrics["unique_users"])
        if unique_users > 0:
            score = score / unique_users
        
        return round(score, 2)
    
    def get_user_engagement(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas de engagement para un usuario específico.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Métricas de engagement del usuario
        """
        if user_id not in self.user_metrics:
            logger.warning(f"No hay datos para el usuario: {user_id}")
            return {
                "user_id": user_id,
                "engagement_score": 0,
                "total_interactions": 0,
                "interaction_types": {},
                "content_count": 0,
                "platforms": []
            }
        
        metrics = self.user_metrics[user_id]
        
        # Calcular puntuación de engagement
        engagement_score = sum(metrics["interaction_types"].values()) / len(metrics["content_interactions"])
        
        return {
            "user_id": user_id,
            "engagement_score": round(engagement_score, 2),
            "total_interactions": metrics["total_interactions"],
            "interaction_types": metrics["interaction_types"],
            "content_count": len(metrics["content_interactions"]),
            "platforms": metrics["platforms"],
            "first_interaction": metrics["first_interaction"],
            "last_interaction": metrics["last_interaction"],
            "top_content": self._get_top_content_for_user(user_id)
        }
    
    def _get_top_content_for_user(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene el contenido con más interacciones para un usuario.
        
        Args:
            user_id: ID del usuario
            limit: Número máximo de elementos a devolver
            
        Returns:
            Lista de contenido con más interacciones
        """
        if user_id not in self.user_metrics:
            return []
        
        content_interactions = self.user_metrics[user_id]["content_interactions"]
        
        # Ordenar por número de interacciones
        sorted_content = sorted(
            content_interactions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limitar resultados
        top_content = sorted_content[:limit]
        
        # Formatear resultados
        result = []
        for content_id, interactions in top_content:
            result.append({
                "content_id": content_id,
                "interactions": interactions
            })
        
        return result
    
    def get_platform_metrics(self, platform: str) -> Dict[str, Any]:
        """
        Obtiene métricas de engagement para una plataforma específica.
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Métricas de engagement de la plataforma
        """
        if platform not in self.platform_metrics:
            logger.warning(f"No hay datos para la plataforma: {platform}")
            return {
                "platform": platform,
                "total_interactions": 0,
                "unique_users": 0,
                "unique_content": 0,
                "interaction_types": {}
            }
        
        metrics = self.platform_metrics[platform]
        
        return {
            "platform": platform,
            "total_interactions": metrics["total_interactions"],
            "unique_users": len(metrics["unique_users"]),
            "unique_content": len(metrics["unique_content"]),
            "interaction_types": metrics["interaction_types"],
            "first_interaction": metrics["first_interaction"],
            "last_interaction": metrics["last_interaction"],
            "daily_interactions": metrics["daily_interactions"]
        }
    
    def get_engagement_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene tendencias de engagement durante un período de tiempo.
        
        Args:
            days: Número de días a analizar
            
        Returns:
            Tendencias de engagement
        """
        # Calcular fecha de inicio
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Inicializar datos de tendencias
        trends = {
            "daily_interactions": {},
            "interaction_types": {},
            "platform_distribution": {},
            "total_interactions": 0,
            "unique_users": set(),
            "unique_content": set()
        }
        
        # Generar fechas para el período
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            trends["daily_interactions"][date_str] = 0
            current_date += timedelta(days=1)
        
        # Filtrar interacciones por fecha
        filtered_interactions = []
        for interaction in self.interactions:
            try:
                interaction_date = datetime.fromisoformat(interaction["timestamp"])
                if start_date <= interaction_date <= end_date:
                    filtered_interactions.append(interaction)
                    
                    # Actualizar contadores
                    date_str = interaction_date.strftime("%Y-%m-%d")
                    trends["daily_interactions"][date_str] += 1
                    trends["total_interactions"] += 1
                    
                    # Actualizar usuarios y contenido únicos
                    trends["unique_users"].add(interaction["user_id"])
                    trends["unique_content"].add(interaction["content_id"])
                    
                    # Actualizar tipos de interacción
                    interaction_type = interaction["interaction_type"]
                    if interaction_type not in trends["interaction_types"]:
                        trends["interaction_types"][interaction_type] = 0
                    trends["interaction_types"][interaction_type] += 1
                    
                    # Actualizar distribución por plataforma
                    platform = interaction["platform"]
                    if platform not in trends["platform_distribution"]:
                        trends["platform_distribution"][platform] = 0
                    trends["platform_distribution"][platform] += 1
            
            except (ValueError, KeyError):
                continue
        
        # Convertir sets a contadores para la respuesta
        trends["unique_users"] = len(trends["unique_users"])
        trends["unique_content"] = len(trends["unique_content"])
        
        # Calcular promedios diarios
        if days > 0:
            trends["avg_daily_interactions"] = trends["total_interactions"] / days
        else:
            trends["avg_daily_interactions"] = 0
        
        return trends
    
    def get_content_comparison(self, content_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compara métricas de engagement entre varios contenidos.
        
        Args:
            content_ids: Lista de IDs de contenido a comparar
            
        Returns:
            Comparación de métricas
        """
        comparison = {
            "engagement_scores": [],
            "interaction_counts": [],
            "user_counts": []
        }
        
        for content_id in content_ids:
            metrics = self.get_content_engagement(content_id)
            
            # Añadir a comparación de puntuaciones
            comparison["engagement_scores"].append({
                "content_id": content_id,
                "score": metrics["engagement_score"]
            })
            
            # Añadir a comparación de interacciones
            comparison["interaction_counts"].append({
                "content_id": content_id,
                "count": metrics["total_interactions"]
            })
            
            # Añadir a comparación de usuarios
            comparison["user_counts"].append({
                "content_id": content_id,
                "count": metrics["unique_users"]
            })
        
        # Ordenar resultados
        comparison["engagement_scores"].sort(key=lambda x: x["score"], reverse=True)
        comparison["interaction_counts"].sort(key=lambda x: x["count"], reverse=True)
        comparison["user_counts"].sort(key=lambda x: x["count"], reverse=True)
        
        return comparison
    
    def export_engagement_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, format_type: str = "json") -> Dict[str, Any]:
        """
        Exporta datos de engagement para análisis externo.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            format_type: Formato de exportación (json, csv)
            
        Returns:
            Datos de exportación y ruta del archivo
        """
        # Validar fechas
        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                logger.error(f"Formato de fecha de inicio no válido: {start_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha de inicio no válido"
                }
        else:
            # Por defecto, último mes
            start_date_obj = datetime.now() - timedelta(days=30)
            start_date = start_date_obj.isoformat()
        
        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                logger.error(f"Formato de fecha de fin no válido: {end_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha de fin no válido"
                }
        else:
            end_date_obj = datetime.now()
            end_date = end_date_obj.isoformat()
        
        # Validar formato
        valid_formats = ["json", "csv"]
        if format_type not in valid_formats:
            logger.error(f"Formato no válido: {format_type}")
            return {
                "status": "error",
                "message": f"Formato no válido. Debe ser uno de: {', '.join(valid_formats)}"
            }
        
        # Filtrar interacciones por fecha
        filtered_interactions = []
        for interaction in self.interactions:
            try:
                interaction_date = datetime.fromisoformat(interaction["timestamp"])
                if start_date_obj <= interaction_date <= end_date_obj:
                    filtered_interactions.append(interaction)
            except ValueError:
                continue
        
        # Crear directorio de exportación
        export_dir = "data/engagement/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engagement_export_{timestamp}.{format_type}"
        filepath = os.path.join(export_dir, filename)
        
        # Exportar datos
        if format_type == "json":
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "interaction_count": len(filtered_interactions)
                },
                "interactions": filtered_interactions,
                "content_metrics": self.content_metrics,
                "platform_metrics": self.platform_metrics
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
        
        elif format_type == "csv":
            import csv
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                # Definir campos
                fieldnames = [
                    "interaction_id", "user_id", "content_id", "platform", 
                    "interaction_type", "timestamp", "metadata"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Escribir interacciones
                for interaction in filtered_interactions:
                    # Crear fila con solo los campos definidos
                    row = {field: interaction.get(field, "") for field in fieldnames}
                    
                    # Convertir metadata a string si existe
                    if "metadata" in row and isinstance(row["metadata"], dict):
                        row["metadata"] = json.dumps(row["metadata"])
                    
                    writer.writerow(row)
        
        logger.info(f"Datos de engagement exportados: {filepath}")
        
        return {
            "status": "success",
            "message": f"Datos exportados a {filepath}",
            "filepath": filepath,
            "interaction_count": len(filtered_interactions),
            "format": format_type
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar analizador
    analyzer = InteractionAnalyzer()
    
    # Ejemplo de interacción
    example_interaction = {
        "user_id": "user123",
        "content_id": "video456",
        "platform": "youtube",
        "interaction_type": "like",
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "device": "mobile",
            "location": "home page"
        }
    }
    
    # Registrar interacción
    interaction_id = analyzer.record_interaction(example_interaction)
    
    if interaction_id:
        print(f"Interacción registrada con ID: {interaction_id}")
        
        # Obtener métricas de contenido
        content_metrics = analyzer.get_content_engagement("video456")
        print(f"Puntuación de engagement: {content_metrics['engagement_score']}")
    else:
        print("Error al registrar interacción")
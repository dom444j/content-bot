import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random
import shutil

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/marketplace.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cta_marketplace")

class CTAMarketplace:
    """
    Sistema de intercambio y optimización de CTAs visuales.
    
    Permite:
    - Registrar CTAs visuales con metadatos
    - Buscar CTAs por rendimiento, plataforma, tema
    - Intercambiar CTAs entre canales
    - Analizar rendimiento de CTAs
    - Recomendar CTAs basados en contexto
    """
    
    def __init__(self, base_path: str = None):
        """
        Inicializa el marketplace de CTAs.
        
        Args:
            base_path: Ruta base para almacenar datos y assets
        """
        self.base_path = base_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.ctas_db_path = os.path.join(self.base_path, "cta_marketplace.json")
        self.cta_assets_path = os.path.join(self.base_path, "cta_assets")
        
        # Crear directorios si no existen
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.cta_assets_path, exist_ok=True)
        
        # Cargar base de datos de CTAs
        self.ctas_db = self._load_ctas_database()
        
        # Métricas del marketplace
        self.marketplace_stats = {
            "total_ctas": len(self.ctas_db.get("ctas", [])),
            "exchanges": 0,
            "reuse_count": 0,
            "top_performing_ctas": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_ctas_database(self) -> Dict[str, Any]:
        """
        Carga la base de datos de CTAs desde el archivo JSON.
        
        Returns:
            Diccionario con la base de datos de CTAs
        """
        if os.path.exists(self.ctas_db_path):
            try:
                with open(self.ctas_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error al cargar base de datos de CTAs: {str(e)}")
                return {"ctas": [], "exchanges": [], "stats": {}}
        else:
            return {"ctas": [], "exchanges": [], "stats": {}}
    
    def _save_ctas_database(self) -> bool:
        """
        Guarda la base de datos de CTAs en el archivo JSON.
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            with open(self.ctas_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.ctas_db, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error al guardar base de datos de CTAs: {str(e)}")
            return False
    
    def register_cta(self, 
                    cta_data: Dict[str, Any], 
                    asset_path: str = None) -> Dict[str, Any]:
        """
        Registra un nuevo CTA visual en el marketplace.
        
        Args:
            cta_data: Datos del CTA (texto, tipo, plataforma, etc.)
            asset_path: Ruta al archivo de asset visual (opcional)
            
        Returns:
            Información sobre el registro del CTA
        """
        try:
            # Validar datos mínimos
            required_fields = ["text", "type", "platform", "position"]
            for field in required_fields:
                if field not in cta_data:
                    return {
                        "status": "error",
                        "message": f"Falta campo requerido: {field}"
                    }
            
            # Generar ID único para el CTA
            cta_id = f"cta_{uuid.uuid4().hex[:8]}"
            
            # Copiar asset si se proporciona
            asset_filename = None
            if asset_path and os.path.exists(asset_path):
                # Obtener extensión del archivo
                _, ext = os.path.splitext(asset_path)
                asset_filename = f"{cta_id}{ext}"
                asset_dest_path = os.path.join(self.cta_assets_path, asset_filename)
                
                # Copiar archivo
                shutil.copy2(asset_path, asset_dest_path)
            
            # Crear registro de CTA
            cta_record = {
                "id": cta_id,
                "text": cta_data["text"],
                "type": cta_data["type"],  # text, button, overlay, animation
                "platform": cta_data["platform"],  # youtube, tiktok, instagram, etc.
                "position": cta_data["position"],  # start, middle, end, specific_time
                "timing": cta_data.get("timing", {
                    "start_time": 0,
                    "duration": 3
                }),
                "style": cta_data.get("style", {
                    "font": "Arial",
                    "color": "#FFFFFF",
                    "background": "#000000",
                    "opacity": 0.8,
                    "animation": "fade"
                }),
                "topic": cta_data.get("topic", "general"),
                "tags": cta_data.get("tags", []),
                "asset_filename": asset_filename,
                "creator_channel_id": cta_data.get("creator_channel_id", ""),
                "is_public": cta_data.get("is_public", True),
                "is_premium": cta_data.get("is_premium", False),
                "price": cta_data.get("price", 0),  # Para CTAs premium
                "performance": {
                    "conversion_rate": 0,
                    "engagement_rate": 0,
                    "retention_impact": 0,
                    "usage_count": 0,
                    "rating": 0,
                    "last_used": None
                },
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Añadir a la base de datos
            self.ctas_db["ctas"].append(cta_record)
            
            # Actualizar estadísticas
            self.marketplace_stats["total_ctas"] = len(self.ctas_db["ctas"])
            self.marketplace_stats["last_updated"] = datetime.now().isoformat()
            
            # Guardar cambios
            self._save_ctas_database()
            
            return {
                "status": "success",
                "message": "CTA registrado correctamente",
                "cta_id": cta_id,
                "cta_data": cta_record
            }
            
        except Exception as e:
            logger.error(f"Error al registrar CTA: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al registrar CTA: {str(e)}"
            }
    
    def search_ctas(self, 
                   filters: Dict[str, Any] = None, 
                   sort_by: str = "performance.conversion_rate", 
                   limit: int = 10) -> Dict[str, Any]:
        """
        Busca CTAs en el marketplace según filtros.
        
        Args:
            filters: Criterios de búsqueda (plataforma, tipo, tema, etc.)
            sort_by: Campo para ordenar resultados
            limit: Número máximo de resultados
            
        Returns:
            Lista de CTAs que cumplen los criterios
        """
        try:
            filters = filters or {}
            results = []
            
            # Aplicar filtros
            for cta in self.ctas_db["ctas"]:
                match = True
                
                for key, value in filters.items():
                    # Manejar campos anidados (ej: performance.conversion_rate)
                    if "." in key:
                        parts = key.split(".")
                        current = cta
                        for part in parts:
                            if part in current:
                                current = current[part]
                            else:
                                match = False
                                break
                        
                        if match and current != value:
                            match = False
                    
                    # Manejar listas (ej: tags)
                    elif key == "tags" and isinstance(value, list):
                        if not all(tag in cta.get("tags", []) for tag in value):
                            match = False
                    
                    # Manejar rangos (ej: performance.conversion_rate_min, performance.conversion_rate_max)
                    elif key.endswith("_min"):
                        base_key = key[:-4]
                        if "." in base_key:
                            parts = base_key.split(".")
                            current = cta
                            for part in parts:
                                if part in current:
                                    current = current[part]
                                else:
                                    match = False
                                    break
                            
                            if match and current < value:
                                match = False
                        else:
                            if cta.get(base_key, 0) < value:
                                match = False
                    
                    elif key.endswith("_max"):
                        base_key = key[:-4]
                        if "." in base_key:
                            parts = base_key.split(".")
                            current = cta
                            for part in parts:
                                if part in current:
                                    current = current[part]
                                else:
                                    match = False
                                    break
                            
                            if match and current > value:
                                match = False
                        else:
                            if cta.get(base_key, 0) > value:
                                match = False
                    
                    # Campos simples
                    elif cta.get(key) != value:
                        match = False
                
                if match:
                    results.append(cta)
            
            # Ordenar resultados
            if sort_by:
                if "." in sort_by:
                    # Ordenar por campo anidado
                    parts = sort_by.split(".")
                    
                    def get_nested_value(item, parts):
                        for part in parts:
                            if part in item:
                                item = item[part]
                            else:
                                return 0
                        return item
                    
                    results.sort(key=lambda x: get_nested_value(x, parts), reverse=True)
                else:
                    # Ordenar por campo simple
                    results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
            
            # Limitar resultados
            results = results[:limit]
            
            return {
                "status": "success",
                "count": len(results),
                "ctas": results
            }
            
        except Exception as e:
            logger.error(f"Error al buscar CTAs: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al buscar CTAs: {str(e)}"
            }
    
    def get_cta_by_id(self, cta_id: str) -> Dict[str, Any]:
        """
        Obtiene un CTA por su ID.
        
        Args:
            cta_id: ID del CTA
            
        Returns:
            Datos del CTA
        """
        try:
            for cta in self.ctas_db["ctas"]:
                if cta["id"] == cta_id:
                    return {
                        "status": "success",
                        "cta": cta
                    }
            
            return {
                "status": "error",
                "message": f"CTA no encontrado: {cta_id}"
            }
            
        except Exception as e:
            logger.error(f"Error al obtener CTA: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener CTA: {str(e)}"
            }
    
    def update_cta_performance(self, 
                              cta_id: str, 
                              performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza métricas de rendimiento de un CTA.
        
        Args:
            cta_id: ID del CTA
            performance_data: Datos de rendimiento a actualizar
            
        Returns:
            Resultado de la actualización
        """
        try:
            cta_found = False
            
            for cta in self.ctas_db["ctas"]:
                if cta["id"] == cta_id:
                    cta_found = True
                    
                    # Actualizar campos de rendimiento
                    for key, value in performance_data.items():
                        if key in cta["performance"]:
                            cta["performance"][key] = value
                    
                    # Actualizar fecha de uso si se incrementa el contador
                    if "usage_count" in performance_data:
                        cta["performance"]["last_used"] = datetime.now().isoformat()
                    
                    # Actualizar fecha de modificación
                    cta["updated_at"] = datetime.now().isoformat()
                    
                    break
            
            if not cta_found:
                return {
                    "status": "error",
                    "message": f"CTA no encontrado: {cta_id}"
                }
            
            # Actualizar top performing CTAs
            self._update_top_performing_ctas()
            
            # Guardar cambios
            self._save_ctas_database()
            
            return {
                "status": "success",
                "message": "Rendimiento de CTA actualizado correctamente"
            }
            
        except Exception as e:
            logger.error(f"Error al actualizar rendimiento de CTA: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al actualizar rendimiento de CTA: {str(e)}"
            }
    
    def _update_top_performing_ctas(self) -> None:
        """
        Actualiza la lista de CTAs con mejor rendimiento.
        """
        try:
            # Ordenar CTAs por tasa de conversión
            sorted_ctas = sorted(
                self.ctas_db["ctas"],
                key=lambda x: x["performance"]["conversion_rate"],
                reverse=True
            )
            
            # Tomar los 10 mejores
            top_ctas = sorted_ctas[:10]
            
            # Actualizar estadísticas
            self.marketplace_stats["top_performing_ctas"] = [
                {
                    "id": cta["id"],
                    "text": cta["text"],
                    "platform": cta["platform"],
                    "conversion_rate": cta["performance"]["conversion_rate"],
                    "usage_count": cta["performance"]["usage_count"]
                }
                for cta in top_ctas
            ]
            
            self.marketplace_stats["last_updated"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error al actualizar top performing CTAs: {str(e)}")
    
    def exchange_cta(self, 
                    source_channel_id: str, 
                    target_channel_id: str,
                    cta_id: str) -> Dict[str, Any]:
        """
        Registra un intercambio de CTA entre canales.
        
        Args:
            source_channel_id: ID del canal origen
            target_channel_id: ID del canal destino
            cta_id: ID del CTA intercambiado
            
        Returns:
            Resultado del intercambio
        """
        try:
            # Verificar que el CTA existe
            cta_found = False
            cta_data = None
            
            for cta in self.ctas_db["ctas"]:
                if cta["id"] == cta_id:
                    cta_found = True
                    cta_data = cta
                    break
            
            if not cta_found:
                return {
                    "status": "error",
                    "message": f"CTA no encontrado: {cta_id}"
                }
            
            # Verificar si el CTA es público o pertenece al canal origen
            if not cta_data["is_public"] and cta_data["creator_channel_id"] != source_channel_id:
                return {
                    "status": "error",
                    "message": "No tienes permisos para intercambiar este CTA"
                }
            
            # Registrar intercambio
            exchange_record = {
                "id": f"exchange_{uuid.uuid4().hex[:8]}",
                "cta_id": cta_id,
                "source_channel_id": source_channel_id,
                "target_channel_id": target_channel_id,
                "timestamp": datetime.now().isoformat(),
                "is_premium": cta_data["is_premium"],
                "price": cta_data["price"]
            }
            
            self.ctas_db["exchanges"].append(exchange_record)
            
            # Actualizar contador de uso del CTA
            for cta in self.ctas_db["ctas"]:
                if cta["id"] == cta_id:
                    cta["performance"]["usage_count"] += 1
                    cta["performance"]["last_used"] = datetime.now().isoformat()
                    break
            
            # Actualizar estadísticas
            self.marketplace_stats["exchanges"] += 1
            self.marketplace_stats["reuse_count"] += 1
            self.marketplace_stats["last_updated"] = datetime.now().isoformat()
            
            # Guardar cambios
            self._save_ctas_database()
            
            return {
                "status": "success",
                "message": "Intercambio de CTA registrado correctamente",
                "exchange_id": exchange_record["id"],
                "cta_data": cta_data
            }
            
        except Exception as e:
            logger.error(f"Error al intercambiar CTA: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al intercambiar CTA: {str(e)}"
            }
    
    def recommend_ctas(self, 
                      context: Dict[str, Any], 
                      limit: int = 5) -> Dict[str, Any]:
        """
        Recomienda CTAs basados en contexto.
        
        Args:
            context: Contexto para la recomendación (plataforma, tema, audiencia)
            limit: Número máximo de recomendaciones
            
        Returns:
            Lista de CTAs recomendados
        """
        try:
            platform = context.get("platform", "")
            topic = context.get("topic", "")
            audience = context.get("audience", {})
            
            # Filtros base
            filters = {}
            if platform:
                filters["platform"] = platform
            if topic:
                filters["topic"] = topic
            
            # Buscar CTAs que cumplan los filtros básicos
            search_result = self.search_ctas(
                filters=filters,
                sort_by="performance.conversion_rate",
                limit=50  # Obtener más para filtrar después
            )
            
            if search_result["status"] != "success":
                return search_result
            
            candidates = search_result["ctas"]
            
            # Si no hay suficientes candidatos, ampliar búsqueda
            if len(candidates) < limit:
                # Buscar sin filtro de tema
                if topic:
                    search_result = self.search_ctas(
                        filters={"platform": platform} if platform else {},
                        sort_by="performance.conversion_rate",
                        limit=50
                    )
                    
                    if search_result["status"] == "success":
                        candidates.extend([c for c in search_result["ctas"] if c["id"] not in [x["id"] for x in candidates]])
            
            # Calcular puntuación para cada candidato
            scored_candidates = []
            
            for cta in candidates:
                score = 0
                
                # Puntos por rendimiento
                score += cta["performance"]["conversion_rate"] * 5
                score += cta["performance"]["engagement_rate"] * 3
                score += cta["performance"]["retention_impact"] * 2
                
                # Puntos por coincidencia de plataforma
                if platform and cta["platform"] == platform:
                    score += 10
                
                # Puntos por coincidencia de tema
                if topic and cta["topic"] == topic:
                    score += 8
                
                # Puntos por coincidencia de tags
                if "tags" in context and context["tags"]:
                    matching_tags = set(context["tags"]).intersection(set(cta.get("tags", [])))
                    score += len(matching_tags) * 2
                
                # Puntos por audiencia
                if audience.get("age_range") and "target_audience" in cta:
                    if audience["age_range"] == cta["target_audience"].get("age_range"):
                        score += 3
                
                # Penalización por uso excesivo
                if cta["performance"]["usage_count"] > 100:
                    score -= (cta["performance"]["usage_count"] - 100) / 100
                
                scored_candidates.append((cta, score))
            
            # Ordenar por puntuación
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Tomar los mejores
            recommendations = [cta for cta, _ in scored_candidates[:limit]]
            
            return {
                "status": "success",
                "count": len(recommendations),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error al recomendar CTAs: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al recomendar CTAs: {str(e)}"
            }
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del marketplace.
        
        Returns:
            Estadísticas del marketplace
        """
        try:
            # Actualizar estadísticas
            self.marketplace_stats["total_ctas"] = len(self.ctas_db["ctas"])
            self.marketplace_stats["exchanges"] = len(self.ctas_db["exchanges"])
            
            # Calcular reutilización total
            total_usage = sum(cta["performance"]["usage_count"] for cta in self.ctas_db["ctas"])
            self.marketplace_stats["reuse_count"] = total_usage
            
            # Actualizar top performing CTAs
            self._update_top_performing_ctas()
            
            # Calcular estadísticas por plataforma
            platform_stats = {}
            for cta in self.ctas_db["ctas"]:
                platform = cta["platform"]
                if platform not in platform_stats:
                    platform_stats[platform] = {
                        "count": 0,
                        "total_usage": 0,
                        "avg_conversion": 0,
                        "top_cta": None
                    }
                
                platform_stats[platform]["count"] += 1
                platform_stats[platform]["total_usage"] += cta["performance"]["usage_count"]
                
                # Actualizar top CTA para la plataforma
                if (platform_stats[platform]["top_cta"] is None or 
                    cta["performance"]["conversion_rate"] > platform_stats[platform]["top_cta"]["conversion_rate"]):
                    platform_stats[platform]["top_cta"] = {
                        "id": cta["id"],
                        "text": cta["text"],
                        "conversion_rate": cta["performance"]["conversion_rate"]
                    }
            
            # Calcular promedios
            for platform in platform_stats:
                if platform_stats[platform]["count"] > 0:
                    total_conversion = sum(
                        cta["performance"]["conversion_rate"] 
                        for cta in self.ctas_db["ctas"] 
                        if cta["platform"] == platform
                    )
                    platform_stats[platform]["avg_conversion"] = total_conversion / platform_stats[platform]["count"]
            
            return {
                "status": "success",
                "basic_stats": self.marketplace_stats,
                "platform_stats": platform_stats,
                "exchange_stats": {
                    "total_exchanges": len(self.ctas_db["exchanges"]),
                    "premium_exchanges": sum(1 for e in self.ctas_db["exchanges"] if e["is_premium"]),
                    "total_value": sum(e["price"] for e in self.ctas_db["exchanges"] if e["is_premium"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas del marketplace: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener estadísticas del marketplace: {str(e)}"
            }
    
    def delete_cta(self, cta_id: str, channel_id: str = None) -> Dict[str, Any]:
        """
        Elimina un CTA del marketplace.
        
        Args:
            cta_id: ID del CTA a eliminar
            channel_id: ID del canal que solicita la eliminación (para verificar permisos)
            
        Returns:
            Resultado de la eliminación
        """
        try:
            cta_found = False
            cta_index = -1
            
            for i, cta in enumerate(self.ctas_db["ctas"]):
                if cta["id"] == cta_id:
                    cta_found = True
                    cta_index = i
                    
                    # Verificar permisos si se proporciona channel_id
                    if channel_id and cta["creator_channel_id"] != channel_id:
                        return {
                            "status": "error",
                            "message": "No tienes permisos para eliminar este CTA"
                        }
                    
                    break
            
            if not cta_found:
                return {
                    "status": "error",
                    "message": f"CTA no encontrado: {cta_id}"
                }
            
            # Eliminar archivo de asset si existe
            cta = self.ctas_db["ctas"][cta_index]
            if cta.get("asset_filename"):
                asset_path = os.path.join(self.cta_assets_path, cta["asset_filename"])
                if os.path.exists(asset_path):
                    os.remove(asset_path)
            
            # Eliminar CTA de la base de datos
            del self.ctas_db["ctas"][cta_index]
            
            # Actualizar estadísticas
            self.marketplace_stats["total_ctas"] = len(self.ctas_db["ctas"])
            self.marketplace_stats["last_updated"] = datetime.now().isoformat()
            
            # Guardar cambios
            self._save_ctas_database()
            
            return {
                "status": "success",
                "message": "CTA eliminado correctamente"
            }
            
        except Exception as e:
            logger.error(f"Error al eliminar CTA: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al eliminar CTA: {str(e)}"
            }
    
    def get_random_cta(self, platform: str = None, topic: str = None) -> Dict[str, Any]:
        """
        Obtiene un CTA aleatorio, opcionalmente filtrado por plataforma y tema.
        
        Args:
            platform: Plataforma para filtrar
            topic: Tema para filtrar
            
        Returns:
            CTA aleatorio
        """
        try:
            # Filtrar CTAs
            filtered_ctas = self.ctas_db["ctas"]
            
            if platform:
                filtered_ctas = [cta for cta in filtered_ctas if cta["platform"] == platform]
            
            if topic:
                filtered_ctas = [cta for cta in filtered_ctas if cta["topic"] == topic]
            
            if not filtered_ctas:
                return {
                    "status": "error",
                    "message": "No se encontraron CTAs con los filtros especificados"
                }
            
            # Seleccionar CTA aleatorio
            random_cta = random.choice(filtered_ctas)
            
            return {
                "status": "success",
                "cta": random_cta
            }
            
        except Exception as e:
            logger.error(f"Error al obtener CTA aleatorio: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener CTA aleatorio: {str(e)}"
            }

# Ejemplo de uso
if __name__ == "__main__":
    marketplace = CTAMarketplace()
    
    # Registrar un CTA de ejemplo
    cta_data = {
        "text": "¡Sigue para más consejos de cripto!",
        "type": "overlay",
        "platform": "tiktok",
        "position": "middle",
        "timing": {
            "start_time": 6,
            "duration": 3
        },
        "style": {
            "font": "Montserrat",
            "color": "#FFFFFF",
            "background": "#FF0000",
            "opacity": 0.9,
            "animation": "slide_up"
        },
        "topic": "crypto",
        "tags": ["finance", "cryptocurrency", "investment"],
        "creator_channel_id": "channel_123",
        "is_public": True
    }
    
    result = marketplace.register_cta(cta_data)
    print(f"Registro de CTA: {result['status']}")
    
    if result['status'] == 'success':
        cta_id = result['cta_id']
        
        # Actualizar rendimiento
        performance_data = {
            "conversion_rate": 0.15,
            "engagement_rate": 0.25,
            "retention_impact": 0.1,
            "usage_count": 10
        }
        
        update_result = marketplace.update_cta_performance(cta_id, performance_data)
        print(f"Actualización de rendimiento: {update_result['status']}")
        
                # Buscar CTAs
        search_result = marketplace.search_ctas(
            filters={"platform": "tiktok", "topic": "crypto"},
            sort_by="performance.conversion_rate"
        )
        print(f"Búsqueda de CTAs: {search_result['status']}")
        print(f"CTAs encontrados: {search_result.get('count', 0)}")
        
        # Recomendar CTAs basados en contexto
        context = {
            "platform": "tiktok",
            "topic": "crypto",
            "tags": ["investment", "finance"],
            "audience": {
                "age_range": "18-34",
                "interests": ["finance", "technology"]
            }
        }
        
        recommendations = marketplace.recommend_ctas(context, limit=3)
        print(f"Recomendaciones: {recommendations['status']}")
        
        # Intercambiar CTA
        exchange_result = marketplace.exchange_cta(
            source_channel_id="channel_123",
            target_channel_id="channel_456",
            cta_id=cta_id
        )
        print(f"Intercambio de CTA: {exchange_result['status']}")
        
        # Obtener estadísticas del marketplace
        stats = marketplace.get_marketplace_stats()
        print(f"Estadísticas: {stats['status']}")
        print(f"Total CTAs: {stats['basic_stats']['total_ctas']}")
        print(f"Total intercambios: {stats['exchange_stats']['total_exchanges']}")
        
        # Obtener CTA aleatorio
        random_cta = marketplace.get_random_cta(platform="tiktok")
        print(f"CTA aleatorio: {random_cta['status']}")
        
        # Eliminar CTA (comentado para no eliminar el ejemplo)
        # delete_result = marketplace.delete_cta(cta_id, channel_id="channel_123")
        # print(f"Eliminación de CTA: {delete_result['status']}")
"""
Módulo para gestionar programas de afiliados y optimizar conversiones.
Permite rastrear enlaces, analizar conversiones y maximizar ingresos por afiliados.
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import requests
from collections import defaultdict

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AffiliateEngine:
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el motor de afiliados.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.affiliate_programs = self.config.get("affiliate_programs", {})
        self.links_data = self._load_links_data()
        self.conversions = self._load_conversions()
        self.second_tier = self._load_second_tier()
        
        # Crear directorios si no existen
        os.makedirs("data/affiliate", exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde el archivo JSON."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            return {}
    
    def _load_links_data(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de enlaces de afiliados."""
        try:
            links_path = "data/affiliate/links.json"
            if os.path.exists(links_path):
                with open(links_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar datos de enlaces: {e}")
            return {}
    
    def _load_conversions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Carga historial de conversiones."""
        try:
            conversions_path = "data/affiliate/conversions.json"
            if os.path.exists(conversions_path):
                with open(conversions_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return defaultdict(list)
        except Exception as e:
            logger.error(f"Error al cargar conversiones: {e}")
            return defaultdict(list)
    
    def _load_second_tier(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de afiliados de segundo nivel."""
        try:
            second_tier_path = "data/affiliate/second_tier.json"
            if os.path.exists(second_tier_path):
                with open(second_tier_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar datos de segundo nivel: {e}")
            return {}
    
    def _save_links_data(self) -> None:
        """Guarda datos de enlaces de afiliados."""
        try:
            links_path = "data/affiliate/links.json"
            with open(links_path, "w", encoding="utf-8") as f:
                json.dump(self.links_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar datos de enlaces: {e}")
    
    def _save_conversions(self) -> None:
        """Guarda historial de conversiones."""
        try:
            conversions_path = "data/affiliate/conversions.json"
            with open(conversions_path, "w", encoding="utf-8") as f:
                json.dump(self.conversions, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar conversiones: {e}")
    
    def _save_second_tier(self) -> None:
        """Guarda datos de afiliados de segundo nivel."""
        try:
            second_tier_path = "data/affiliate/second_tier.json"
            with open(second_tier_path, "w", encoding="utf-8") as f:
                json.dump(self.second_tier, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar datos de segundo nivel: {e}")
    
    def create_affiliate_link(self, program_id: str, product_id: str, channel_id: str, 
                             campaign: Optional[str] = None) -> Dict[str, Any]:
        """
        Crea un nuevo enlace de afiliado.
        
        Args:
            program_id: ID del programa de afiliados
            product_id: ID del producto
            channel_id: ID del canal
            campaign: Nombre de la campaña (opcional)
            
        Returns:
            Datos del enlace creado
        """
        if program_id not in self.affiliate_programs:
            logger.warning(f"Programa de afiliados no encontrado: {program_id}")
            return {
                "status": "error",
                "message": "Programa de afiliados no encontrado"
            }
        
        program = self.affiliate_programs[program_id]
        base_url = program.get("base_url", "")
        
        if not base_url:
            logger.warning(f"URL base no configurada para programa: {program_id}")
            return {
                "status": "error",
                "message": "URL base no configurada para el programa"
            }
        
        # Generar ID único para el enlace
        link_id = f"{program_id}_{product_id}_{channel_id}_{int(time.time())}"
        
        # Crear enlace según formato del programa
        affiliate_id = program.get("affiliate_id", "")
        tracking_param = program.get("tracking_param", "ref")
        
        # Construir URL
        if "?" in base_url:
            separator = "&"
        else:
            separator = "?"
        
        # Añadir parámetros específicos del producto
        product_url = f"{base_url}/product/{product_id}"
        
        # Añadir parámetros de seguimiento
        tracking_value = f"{affiliate_id}_{channel_id}"
        if campaign:
            tracking_value += f"_{campaign}"
        
        full_url = f"{product_url}{separator}{tracking_param}={tracking_value}"
        
        # Crear datos del enlace
        link_data = {
            "link_id": link_id,
            "program_id": program_id,
            "product_id": product_id,
            "channel_id": channel_id,
            "campaign": campaign,
            "url": full_url,
            "short_url": self._generate_short_url(full_url),
            "created_at": datetime.now().isoformat(),
            "clicks": 0,
            "conversions": 0,
            "revenue": 0.0,
            "active": True
        }
        
        # Guardar enlace
        self.links_data[link_id] = link_data
        self._save_links_data()
        
        logger.info(f"Enlace de afiliado creado: {link_id}")
        
        return link_data
    
    def _generate_short_url(self, url: str) -> str:
        """
        Genera una URL corta para el enlace de afiliado.
        En una implementación real, usaría un servicio como Bitly o TinyURL.
        
        Args:
            url: URL original
            
        Returns:
            URL acortada
        """
        # Simulación de acortamiento
        # En producción, integrar con API de acortador
        short_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        return f"https://short.link/{short_id}"
    
    def track_click(self, link_id: str, source: Optional[str] = None, 
                   user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registra un clic en un enlace de afiliado.
        
        Args:
            link_id: ID del enlace
            source: Fuente del clic (opcional)
            user_data: Datos del usuario (opcional)
            
        Returns:
            Resultado del seguimiento
        """
        if link_id not in self.links_data:
            logger.warning(f"Enlace no encontrado: {link_id}")
            return {
                "status": "error",
                "message": "Enlace no encontrado"
            }
        
        # Actualizar contadores
        self.links_data[link_id]["clicks"] += 1
        
        # Registrar evento de clic
        click_data = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "user_data": user_data or {}
        }
        
        if "click_history" not in self.links_data[link_id]:
            self.links_data[link_id]["click_history"] = []
        
        self.links_data[link_id]["click_history"].append(click_data)
        
        # Guardar datos
        self._save_links_data()
        
        logger.info(f"Clic registrado para enlace: {link_id}")
        
        return {
            "status": "success",
            "message": "Clic registrado correctamente",
            "link_id": link_id,
            "clicks": self.links_data[link_id]["clicks"]
        }
    
    def record_conversion(self, link_id: str, amount: float, order_id: str,
                         commission: Optional[float] = None, 
                         details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registra una conversión de afiliado.
        
        Args:
            link_id: ID del enlace
            amount: Monto de la venta
            order_id: ID de la orden
            commission: Comisión (opcional)
            details: Detalles adicionales (opcional)
            
        Returns:
            Resultado del registro
        """
        if link_id not in self.links_data:
            logger.warning(f"Enlace no encontrado: {link_id}")
            return {
                "status": "error",
                "message": "Enlace no encontrado"
            }
        
        link_data = self.links_data[link_id]
        program_id = link_data["program_id"]
        
        if program_id not in self.affiliate_programs:
            logger.warning(f"Programa de afiliados no encontrado: {program_id}")
            return {
                "status": "error",
                "message": "Programa de afiliados no encontrado"
            }
        
        program = self.affiliate_programs[program_id]
        
        # Calcular comisión si no se proporciona
        if commission is None:
            commission_rate = program.get("commission_rate", 0.1)  # 10% por defecto
            commission = amount * commission_rate
        
        # Actualizar datos del enlace
        self.links_data[link_id]["conversions"] += 1
        self.links_data[link_id]["revenue"] += commission
        
        # Crear registro de conversión
        conversion_data = {
            "conversion_id": f"{link_id}_{order_id}",
            "link_id": link_id,
            "order_id": order_id,
            "amount": amount,
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "status": "pending"  # pending, approved, rejected
        }
        
        # Guardar conversión
        if link_id not in self.conversions:
            self.conversions[link_id] = []
        
        self.conversions[link_id].append(conversion_data)
        
        # Procesar comisiones de segundo nivel si están habilitadas
        if program.get("second_tier_enabled", False):
            self._process_second_tier_commission(link_data, commission)
        
        # Guardar datos
        self._save_links_data()
        self._save_conversions()
        
        logger.info(f"Conversión registrada para enlace: {link_id}, monto: {amount}, comisión: {commission}")
        
        return {
            "status": "success",
            "message": "Conversión registrada correctamente",
            "conversion_id": conversion_data["conversion_id"],
            "commission": commission
        }
    
    def _process_second_tier_commission(self, link_data: Dict[str, Any], commission: float) -> None:
        """
        Procesa comisiones de segundo nivel.
        
        Args:
            link_data: Datos del enlace
            commission: Comisión de primer nivel
        """
        channel_id = link_data["channel_id"]
        program_id = link_data["program_id"]
        
        # Verificar si el canal tiene un referidor
        if channel_id in self.second_tier and "referrer_id" in self.second_tier[channel_id]:
            referrer_id = self.second_tier[channel_id]["referrer_id"]
            
            # Obtener tasa de comisión de segundo nivel
            program = self.affiliate_programs[program_id]
            second_tier_rate = program.get("second_tier_rate", 0.1)  # 10% por defecto
            
            # Calcular comisión de segundo nivel
            second_tier_commission = commission * second_tier_rate
            
            # Actualizar datos del referidor
            if referrer_id not in self.second_tier:
                self.second_tier[referrer_id] = {
                    "total_commission": 0.0,
                    "referrals": []
                }
            
            self.second_tier[referrer_id]["total_commission"] += second_tier_commission
            
            # Registrar comisión
            commission_data = {
                "timestamp": datetime.now().isoformat(),
                "referred_channel": channel_id,
                "program_id": program_id,
                "original_commission": commission,
                "second_tier_commission": second_tier_commission
            }
            
            if "commissions" not in self.second_tier[referrer_id]:
                self.second_tier[referrer_id]["commissions"] = []
            
            self.second_tier[referrer_id]["commissions"].append(commission_data)
            
            # Guardar datos
            self._save_second_tier()
            
            logger.info(f"Comisión de segundo nivel procesada: {second_tier_commission} para referidor: {referrer_id}")
    
    def register_referral(self, channel_id: str, referrer_id: str) -> Dict[str, Any]:
        """
        Registra una relación de referido para comisiones de segundo nivel.
        
        Args:
            channel_id: ID del canal referido
            referrer_id: ID del canal referidor
            
        Returns:
            Resultado del registro
        """
        # Evitar auto-referencia
        if channel_id == referrer_id:
            return {
                "status": "error",
                "message": "Un canal no puede referirse a sí mismo"
            }
        
        # Registrar relación
        if channel_id not in self.second_tier:
            self.second_tier[channel_id] = {}
        
        self.second_tier[channel_id]["referrer_id"] = referrer_id
        self.second_tier[channel_id]["registered_at"] = datetime.now().isoformat()
        
        # Actualizar lista de referidos del referidor
        if referrer_id not in self.second_tier:
            self.second_tier[referrer_id] = {
                "total_commission": 0.0,
                "referrals": []
            }
        
        if "referrals" not in self.second_tier[referrer_id]:
            self.second_tier[referrer_id]["referrals"] = []
        
        self.second_tier[referrer_id]["referrals"].append({
            "channel_id": channel_id,
            "registered_at": datetime.now().isoformat()
        })
        
        # Guardar datos
        self._save_second_tier()
        
        logger.info(f"Referido registrado: {channel_id} referido por {referrer_id}")
        
        return {
            "status": "success",
            "message": "Referido registrado correctamente",
            "channel_id": channel_id,
            "referrer_id": referrer_id
        }
    
    def get_affiliate_performance(self, channel_id: Optional[str] = None, 
                                program_id: Optional[str] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento de afiliados.
        
        Args:
            channel_id: Filtrar por canal (opcional)
            program_id: Filtrar por programa (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Métricas de rendimiento
        """
        # Convertir fechas si se proporcionan
        start_datetime = None
        end_datetime = None
        
        if start_date:
            start_datetime = datetime.fromisoformat(start_date)
        
        if end_date:
            end_datetime = datetime.fromisoformat(end_date)
        else:
            end_datetime = datetime.now()
        
        # Si no se proporciona fecha de inicio, usar 30 días atrás
        if not start_datetime:
            start_datetime = end_datetime - timedelta(days=30)
        
        # Filtrar enlaces
        filtered_links = {}
        for link_id, link_data in self.links_data.items():
            # Filtrar por canal
            if channel_id and link_data["channel_id"] != channel_id:
                continue
            
            # Filtrar por programa
            if program_id and link_data["program_id"] != program_id:
                continue
            
            # Añadir enlace filtrado
            filtered_links[link_id] = link_data
        
        # Calcular métricas
        total_clicks = 0
        total_conversions = 0
        total_revenue = 0.0
        conversion_rate = 0.0
        
        for link_data in filtered_links.values():
            total_clicks += link_data["clicks"]
            total_conversions += link_data["conversions"]
            total_revenue += link_data["revenue"]
        
        if total_clicks > 0:
            conversion_rate = (total_conversions / total_clicks) * 100
        
        # Calcular métricas por programa
        program_metrics = {}
        for link_data in filtered_links.values():
            prog_id = link_data["program_id"]
            
            if prog_id not in program_metrics:
                program_metrics[prog_id] = {
                    "clicks": 0,
                    "conversions": 0,
                    "revenue": 0.0,
                    "conversion_rate": 0.0
                }
            
            program_metrics[prog_id]["clicks"] += link_data["clicks"]
            program_metrics[prog_id]["conversions"] += link_data["conversions"]
            program_metrics[prog_id]["revenue"] += link_data["revenue"]
        
        # Calcular tasas de conversión por programa
        for prog_id, metrics in program_metrics.items():
            if metrics["clicks"] > 0:
                metrics["conversion_rate"] = (metrics["conversions"] / metrics["clicks"]) * 100
        
        # Obtener métricas de segundo nivel si se filtra por canal
        second_tier_metrics = None
        if channel_id and channel_id in self.second_tier:
            second_tier_metrics = {
                "total_commission": self.second_tier[channel_id].get("total_commission", 0.0),
                "referrals_count": len(self.second_tier[channel_id].get("referrals", [])),
                "commissions_count": len(self.second_tier[channel_id].get("commissions", []))
            }
        
        # Crear resultado
        result = {
            "period": {
                "start_date": start_datetime.isoformat(),
                "end_date": end_datetime.isoformat()
            },
            "overall": {
                "clicks": total_clicks,
                "conversions": total_conversions,
                "revenue": round(total_revenue, 2),
                "conversion_rate": round(conversion_rate, 2),
                "epc": round(total_revenue / total_clicks, 4) if total_clicks > 0 else 0  # Earnings per click
            },
            "by_program": program_metrics,
            "links_count": len(filtered_links),
            "second_tier": second_tier_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_top_performing_links(self, limit: int = 10, 
                               metric: str = "revenue") -> List[Dict[str, Any]]:
        """
        Obtiene los enlaces con mejor rendimiento.
        
        Args:
            limit: Número máximo de enlaces a devolver
            metric: Métrica para ordenar (revenue, conversions, clicks, conversion_rate)
            
        Returns:
            Lista de enlaces ordenados por rendimiento
        """
        # Validar métrica
        valid_metrics = ["revenue", "conversions", "clicks", "conversion_rate"]
        if metric not in valid_metrics:
            logger.warning(f"Métrica no válida: {metric}")
            metric = "revenue"  # Usar ingresos por defecto
        
        # Calcular tasa de conversión para cada enlace
        for link_id, link_data in self.links_data.items():
            if link_data["clicks"] > 0:
                link_data["conversion_rate"] = (link_data["conversions"] / link_data["clicks"]) * 100
            else:
                link_data["conversion_rate"] = 0.0
        
        # Ordenar enlaces por métrica
        sorted_links = sorted(
            self.links_data.values(),
            key=lambda x: x.get(metric, 0),
            reverse=True
        )
        
        # Limitar resultados
        top_links = sorted_links[:limit]
        
        return top_links
    
    def generate_affiliate_recommendations(self, channel_id: str) -> Dict[str, Any]:
        """
        Genera recomendaciones para optimizar ingresos por afiliados.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Recomendaciones de afiliados
        """
        # Filtrar enlaces del canal
        channel_links = [
            link for link_id, link in self.links_data.items()
            if link["channel_id"] == channel_id
        ]
        
        if not channel_links:
            return {
                "status": "warning",
                "message": "No se encontraron enlaces de afiliados para este canal",
                "recommendations": []
            }
        
        # Calcular métricas
        total_clicks = sum(link["clicks"] for link in channel_links)
        total_conversions = sum(link["conversions"] for link in channel_links)
        total_revenue = sum(link["revenue"] for link in channel_links)
        
        if total_clicks > 0:
            overall_conversion_rate = (total_conversions / total_clicks) * 100
        else:
            overall_conversion_rate = 0.0
        
        # Identificar enlaces con mejor y peor rendimiento
        if channel_links:
            best_link = max(channel_links, key=lambda x: x.get("revenue", 0))
            worst_link = min(channel_links, key=lambda x: x.get("revenue", 0))
        else:
            best_link = None
            worst_link = None
        
        # Generar recomendaciones
        recommendations = []
        
        # Recomendación 1: Mejorar enlaces de bajo rendimiento
        if worst_link and worst_link["clicks"] > 10 and worst_link["conversion_rate"] < 1.0:
            recommendations.append({
                "type": "low_performer",
                "priority": "high",
                "link_id": worst_link["link_id"],
                "message": "Este enlace tiene una tasa de conversión baja.",
                "actions": [
                    "Mejorar la ubicación del enlace en el contenido",
                    "Usar un CTA más persuasivo",
                    "Considerar cambiar el producto promocionado"
                ]
            })
        
        # Recomendación 2: Potenciar enlaces exitosos
        if best_link and best_link["conversion_rate"] > 5.0:
            recommendations.append({
                "type": "high_performer",
                "priority": "medium",
                "link_id": best_link["link_id"],
                "message": "Este enlace tiene un buen rendimiento. Potencia su alcance.",
                "actions": [
                    "Destacar este producto en más contenidos",
                    "Crear contenido específico sobre este producto",
                    "Considerar negociar mejores comisiones con el vendedor"
                ]
            })
        
        # Recomendación 3: Diversificar programas
        program_counts = {}
        for link in channel_links:
            program_id = link["program_id"]
            if program_id not in program_counts:
                program_counts[program_id] = 0
            program_counts[program_id] += 1
        
        if len(program_counts) == 1:
            recommendations.append({
                "type": "diversification",
                "priority": "medium",
                "message": "Estás usando un solo programa de afiliados. Considera diversificar.",
                "actions": [
                    "Explorar programas de afiliados complementarios",
                    "Añadir productos de diferentes categorías",
                    "Probar programas con mayores comisiones"
                ]
            })
        
        # Recomendación 4: Optimizar CTAs
        if total_clicks > 0 and overall_conversion_rate < 2.0:
            recommendations.append({
                "type": "cta_optimization",
                "priority": "high",
                "message": "La tasa de conversión general es baja. Optimiza tus CTAs.",
                "actions": [
                    "Usar CTAs más específicos y relevantes",
                    "Colocar enlaces en momentos estratégicos del contenido",
                    "Añadir testimonios o pruebas sociales junto a los enlaces"
                ]
            })
        
        # Recomendación 5: Segundo nivel
        if channel_id not in self.second_tier or "referrer_id" not in self.second_tier[channel_id]:
            recommendations.append({
                "type": "second_tier",
                "priority": "low",
                "message": "No estás aprovechando las comisiones de segundo nivel.",
                "actions": [
                    "Registrarte como referido de otro canal para obtener mejores tasas",
                    "Invitar a otros creadores a usar tus enlaces de referido",
                    "Crear un programa de incentivos para referidos"
                ]
            })
        
        # Crear resultado
        result = {
            "channel_id": channel_id,
            "metrics": {
                "total_links": len(channel_links),
                "total_clicks": total_clicks,
                "total_conversions": total_conversions,
                "total_revenue": round(total_revenue, 2),
                "conversion_rate": round(overall_conversion_rate, 2)
            },
            "best_performer": best_link["link_id"] if best_link else None,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def update_affiliate_status(self, link_id: str, active: bool) -> Dict[str, Any]:
        """
        Actualiza el estado de un enlace de afiliado.
        
        Args:
            link_id: ID del enlace
            active: Estado activo/inactivo
            
        Returns:
            Resultado de la actualización
        """
        if link_id not in self.links_data:
            logger.warning(f"Enlace no encontrado: {link_id}")
            return {
                "status": "error",
                "message": "Enlace no encontrado"
            }
        
        # Actualizar estado
        self.links_data[link_id]["active"] = active
        self.links_data[link_id]["updated_at"] = datetime.now().isoformat()
        
        # Guardar datos
        self._save_links_data()
        
        logger.info(f"Estado del enlace {link_id} actualizado a: {active}")
        
        return {
            "status": "success",
            "message": f"Estado del enlace actualizado a: {'activo' if active else 'inactivo'}",
            "link_id": link_id,
            "active": active
        }
    
    def sync_affiliate_data(self, program_id: str) -> Dict[str, Any]:
        """
        Sincroniza datos de conversiones con la API del programa de afiliados.
        En una implementación real, se conectaría a las APIs de los programas.
        
        Args:
            program_id: ID del programa de afiliados
            
        Returns:
            Resultado de la sincronización
        """
        if program_id not in self.affiliate_programs:
            logger.warning(f"Programa de afiliados no encontrado: {program_id}")
            return {
                "status": "error",
                "message": "Programa de afiliados no encontrado"
            }
        
        program = self.affiliate_programs[program_id]
        
        # Verificar si hay API configurada
        if not program.get("api_url"):
            logger.warning(f"API no configurada para programa: {program_id}")
            return {
                "status": "error",
                "message": "API no configurada para el programa"
            }
        
        # En una implementación real, aquí se conectaría a la API
        # Simulación de sincronización
        try:
            # Simular respuesta de API
            synced_conversions = 0
            updated_conversions = 0
            
            # Actualizar estado de conversiones pendientes
            for link_id, conversions in self.conversions.items():
                                link_data = self.links_data.get(link_id)
                if not link_data or link_data["program_id"] != program_id:
                    continue
                
                for i, conversion in enumerate(conversions):
                    if conversion["status"] == "pending":
                        # Simular verificación con API
                        # En producción, se consultaría el estado real
                        new_status = random.choice(["approved", "rejected", "pending"])
                        
                        if new_status != "pending":
                            self.conversions[link_id][i]["status"] = new_status
                            updated_conversions += 1
                            
                            # Si es aprobada, no hacer nada (ya se contabilizó al registrar)
                            # Si es rechazada, revertir comisión
                            if new_status == "rejected":
                                commission = conversion["commission"]
                                self.links_data[link_id]["revenue"] -= commission
                                self.links_data[link_id]["conversions"] -= 1
                                
                                # Revertir comisión de segundo nivel si aplica
                                if program.get("second_tier_enabled", False):
                                    channel_id = link_data["channel_id"]
                                    if channel_id in self.second_tier and "referrer_id" in self.second_tier[channel_id]:
                                        referrer_id = self.second_tier[channel_id]["referrer_id"]
                                        second_tier_rate = program.get("second_tier_rate", 0.1)
                                        second_tier_commission = commission * second_tier_rate
                                        
                                        if referrer_id in self.second_tier:
                                            self.second_tier[referrer_id]["total_commission"] -= second_tier_commission
                
                synced_conversions += len(conversions)
            
            # Guardar datos actualizados
            self._save_links_data()
            self._save_conversions()
            self._save_second_tier()
            
            logger.info(f"Sincronización completada para programa: {program_id}")
            
            return {
                "status": "success",
                "message": "Sincronización completada",
                "program_id": program_id,
                "synced_conversions": synced_conversions,
                "updated_conversions": updated_conversions
            }
            
        except Exception as e:
            logger.error(f"Error al sincronizar datos: {e}")
            return {
                "status": "error",
                "message": f"Error al sincronizar datos: {str(e)}"
            }
    
    def get_conversion_details(self, conversion_id: str) -> Dict[str, Any]:
        """
        Obtiene detalles de una conversión específica.
        
        Args:
            conversion_id: ID de la conversión
            
        Returns:
            Detalles de la conversión
        """
        # Buscar conversión en todos los enlaces
        for link_id, conversions in self.conversions.items():
            for conversion in conversions:
                if conversion.get("conversion_id") == conversion_id:
                    # Añadir datos del enlace
                    result = conversion.copy()
                    
                    if link_id in self.links_data:
                        link_data = self.links_data[link_id]
                        result["program_id"] = link_data.get("program_id")
                        result["product_id"] = link_data.get("product_id")
                        result["channel_id"] = link_data.get("channel_id")
                        result["campaign"] = link_data.get("campaign")
                    
                    return result
        
        logger.warning(f"Conversión no encontrada: {conversion_id}")
        return {
            "status": "error",
            "message": "Conversión no encontrada"
        }
    
    def get_second_tier_earnings(self, channel_id: str) -> Dict[str, Any]:
        """
        Obtiene ganancias de comisiones de segundo nivel.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Datos de ganancias de segundo nivel
        """
        if channel_id not in self.second_tier:
            return {
                "status": "warning",
                "message": "No se encontraron datos de segundo nivel para este canal",
                "earnings": 0.0,
                "referrals": []
            }
        
        channel_data = self.second_tier[channel_id]
        total_commission = channel_data.get("total_commission", 0.0)
        referrals = channel_data.get("referrals", [])
        commissions = channel_data.get("commissions", [])
        
        # Calcular estadísticas por referido
        referral_stats = {}
        for commission in commissions:
            referred_channel = commission.get("referred_channel")
            if referred_channel not in referral_stats:
                referral_stats[referred_channel] = {
                    "total_commission": 0.0,
                    "commissions_count": 0
                }
            
            referral_stats[referred_channel]["total_commission"] += commission.get("second_tier_commission", 0.0)
            referral_stats[referred_channel]["commissions_count"] += 1
        
        # Crear lista de referidos con estadísticas
        referrals_with_stats = []
        for referral in referrals:
            referred_channel = referral.get("channel_id")
            stats = referral_stats.get(referred_channel, {
                "total_commission": 0.0,
                "commissions_count": 0
            })
            
            referrals_with_stats.append({
                "channel_id": referred_channel,
                "registered_at": referral.get("registered_at"),
                "total_commission": stats["total_commission"],
                "commissions_count": stats["commissions_count"]
            })
        
        # Ordenar por comisión total
        referrals_with_stats.sort(key=lambda x: x["total_commission"], reverse=True)
        
        return {
            "status": "success",
            "channel_id": channel_id,
            "total_earnings": round(total_commission, 2),
            "referrals_count": len(referrals),
            "commissions_count": len(commissions),
            "referrals": referrals_with_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_affiliate_data(self, channel_id: Optional[str] = None, 
                            format: str = "json") -> Dict[str, Any]:
        """
        Exporta datos de afiliados.
        
        Args:
            channel_id: ID del canal para filtrar (opcional)
            format: Formato de exportación (json, csv)
            
        Returns:
            Resultado de la exportación
        """
        # Crear directorio de exportación
        export_dir = "exports/affiliate"
        os.makedirs(export_dir, exist_ok=True)
        
        # Filtrar datos por canal si se especifica
        filtered_links = {}
        filtered_conversions = {}
        
        for link_id, link_data in self.links_data.items():
            if channel_id is None or link_data.get("channel_id") == channel_id:
                filtered_links[link_id] = link_data
                
                if link_id in self.conversions:
                    filtered_conversions[link_id] = self.conversions[link_id]
        
        # Nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{export_dir}/affiliate_data_{timestamp}"
        
        if channel_id:
            filename_base += f"_{channel_id}"
        
        # Exportar según formato
        if format.lower() == "json":
            export_data = {
                "links": filtered_links,
                "conversions": filtered_conversions,
                "exported_at": datetime.now().isoformat()
            }
            
            if channel_id and channel_id in self.second_tier:
                export_data["second_tier"] = self.second_tier[channel_id]
            
            filename = f"{filename_base}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "status": "success",
                "message": "Datos exportados en formato JSON",
                "file": filename
            }
            
        elif format.lower() == "csv":
            # Exportar enlaces
            links_filename = f"{filename_base}_links.csv"
            with open(links_filename, "w", encoding="utf-8", newline="") as f:
                if filtered_links:
                    # Obtener todas las claves posibles
                    all_keys = set()
                    for link_data in filtered_links.values():
                        all_keys.update(link_data.keys())
                    
                    # Escribir CSV
                    import csv
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    
                    for link_data in filtered_links.values():
                        # Asegurar que todas las claves estén presentes
                        row = {key: link_data.get(key, "") for key in all_keys}
                        writer.writerow(row)
            
            # Exportar conversiones
            conversions_filename = f"{filename_base}_conversions.csv"
            with open(conversions_filename, "w", encoding="utf-8", newline="") as f:
                all_conversions = []
                for link_conversions in filtered_conversions.values():
                    all_conversions.extend(link_conversions)
                
                if all_conversions:
                    # Obtener todas las claves posibles
                    all_keys = set()
                    for conversion in all_conversions:
                        all_keys.update(conversion.keys())
                    
                    # Escribir CSV
                    import csv
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    
                    for conversion in all_conversions:
                        # Asegurar que todas las claves estén presentes
                        row = {key: conversion.get(key, "") for key in all_keys}
                        writer.writerow(row)
            
            return {
                "status": "success",
                "message": "Datos exportados en formato CSV",
                "files": [links_filename, conversions_filename]
            }
            
        else:
            return {
                "status": "error",
                "message": f"Formato no soportado: {format}. Formatos disponibles: json, csv"
            }
    
    def import_affiliate_data(self, file_path: str) -> Dict[str, Any]:
        """
        Importa datos de afiliados desde un archivo JSON.
        
        Args:
            file_path: Ruta al archivo de importación
            
        Returns:
            Resultado de la importación
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "message": f"Archivo no encontrado: {file_path}"
                }
            
            # Verificar extensión
            if not file_path.endswith(".json"):
                return {
                    "status": "error",
                    "message": "Solo se admiten archivos JSON para importación"
                }
            
            # Cargar datos
            with open(file_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)
            
            # Verificar estructura
            if "links" not in import_data:
                return {
                    "status": "error",
                    "message": "Formato de archivo inválido: faltan datos de enlaces"
                }
            
            # Importar enlaces
            links_count = 0
            for link_id, link_data in import_data["links"].items():
                self.links_data[link_id] = link_data
                links_count += 1
            
            # Importar conversiones si existen
            conversions_count = 0
            if "conversions" in import_data:
                for link_id, conversions in import_data["conversions"].items():
                    self.conversions[link_id] = conversions
                    conversions_count += len(conversions)
            
            # Importar datos de segundo nivel si existen
            second_tier_updated = False
            if "second_tier" in import_data and "channel_id" in import_data["second_tier"]:
                channel_id = import_data["second_tier"]["channel_id"]
                self.second_tier[channel_id] = import_data["second_tier"]
                second_tier_updated = True
            
            # Guardar datos
            self._save_links_data()
            self._save_conversions()
            if second_tier_updated:
                self._save_second_tier()
            
            logger.info(f"Datos importados: {links_count} enlaces, {conversions_count} conversiones")
            
            return {
                "status": "success",
                "message": "Datos importados correctamente",
                "links_count": links_count,
                "conversions_count": conversions_count,
                "second_tier_updated": second_tier_updated
            }
            
        except Exception as e:
            logger.error(f"Error al importar datos: {e}")
            return {
                "status": "error",
                "message": f"Error al importar datos: {str(e)}"
            }
    
    def delete_affiliate_link(self, link_id: str) -> Dict[str, Any]:
        """
        Elimina un enlace de afiliado y sus conversiones.
        
        Args:
            link_id: ID del enlace
            
        Returns:
            Resultado de la eliminación
        """
        if link_id not in self.links_data:
            logger.warning(f"Enlace no encontrado: {link_id}")
            return {
                "status": "error",
                "message": "Enlace no encontrado"
            }
        
        # Guardar datos para el resultado
        link_data = self.links_data[link_id]
        conversions_count = len(self.conversions.get(link_id, []))
        
        # Eliminar enlace
        del self.links_data[link_id]
        
        # Eliminar conversiones
        if link_id in self.conversions:
            del self.conversions[link_id]
        
        # Guardar datos
        self._save_links_data()
        self._save_conversions()
        
        logger.info(f"Enlace eliminado: {link_id}, conversiones: {conversions_count}")
        
        return {
            "status": "success",
            "message": "Enlace y conversiones eliminados correctamente",
            "link_id": link_id,
            "program_id": link_data.get("program_id"),
            "channel_id": link_data.get("channel_id"),
            "conversions_removed": conversions_count
        }
    
    def batch_update_links(self, channel_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza múltiples enlaces de un canal en lote.
        
        Args:
            channel_id: ID del canal
            updates: Actualizaciones a aplicar
            
        Returns:
            Resultado de la actualización
        """
        # Filtrar enlaces del canal
        channel_links = [
            link_id for link_id, link in self.links_data.items()
            if link.get("channel_id") == channel_id
        ]
        
        if not channel_links:
            return {
                "status": "warning",
                "message": "No se encontraron enlaces para este canal",
                "updated_count": 0
            }
        
        # Aplicar actualizaciones
        updated_count = 0
        for link_id in channel_links:
            updated = False
            
            for key, value in updates.items():
                if key in self.links_data[link_id]:
                    self.links_data[link_id][key] = value
                    updated = True
            
            if updated:
                self.links_data[link_id]["updated_at"] = datetime.now().isoformat()
                updated_count += 1
        
        # Guardar datos si hubo actualizaciones
        if updated_count > 0:
            self._save_links_data()
        
        logger.info(f"Actualización en lote: {updated_count} enlaces actualizados para canal {channel_id}")
        
        return {
            "status": "success",
            "message": f"{updated_count} enlaces actualizados correctamente",
            "channel_id": channel_id,
            "updated_count": updated_count,
            "total_links": len(channel_links)
        }
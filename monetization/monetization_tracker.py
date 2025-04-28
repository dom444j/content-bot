"""
Módulo para el seguimiento de monetización.

Este módulo permite rastrear, analizar y optimizar los ingresos
generados a través de diferentes canales y fuentes de monetización.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/monetization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MonetizationTracker")

class MonetizationTracker:
    """
    Sistema de seguimiento de monetización.
    
    Permite rastrear ingresos de múltiples fuentes, analizar rendimiento,
    y optimizar estrategias de monetización para maximizar ingresos.
    """
    
    def __init__(self):
        """Inicializa el sistema de seguimiento de monetización."""
        self.revenue_entries = self._load_revenue_entries()
        self.revenue_sources = self._load_revenue_sources()
        self.revenue_goals = self._load_revenue_goals()
        self.payout_records = self._load_payout_records()
        
        # Asegurar que existan los directorios necesarios
        os.makedirs("data/monetization", exist_ok=True)
        
        logger.info("MonetizationTracker inicializado")
    
    def _load_revenue_entries(self) -> List[Dict[str, Any]]:
        """Carga entradas de ingresos."""
        try:
            entries_path = "data/monetization/revenue_entries.json"
            if os.path.exists(entries_path):
                with open(entries_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error al cargar entradas de ingresos: {e}")
            return []
    
    def _load_revenue_sources(self) -> Dict[str, Dict[str, Any]]:
        """Carga fuentes de ingresos."""
        try:
            sources_path = "data/monetization/revenue_sources.json"
            if os.path.exists(sources_path):
                with open(sources_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar fuentes de ingresos: {e}")
            return {}
    
    def _load_revenue_goals(self) -> List[Dict[str, Any]]:
        """Carga metas de ingresos."""
        try:
            goals_path = "data/monetization/revenue_goals.json"
            if os.path.exists(goals_path):
                with open(goals_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error al cargar metas de ingresos: {e}")
            return []
    
    def _load_payout_records(self) -> List[Dict[str, Any]]:
        """Carga registros de pagos."""
        try:
            payouts_path = "data/monetization/payout_records.json"
            if os.path.exists(payouts_path):
                with open(payouts_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error al cargar registros de pagos: {e}")
            return []
    
    def _save_revenue_entries(self) -> None:
        """Guarda entradas de ingresos."""
        try:
            entries_path = "data/monetization/revenue_entries.json"
            with open(entries_path, "w", encoding="utf-8") as f:
                json.dump(self.revenue_entries, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar entradas de ingresos: {e}")
    
    def _save_revenue_sources(self) -> None:
        """Guarda fuentes de ingresos."""
        try:
            sources_path = "data/monetization/revenue_sources.json"
            with open(sources_path, "w", encoding="utf-8") as f:
                json.dump(self.revenue_sources, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar fuentes de ingresos: {e}")
    
    def _save_revenue_goals(self) -> None:
        """Guarda metas de ingresos."""
        try:
            goals_path = "data/monetization/revenue_goals.json"
            with open(goals_path, "w", encoding="utf-8") as f:
                json.dump(self.revenue_goals, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar metas de ingresos: {e}")
    
    def _save_payout_records(self) -> None:
        """Guarda registros de pagos."""
        try:
            payouts_path = "data/monetization/payout_records.json"
            with open(payouts_path, "w", encoding="utf-8") as f:
                json.dump(self.payout_records, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar registros de pagos: {e}")
    
    def register_revenue_source(self, 
                               name: str, 
                               source_type: str,
                               platform: Optional[str] = None,
                               description: Optional[str] = None,
                               payment_schedule: Optional[str] = None,
                               payment_threshold: Optional[float] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registra una nueva fuente de ingresos.
        
        Args:
            name: Nombre de la fuente
            source_type: Tipo de fuente (ads, sponsorship, affiliate, subscription, etc.)
            platform: Plataforma asociada (opcional)
            description: Descripción de la fuente (opcional)
            payment_schedule: Programación de pagos (monthly, quarterly, etc.) (opcional)
            payment_threshold: Umbral mínimo para pagos (opcional)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            Datos de la fuente registrada
        """
        # Validar parámetros
        if not name or not source_type:
            logger.error("Parámetros incompletos para registrar fuente")
            return {
                "status": "error",
                "message": "Nombre y tipo son obligatorios"
            }
        
        # Validar tipo de fuente
        valid_source_types = ["ads", "sponsorship", "affiliate", "subscription", 
                             "donation", "merchandise", "content_sale", "token", "other"]
        
        if source_type not in valid_source_types:
            logger.error(f"Tipo de fuente no válido: {source_type}")
            return {
                "status": "error",
                "message": f"Tipo de fuente no válido. Debe ser uno de: {', '.join(valid_source_types)}"
            }
        
        # Generar ID único
        source_id = str(uuid.uuid4())
        
        # Crear fuente
        source = {
            "source_id": source_id,
            "name": name,
            "source_type": source_type,
            "platform": platform,
            "description": description or f"{name} revenue source",
            "payment_schedule": payment_schedule,
            "payment_threshold": payment_threshold,
            "metadata": metadata or {},
            "total_revenue": 0.0,
            "last_revenue_date": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active"  # active, inactive, archived
        }
        
        # Guardar fuente
        self.revenue_sources[source_id] = source
        self._save_revenue_sources()
        
        logger.info(f"Fuente de ingresos registrada: {name} ({source_type})")
        
        return source
    
    def record_revenue(self, 
                      source_id: str, 
                      amount: float,
                      currency: str = "USD",
                      date: Optional[str] = None,
                      content_id: Optional[str] = None,
                      creator_id: Optional[str] = None,
                      details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registra una entrada de ingresos.
        
        Args:
            source_id: ID de la fuente de ingresos
            amount: Cantidad de ingresos
            currency: Moneda (por defecto USD)
            date: Fecha del ingreso (opcional, por defecto ahora)
            content_id: ID del contenido asociado (opcional)
            creator_id: ID del creador asociado (opcional)
            details: Detalles adicionales (opcional)
            
        Returns:
            Datos de la entrada registrada
        """
        # Verificar que la fuente existe
        if source_id not in self.revenue_sources:
            logger.error(f"Fuente de ingresos no encontrada: {source_id}")
            return {
                "status": "error",
                "message": "Fuente de ingresos no encontrada"
            }
        
        # Validar cantidad
                if amount <= 0:
            logger.error(f"Cantidad no válida: {amount}")
            return {
                "status": "error",
                "message": "La cantidad debe ser mayor que cero"
            }
        
        # Usar fecha actual si no se proporciona
        if date is None:
            date = datetime.now().isoformat()
        else:
            # Validar formato de fecha
            try:
                datetime.fromisoformat(date)
            except ValueError:
                logger.error(f"Formato de fecha no válido: {date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha no válido. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                }
        
        # Crear entrada de ingresos
        entry_id = str(uuid.uuid4())
        
        entry = {
            "entry_id": entry_id,
            "source_id": source_id,
            "source_name": self.revenue_sources[source_id]["name"],
            "source_type": self.revenue_sources[source_id]["source_type"],
            "amount": amount,
            "currency": currency,
            "date": date,
            "content_id": content_id,
            "creator_id": creator_id,
            "details": details or {},
            "recorded_at": datetime.now().isoformat()
        }
        
        # Actualizar estadísticas de la fuente
        self.revenue_sources[source_id]["total_revenue"] += amount
        self.revenue_sources[source_id]["last_revenue_date"] = date
        self.revenue_sources[source_id]["updated_at"] = datetime.now().isoformat()
        
        # Guardar entrada y fuente actualizada
        self.revenue_entries.append(entry)
        self._save_revenue_entries()
        self._save_revenue_sources()
        
        # Verificar metas de ingresos
        self._check_revenue_goals()
        
        logger.info(f"Ingreso registrado: {amount} {currency} de {self.revenue_sources[source_id]['name']}")
        
        return entry
    
    def _check_revenue_goals(self) -> None:
        """Verifica el progreso de las metas de ingresos."""
        now = datetime.now()
        
        for goal in self.revenue_goals:
            # Omitir metas completadas o expiradas
            if goal["status"] != "active":
                continue
            
            # Verificar si la meta ha expirado
            if "end_date" in goal and goal["end_date"]:
                try:
                    end_date = datetime.fromisoformat(goal["end_date"])
                    if end_date < now:
                        goal["status"] = "expired"
                        goal["updated_at"] = now.isoformat()
                        logger.info(f"Meta de ingresos expirada: {goal['name']}")
                        continue
                except ValueError:
                    pass
            
            # Calcular progreso actual
            current_amount = self._calculate_goal_progress(goal)
            goal["current_amount"] = current_amount
            
            # Verificar si se ha alcanzado la meta
            if current_amount >= goal["target_amount"]:
                goal["status"] = "completed"
                goal["completion_date"] = now.isoformat()
                goal["updated_at"] = now.isoformat()
                logger.info(f"Meta de ingresos completada: {goal['name']}")
            else:
                goal["updated_at"] = now.isoformat()
        
        # Guardar metas actualizadas
        self._save_revenue_goals()
    
    def _calculate_goal_progress(self, goal: Dict[str, Any]) -> float:
        """
        Calcula el progreso actual hacia una meta de ingresos.
        
        Args:
            goal: Meta de ingresos
            
        Returns:
            Cantidad actual acumulada
        """
        # Filtrar entradas según criterios de la meta
        filtered_entries = []
        
        for entry in self.revenue_entries:
            # Filtrar por fecha de inicio
            if "start_date" in goal and goal["start_date"]:
                try:
                    start_date = datetime.fromisoformat(goal["start_date"])
                    entry_date = datetime.fromisoformat(entry["date"])
                    if entry_date < start_date:
                        continue
                except ValueError:
                    pass
            
            # Filtrar por fecha de fin
            if "end_date" in goal and goal["end_date"]:
                try:
                    end_date = datetime.fromisoformat(goal["end_date"])
                    entry_date = datetime.fromisoformat(entry["date"])
                    if entry_date > end_date:
                        continue
                except ValueError:
                    pass
            
            # Filtrar por fuente
            if "source_id" in goal and goal["source_id"]:
                if entry["source_id"] != goal["source_id"]:
                    continue
            
            # Filtrar por tipo de fuente
            if "source_type" in goal and goal["source_type"]:
                if entry["source_type"] != goal["source_type"]:
                    continue
            
            # Filtrar por creador
            if "creator_id" in goal and goal["creator_id"]:
                if entry["creator_id"] != goal["creator_id"]:
                    continue
            
            # Filtrar por contenido
            if "content_id" in goal and goal["content_id"]:
                if entry["content_id"] != goal["content_id"]:
                    continue
            
            # Añadir entrada filtrada
            filtered_entries.append(entry)
        
        # Sumar cantidades
        total = sum(entry["amount"] for entry in filtered_entries)
        
        return total
    
    def create_revenue_goal(self, 
                           name: str, 
                           target_amount: float,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           source_id: Optional[str] = None,
                           source_type: Optional[str] = None,
                           creator_id: Optional[str] = None,
                           content_id: Optional[str] = None,
                           description: Optional[str] = None) -> Dict[str, Any]:
        """
        Crea una nueva meta de ingresos.
        
        Args:
            name: Nombre de la meta
            target_amount: Cantidad objetivo
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            source_id: ID de la fuente específica (opcional)
            source_type: Tipo de fuente (opcional)
            creator_id: ID del creador (opcional)
            content_id: ID del contenido (opcional)
            description: Descripción de la meta (opcional)
            
        Returns:
            Datos de la meta creada
        """
        # Validar parámetros
        if not name or target_amount <= 0:
            logger.error(f"Parámetros no válidos: {name}, {target_amount}")
            return {
                "status": "error",
                "message": "Nombre y cantidad objetivo válida son obligatorios"
            }
        
        # Validar fechas
        if start_date:
            try:
                datetime.fromisoformat(start_date)
            except ValueError:
                logger.error(f"Formato de fecha de inicio no válido: {start_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha de inicio no válido"
                }
        
        if end_date:
            try:
                datetime.fromisoformat(end_date)
            except ValueError:
                logger.error(f"Formato de fecha de fin no válido: {end_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha de fin no válido"
                }
        
        # Validar fuente si se proporciona
        if source_id and source_id not in self.revenue_sources:
            logger.error(f"Fuente de ingresos no encontrada: {source_id}")
            return {
                "status": "error",
                "message": "Fuente de ingresos no encontrada"
            }
        
        # Generar ID único
        goal_id = str(uuid.uuid4())
        
        # Crear meta
        goal = {
            "goal_id": goal_id,
            "name": name,
            "description": description or f"Meta de ingresos: {name}",
            "target_amount": target_amount,
            "current_amount": 0.0,
            "start_date": start_date,
            "end_date": end_date,
            "source_id": source_id,
            "source_type": source_type,
            "creator_id": creator_id,
            "content_id": content_id,
            "status": "active",  # active, completed, expired
            "completion_date": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Calcular progreso inicial
        current_amount = self._calculate_goal_progress(goal)
        goal["current_amount"] = current_amount
        
        # Verificar si ya se ha alcanzado la meta
        if current_amount >= target_amount:
            goal["status"] = "completed"
            goal["completion_date"] = datetime.now().isoformat()
        
        # Guardar meta
        self.revenue_goals.append(goal)
        self._save_revenue_goals()
        
        logger.info(f"Meta de ingresos creada: {name} ({target_amount})")
        
        return goal
    
    def record_payout(self, 
                     source_id: str, 
                     amount: float,
                     currency: str = "USD",
                     date: Optional[str] = None,
                     transaction_id: Optional[str] = None,
                     payment_method: Optional[str] = None,
                     status: str = "completed",
                     notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Registra un pago recibido de una fuente de ingresos.
        
        Args:
            source_id: ID de la fuente de ingresos
            amount: Cantidad del pago
            currency: Moneda (por defecto USD)
            date: Fecha del pago (opcional, por defecto ahora)
            transaction_id: ID de transacción externa (opcional)
            payment_method: Método de pago (opcional)
            status: Estado del pago (completed, pending, failed)
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos del pago registrado
        """
        # Verificar que la fuente existe
        if source_id not in self.revenue_sources:
            logger.error(f"Fuente de ingresos no encontrada: {source_id}")
            return {
                "status": "error",
                "message": "Fuente de ingresos no encontrada"
            }
        
        # Validar cantidad
        if amount <= 0:
            logger.error(f"Cantidad no válida: {amount}")
            return {
                "status": "error",
                "message": "La cantidad debe ser mayor que cero"
            }
        
        # Validar estado
        valid_statuses = ["completed", "pending", "failed"]
        if status not in valid_statuses:
            logger.error(f"Estado no válido: {status}")
            return {
                "status": "error",
                "message": f"Estado no válido. Debe ser uno de: {', '.join(valid_statuses)}"
            }
        
        # Usar fecha actual si no se proporciona
        if date is None:
            date = datetime.now().isoformat()
        else:
            # Validar formato de fecha
            try:
                datetime.fromisoformat(date)
            except ValueError:
                logger.error(f"Formato de fecha no válido: {date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha no válido. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                }
        
        # Generar ID único
        payout_id = str(uuid.uuid4())
        
        # Crear registro de pago
        payout = {
            "payout_id": payout_id,
            "source_id": source_id,
            "source_name": self.revenue_sources[source_id]["name"],
            "source_type": self.revenue_sources[source_id]["source_type"],
            "amount": amount,
            "currency": currency,
            "date": date,
            "transaction_id": transaction_id,
            "payment_method": payment_method,
            "status": status,
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Guardar pago
        self.payout_records.append(payout)
        self._save_payout_records()
        
        logger.info(f"Pago registrado: {amount} {currency} de {self.revenue_sources[source_id]['name']}")
        
        return payout
    
    def update_payout_status(self, payout_id: str, status: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Actualiza el estado de un pago.
        
        Args:
            payout_id: ID del pago
            status: Nuevo estado (completed, pending, failed)
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos del pago actualizado
        """
        # Validar estado
        valid_statuses = ["completed", "pending", "failed"]
        if status not in valid_statuses:
            logger.error(f"Estado no válido: {status}")
            return {
                "status": "error",
                "message": f"Estado no válido. Debe ser uno de: {', '.join(valid_statuses)}"
            }
        
        # Buscar pago
        for payout in self.payout_records:
            if payout["payout_id"] == payout_id:
                # Actualizar estado
                payout["status"] = status
                payout["updated_at"] = datetime.now().isoformat()
                
                # Actualizar notas si se proporcionan
                if notes:
                    if "status_history" not in payout:
                        payout["status_history"] = []
                    
                    payout["status_history"].append({
                        "status": status,
                        "notes": notes,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Guardar cambios
                self._save_payout_records()
                
                logger.info(f"Estado de pago actualizado: {payout_id} a {status}")
                
                return payout
        
        # Pago no encontrado
        logger.error(f"Pago no encontrado: {payout_id}")
        return {
            "status": "error",
            "message": "Pago no encontrado"
        }
    
    def get_revenue_summary(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           group_by: str = "source_type",
                           creator_id: Optional[str] = None,
                           content_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene un resumen de ingresos.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            group_by: Campo para agrupar (source_type, source_id, date, creator_id, content_id)
            creator_id: Filtrar por creador (opcional)
            content_id: Filtrar por contenido (opcional)
            
        Returns:
            Resumen de ingresos
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
        
        # Validar campo de agrupación
        valid_group_fields = ["source_type", "source_id", "date", "creator_id", "content_id"]
        if group_by not in valid_group_fields:
            logger.error(f"Campo de agrupación no válido: {group_by}")
            return {
                "status": "error",
                "message": f"Campo de agrupación no válido. Debe ser uno de: {', '.join(valid_group_fields)}"
            }
        
        # Filtrar entradas
        filtered_entries = []
        
        for entry in self.revenue_entries:
            # Filtrar por fecha
            try:
                entry_date = datetime.fromisoformat(entry["date"])
                if entry_date < start_date_obj or entry_date > end_date_obj:
                    continue
            except ValueError:
                continue
            
            # Filtrar por creador
            if creator_id and entry.get("creator_id") != creator_id:
                continue
            
            # Filtrar por contenido
            if content_id and entry.get("content_id") != content_id:
                continue
            
            # Añadir entrada filtrada
            filtered_entries.append(entry)
        
        # Agrupar entradas
        grouped_data = defaultdict(float)
        
        for entry in filtered_entries:
            if group_by == "date":
                # Agrupar por día
                try:
                    date_obj = datetime.fromisoformat(entry["date"])
                    group_key = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    group_key = "unknown_date"
            else:
                # Agrupar por campo especificado
                group_key = entry.get(group_by, "unknown")
            
            grouped_data[group_key] += entry["amount"]
        
        # Preparar resumen
        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "group_by": group_by,
            "total_revenue": sum(grouped_data.values()),
            "entry_count": len(filtered_entries),
            "grouped_data": dict(grouped_data)
        }
        
        # Añadir filtros aplicados
        if creator_id:
            summary["creator_id"] = creator_id
        
        if content_id:
            summary["content_id"] = content_id
        
        return summary
    
    def get_revenue_trends(self, 
                          period: str = "monthly",
                          months: int = 12,
                          source_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene tendencias de ingresos a lo largo del tiempo.
        
        Args:
            period: Período de agrupación (daily, weekly, monthly, quarterly, yearly)
            months: Número de meses a analizar
            source_type: Filtrar por tipo de fuente (opcional)
            
        Returns:
            Tendencias de ingresos
        """
        # Validar período
        valid_periods = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if period not in valid_periods:
            logger.error(f"Período no válido: {period}")
            return {
                "status": "error",
                "message": f"Período no válido. Debe ser uno de: {', '.join(valid_periods)}"
            }
        
        # Calcular fecha de inicio
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        
        # Inicializar datos de tendencias
        trends = {
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data": defaultdict(float),
            "source_type": source_type
        }
        
        # Filtrar entradas
        for entry in self.revenue_entries:
            # Filtrar por fecha
            try:
                entry_date = datetime.fromisoformat(entry["date"])
                if entry_date < start_date or entry_date > end_date:
                    continue
            except ValueError:
                continue
            
            # Filtrar por tipo de fuente
            if source_type and entry["source_type"] != source_type:
                continue
            
            # Determinar clave de período
            if period == "daily":
                period_key = entry_date.strftime("%Y-%m-%d")
            elif period == "weekly":
                # Semana del año
                week_num = entry_date.isocalendar()[1]
                period_key = f"{entry_date.year}-W{week_num:02d}"
            elif period == "monthly":
                period_key = entry_date.strftime("%Y-%m")
            elif period == "quarterly":
                quarter = (entry_date.month - 1) // 3 + 1
                period_key = f"{entry_date.year}-Q{quarter}"
            elif period == "yearly":
                period_key = str(entry_date.year)
            
            # Sumar ingresos
            trends["data"][period_key] += entry["amount"]
        
        # Ordenar períodos cronológicamente
        sorted_data = dict(sorted(trends["data"].items()))
        trends["data"] = sorted_data
        
        # Calcular estadísticas
        if sorted_data:
            values = list(sorted_data.values())
            trends["total"] = sum(values)
            trends["average"] = sum(values) / len(values)
            trends["min"] = min(values)
            trends["max"] = max(values)
            
            # Calcular crecimiento
            if len(values) >= 2:
                first_value = values[0]
                last_value = values[-1]
                if first_value > 0:
                    growth = ((last_value - first_value) / first_value) * 100
                    trends["growth_percent"] = growth
        
        return trends
    
    def get_source_performance(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene el rendimiento de una fuente de ingresos específica o todas.
        
        Args:
            source_id: ID de la fuente (opcional, todas si no se especifica)
            
        Returns:
            Datos de rendimiento
        """
        # Inicializar resultado
        performance = {
            "sources": [],
            "total_revenue": 0.0,
            "total_payouts": 0.0,
            "source_count": 0
        }
        
        # Filtrar fuentes
        sources_to_analyze = []
        
        if source_id:
            # Fuente específica
            if source_id in self.revenue_sources:
                sources_to_analyze = [self.revenue_sources[source_id]]
            else:
                logger.error(f"Fuente de ingresos no encontrada: {source_id}")
                return {
                    "status": "error",
                    "message": "Fuente de ingresos no encontrada"
                }
        else:
            # Todas las fuentes
            sources_to_analyze = list(self.revenue_sources.values())
        
        # Analizar cada fuente
        for source in sources_to_analyze:
            source_id = source["source_id"]
            
            # Calcular ingresos totales
            total_revenue = source["total_revenue"]
            
            # Calcular pagos recibidos
            total_payouts = sum(
                payout["amount"] 
                for payout in self.payout_records 
                if payout["source_id"] == source_id and payout["status"] == "completed"
            )
            
            # Calcular pagos pendientes
            pending_payouts = sum(
                payout["amount"] 
                for payout in self.payout_records 
                if payout["source_id"] == source_id and payout["status"] == "pending"
            )
            
            # Calcular ingresos por mes (últimos 3 meses)
            now = datetime.now()
            three_months_ago = now - timedelta(days=90)
            
            monthly_revenue = defaultdict(float)
            
            for entry in self.revenue_entries:
                if entry["source_id"] != source_id:
                    continue
                
                try:
                    entry_date = datetime.fromisoformat(entry["date"])
                    if entry_date < three_months_ago:
                        continue
                    
                    month_key = entry_date.strftime("%Y-%m")
                    monthly_revenue[month_key] += entry["amount"]
                except ValueError:
                    continue
            
            # Calcular promedio mensual
            if monthly_revenue:
                avg_monthly = sum(monthly_revenue.values()) / len(monthly_revenue)
            else:
                avg_monthly = 0.0
            
            # Crear resumen de fuente
            source_summary = {
                "source_id": source_id,
                "name": source["name"],
                "source_type": source["source_type"],
                "platform": source["platform"],
                "status": source["status"],
                "total_revenue": total_revenue,
                "total_payouts": total_payouts,
                "pending_payouts": pending_payouts,
                "balance": total_revenue - total_payouts,
                "monthly_revenue": dict(monthly_revenue),
                "avg_monthly_revenue": avg_monthly,
                "last_revenue_date": source["last_revenue_date"],
                "created_at": source["created_at"]
            }
            
            # Añadir a resultados
            performance["sources"].append(source_summary)
            performance["total_revenue"] += total_revenue
            performance["total_payouts"] += total_payouts
        
        # Actualizar conteo de fuentes
        performance["source_count"] = len(performance["sources"])
        
        # Ordenar fuentes por ingresos totales
        performance["sources"].sort(key=lambda x: x["total_revenue"], reverse=True)
        
        return performance
    
    def get_revenue_goals_status(self, include_completed: bool = True) -> Dict[str, Any]:
        """
        Obtiene el estado de todas las metas de ingresos.
        
        Args:
            include_completed: Incluir metas completadas (por defecto True)
            
        Returns:
            Estado de las metas
        """
        # Inicializar resultado
        goals_status = {
            "active_goals": [],
            "completed_goals": [],
            "expired_goals": [],
            "total_goals": len(self.revenue_goals),
            "active_count": 0,
            "completed_count": 0,
            "expired_count": 0
        }
        
        # Actualizar metas antes de devolver estado
        self._check_revenue_goals()
        
        # Procesar cada meta
        for goal in self.revenue_goals:
            # Calcular porcentaje de progreso
            if goal["target_amount"] > 0:
                progress_percent = (goal["current_amount"] / goal["target_amount"]) * 100
            else:
                progress_percent = 0
            
            # Crear resumen de meta
            goal_summary = {
                "goal_id": goal["goal_id"],
                "name": goal["name"],
                "description": goal["description"],
                "target_amount": goal["target_amount"],
                "current_amount": goal["current_amount"],
                "progress_percent": progress_percent,
                "status": goal["status"],
                "start_date": goal["start_date"],
                "end_date": goal["end_date"],
                "created_at": goal["created_at"]
            }
            
            # Añadir a la categoría correspondiente
            if goal["status"] == "active":
                goals_status["active_goals"].append(goal_summary)
                goals_status["active_count"] += 1
            elif goal["status"] == "completed":
                if include_completed:
                    goals_status["completed_goals"].append(goal_summary)
                goals_status["completed_count"] += 1
            elif goal["status"] == "expired":
                goals_status["expired_goals"].append(goal_summary)
                goals_status["expired_count"] += 1
        
        # Ordenar metas activas por progreso
        goals_status["active_goals"].sort(key=lambda x: x["progress_percent"], reverse=True)
        
        # Ordenar metas completadas por fecha de completado
        goals_status["completed_goals"].sort(
            key=lambda x: datetime.fromisoformat(self.revenue_goals[
                next(i for i, g in enumerate(self.revenue_goals) if g["goal_id"] == x["goal_id"])
            ]["completion_date"]),
            reverse=True
        )
        
        return goals_status
    
    def get_content_performance(self, content_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Obtiene el rendimiento de monetización por contenido.
        
        Args:
            content_ids: Lista de IDs de contenido (opcional, todos si no se especifica)
            
        Returns:
            Rendimiento por contenido
        """
        # Inicializar resultado
        performance = {
            "content_items": [],
            "total_revenue": 0.0,
            "content_count": 0
        }
        
        # Agrupar ingresos por contenido
        content_revenue = defaultdict(float)
        content_sources = defaultdict(set)
        content_entries = defaultdict(list)
        
        for entry in self.revenue_entries:
            content_id = entry.get("content_id")
            if not content_id:
                continue
            
            # Filtrar por contenidos específicos si se proporcionan
            if content_ids and content_id not in content_ids:
                continue
            
            # Sumar ingresos
            content_revenue[content_id] += entry["amount"]
            
            # Registrar fuente
            content_sources[content_id].add(entry["source_type"])
            
            # Guardar entrada
            content_entries[content_id].append({
                "entry_id": entry["entry_id"],
                "amount": entry["amount"],
                "source_type": entry["source_type"],
                "date": entry["date"]
            })
        
        # Procesar cada contenido
        for content_id, revenue in content_revenue.items():
                        # Ordenar entradas por fecha
            sorted_entries = sorted(
                content_entries[content_id],
                key=lambda x: datetime.fromisoformat(x["date"])
            )
            
            # Calcular primera y última fecha
            if sorted_entries:
                first_date = datetime.fromisoformat(sorted_entries[0]["date"]).strftime("%Y-%m-%d")
                last_date = datetime.fromisoformat(sorted_entries[-1]["date"]).strftime("%Y-%m-%d")
            else:
                first_date = None
                last_date = None
            
            # Calcular fuentes de ingresos
            source_breakdown = defaultdict(float)
            for entry in sorted_entries:
                source_breakdown[entry["source_type"]] += entry["amount"]
            
            # Crear resumen de contenido
            content_summary = {
                "content_id": content_id,
                "total_revenue": revenue,
                "entry_count": len(sorted_entries),
                "first_revenue_date": first_date,
                "last_revenue_date": last_date,
                "source_types": list(content_sources[content_id]),
                "source_breakdown": dict(source_breakdown),
                "recent_entries": sorted_entries[-5:] if len(sorted_entries) > 5 else sorted_entries
            }
            
            # Añadir a resultados
            performance["content_items"].append(content_summary)
            performance["total_revenue"] += revenue
        
        # Actualizar conteo de contenidos
        performance["content_count"] = len(performance["content_items"])
        
        # Ordenar contenidos por ingresos totales
        performance["content_items"].sort(key=lambda x: x["total_revenue"], reverse=True)
        
        return performance
    
    def get_creator_performance(self, creator_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Obtiene el rendimiento de monetización por creador.
        
        Args:
            creator_ids: Lista de IDs de creador (opcional, todos si no se especifica)
            
        Returns:
            Rendimiento por creador
        """
        # Inicializar resultado
        performance = {
            "creators": [],
            "total_revenue": 0.0,
            "creator_count": 0
        }
        
        # Agrupar ingresos por creador
        creator_revenue = defaultdict(float)
        creator_sources = defaultdict(set)
        creator_content = defaultdict(set)
        creator_entries = defaultdict(list)
        
        for entry in self.revenue_entries:
            creator_id = entry.get("creator_id")
            if not creator_id:
                continue
            
            # Filtrar por creadores específicos si se proporcionan
            if creator_ids and creator_id not in creator_ids:
                continue
            
            # Sumar ingresos
            creator_revenue[creator_id] += entry["amount"]
            
            # Registrar fuente
            creator_sources[creator_id].add(entry["source_type"])
            
            # Registrar contenido
            if entry.get("content_id"):
                creator_content[creator_id].add(entry["content_id"])
            
            # Guardar entrada
            creator_entries[creator_id].append({
                "entry_id": entry["entry_id"],
                "amount": entry["amount"],
                "source_type": entry["source_type"],
                "content_id": entry.get("content_id"),
                "date": entry["date"]
            })
        
        # Procesar cada creador
        for creator_id, revenue in creator_revenue.items():
            # Ordenar entradas por fecha
            sorted_entries = sorted(
                creator_entries[creator_id],
                key=lambda x: datetime.fromisoformat(x["date"])
            )
            
            # Calcular primera y última fecha
            if sorted_entries:
                first_date = datetime.fromisoformat(sorted_entries[0]["date"]).strftime("%Y-%m-%d")
                last_date = datetime.fromisoformat(sorted_entries[-1]["date"]).strftime("%Y-%m-%d")
            else:
                first_date = None
                last_date = None
            
            # Calcular fuentes de ingresos
            source_breakdown = defaultdict(float)
            for entry in sorted_entries:
                source_breakdown[entry["source_type"]] += entry["amount"]
            
            # Calcular ingresos por contenido
            content_breakdown = defaultdict(float)
            for entry in sorted_entries:
                if entry.get("content_id"):
                    content_breakdown[entry["content_id"]] += entry["amount"]
            
            # Crear resumen de creador
            creator_summary = {
                "creator_id": creator_id,
                "total_revenue": revenue,
                "entry_count": len(sorted_entries),
                "content_count": len(creator_content[creator_id]),
                "first_revenue_date": first_date,
                "last_revenue_date": last_date,
                "source_types": list(creator_sources[creator_id]),
                "source_breakdown": dict(source_breakdown),
                "top_content": dict(sorted(content_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]),
                "recent_entries": sorted_entries[-5:] if len(sorted_entries) > 5 else sorted_entries
            }
            
            # Añadir a resultados
            performance["creators"].append(creator_summary)
            performance["total_revenue"] += revenue
        
        # Actualizar conteo de creadores
        performance["creator_count"] = len(performance["creators"])
        
        # Ordenar creadores por ingresos totales
        performance["creators"].sort(key=lambda x: x["total_revenue"], reverse=True)
        
        return performance
    
    def update_revenue_source(self, 
                             source_id: str, 
                             name: Optional[str] = None,
                             description: Optional[str] = None,
                             payment_schedule: Optional[str] = None,
                             payment_threshold: Optional[float] = None,
                             status: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Actualiza una fuente de ingresos existente.
        
        Args:
            source_id: ID de la fuente
            name: Nuevo nombre (opcional)
            description: Nueva descripción (opcional)
            payment_schedule: Nueva programación de pagos (opcional)
            payment_threshold: Nuevo umbral de pagos (opcional)
            status: Nuevo estado (active, inactive, archived) (opcional)
            metadata: Nuevos metadatos (opcional)
            
        Returns:
            Datos de la fuente actualizada
        """
        # Verificar que la fuente existe
        if source_id not in self.revenue_sources:
            logger.error(f"Fuente de ingresos no encontrada: {source_id}")
            return {
                "status": "error",
                "message": "Fuente de ingresos no encontrada"
            }
        
        # Validar estado si se proporciona
        if status:
            valid_statuses = ["active", "inactive", "archived"]
            if status not in valid_statuses:
                logger.error(f"Estado no válido: {status}")
                return {
                    "status": "error",
                    "message": f"Estado no válido. Debe ser uno de: {', '.join(valid_statuses)}"
                }
        
        # Obtener fuente actual
        source = self.revenue_sources[source_id]
        
        # Actualizar campos
        if name:
            source["name"] = name
        
        if description:
            source["description"] = description
        
        if payment_schedule:
            source["payment_schedule"] = payment_schedule
        
        if payment_threshold is not None:
            source["payment_threshold"] = payment_threshold
        
        if status:
            source["status"] = status
        
        if metadata:
            # Actualizar metadatos existentes en lugar de reemplazarlos
            if "metadata" not in source:
                source["metadata"] = {}
            
            source["metadata"].update(metadata)
        
        # Actualizar timestamp
        source["updated_at"] = datetime.now().isoformat()
        
        # Guardar cambios
        self._save_revenue_sources()
        
        logger.info(f"Fuente de ingresos actualizada: {source['name']} ({source_id})")
        
        return source
    
    def update_revenue_goal(self, 
                           goal_id: str,
                           name: Optional[str] = None,
                           description: Optional[str] = None,
                           target_amount: Optional[float] = None,
                           end_date: Optional[str] = None,
                           status: Optional[str] = None) -> Dict[str, Any]:
        """
        Actualiza una meta de ingresos existente.
        
        Args:
            goal_id: ID de la meta
            name: Nuevo nombre (opcional)
            description: Nueva descripción (opcional)
            target_amount: Nueva cantidad objetivo (opcional)
            end_date: Nueva fecha de fin (opcional)
            status: Nuevo estado (active, completed, expired) (opcional)
            
        Returns:
            Datos de la meta actualizada
        """
        # Buscar meta
        goal_index = None
        for i, goal in enumerate(self.revenue_goals):
            if goal["goal_id"] == goal_id:
                goal_index = i
                break
        
        if goal_index is None:
            logger.error(f"Meta de ingresos no encontrada: {goal_id}")
            return {
                "status": "error",
                "message": "Meta de ingresos no encontrada"
            }
        
        # Validar estado si se proporciona
        if status:
            valid_statuses = ["active", "completed", "expired"]
            if status not in valid_statuses:
                logger.error(f"Estado no válido: {status}")
                return {
                    "status": "error",
                    "message": f"Estado no válido. Debe ser uno de: {', '.join(valid_statuses)}"
                }
        
        # Validar fecha si se proporciona
        if end_date:
            try:
                datetime.fromisoformat(end_date)
            except ValueError:
                logger.error(f"Formato de fecha no válido: {end_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha no válido. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                }
        
        # Obtener meta actual
        goal = self.revenue_goals[goal_index]
        
        # Actualizar campos
        if name:
            goal["name"] = name
        
        if description:
            goal["description"] = description
        
        if target_amount is not None and target_amount > 0:
            goal["target_amount"] = target_amount
            
            # Verificar si se ha alcanzado la meta con el nuevo objetivo
            if goal["current_amount"] >= target_amount and goal["status"] == "active":
                goal["status"] = "completed"
                goal["completion_date"] = datetime.now().isoformat()
        
        if end_date:
            goal["end_date"] = end_date
            
            # Verificar si la meta ha expirado con la nueva fecha
            try:
                end_date_obj = datetime.fromisoformat(end_date)
                if end_date_obj < datetime.now() and goal["status"] == "active":
                    goal["status"] = "expired"
            except ValueError:
                pass
        
        if status:
            goal["status"] = status
            
            # Si se marca como completada, establecer fecha de completado
            if status == "completed" and not goal.get("completion_date"):
                goal["completion_date"] = datetime.now().isoformat()
        
        # Actualizar timestamp
        goal["updated_at"] = datetime.now().isoformat()
        
        # Guardar cambios
        self._save_revenue_goals()
        
        logger.info(f"Meta de ingresos actualizada: {goal['name']} ({goal_id})")
        
        return goal
    
    def delete_revenue_source(self, source_id: str, archive_instead: bool = True) -> Dict[str, Any]:
        """
        Elimina o archiva una fuente de ingresos.
        
        Args:
            source_id: ID de la fuente
            archive_instead: Archivar en lugar de eliminar (por defecto True)
            
        Returns:
            Resultado de la operación
        """
        # Verificar que la fuente existe
        if source_id not in self.revenue_sources:
            logger.error(f"Fuente de ingresos no encontrada: {source_id}")
            return {
                "status": "error",
                "message": "Fuente de ingresos no encontrada"
            }
        
        # Obtener nombre para el log
        source_name = self.revenue_sources[source_id]["name"]
        
        if archive_instead:
            # Archivar fuente
            self.revenue_sources[source_id]["status"] = "archived"
            self.revenue_sources[source_id]["updated_at"] = datetime.now().isoformat()
            self._save_revenue_sources()
            
            logger.info(f"Fuente de ingresos archivada: {source_name} ({source_id})")
            
            return {
                "status": "success",
                "message": f"Fuente de ingresos archivada: {source_name}",
                "source_id": source_id
            }
        else:
            # Eliminar fuente
            del self.revenue_sources[source_id]
            self._save_revenue_sources()
            
            logger.info(f"Fuente de ingresos eliminada: {source_name} ({source_id})")
            
            return {
                "status": "success",
                "message": f"Fuente de ingresos eliminada: {source_name}",
                "source_id": source_id
            }
    
    def delete_revenue_goal(self, goal_id: str) -> Dict[str, Any]:
        """
        Elimina una meta de ingresos.
        
        Args:
            goal_id: ID de la meta
            
        Returns:
            Resultado de la operación
        """
        # Buscar meta
        goal_index = None
        goal_name = None
        
        for i, goal in enumerate(self.revenue_goals):
            if goal["goal_id"] == goal_id:
                goal_index = i
                goal_name = goal["name"]
                break
        
        if goal_index is None:
            logger.error(f"Meta de ingresos no encontrada: {goal_id}")
            return {
                "status": "error",
                "message": "Meta de ingresos no encontrada"
            }
        
        # Eliminar meta
        self.revenue_goals.pop(goal_index)
        self._save_revenue_goals()
        
        logger.info(f"Meta de ingresos eliminada: {goal_name} ({goal_id})")
        
        return {
            "status": "success",
            "message": f"Meta de ingresos eliminada: {goal_name}",
            "goal_id": goal_id
        }
    
    def export_revenue_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           format_type: str = "json") -> Dict[str, Any]:
        """
        Exporta datos de ingresos para análisis externo.
        
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
            # Por defecto, último año
            start_date_obj = datetime.now() - timedelta(days=365)
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
        
        # Filtrar entradas por fecha
        filtered_entries = []
        
        for entry in self.revenue_entries:
            try:
                entry_date = datetime.fromisoformat(entry["date"])
                if entry_date >= start_date_obj and entry_date <= end_date_obj:
                    filtered_entries.append(entry)
            except ValueError:
                continue
        
        # Crear directorio de exportación
        export_dir = "data/monetization/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"revenue_export_{timestamp}.{format_type}"
        filepath = os.path.join(export_dir, filename)
        
        # Exportar datos
        if format_type == "json":
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "entry_count": len(filtered_entries),
                    "total_revenue": sum(entry["amount"] for entry in filtered_entries)
                },
                "entries": filtered_entries,
                "sources": self.revenue_sources
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
        
        elif format_type == "csv":
            import csv
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                # Definir campos
                fieldnames = [
                    "entry_id", "source_id", "source_name", "source_type", 
                    "amount", "currency", "date", "content_id", "creator_id", 
                    "recorded_at"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Escribir entradas
                for entry in filtered_entries:
                    # Crear fila con solo los campos definidos
                    row = {field: entry.get(field, "") for field in fieldnames}
                    writer.writerow(row)
        
        logger.info(f"Datos de ingresos exportados: {filepath}")
        
        return {
            "status": "success",
            "message": f"Datos exportados a {filepath}",
            "filepath": filepath,
            "entry_count": len(filtered_entries),
            "total_revenue": sum(entry["amount"] for entry in filtered_entries),
            "format": format_type
        }
                
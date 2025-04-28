"""
Módulo para gestionar un marketplace B2B para creadores de contenido.
Permite conectar creadores con marcas, gestionar colaboraciones y monetizar audiencias.
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid
from collections import defaultdict

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class B2BMarketplace:
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el marketplace B2B.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Cargar datos
        self.creators = self._load_creators()
        self.brands = self._load_brands()
        self.deals = self._load_deals()
        self.proposals = self._load_proposals()
        self.reviews = self._load_reviews()
        
        # Crear directorios si no existen
        os.makedirs("data/marketplace", exist_ok=True)
    
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
    
    def _load_creators(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de creadores."""
        try:
            creators_path = "data/marketplace/creators.json"
            if os.path.exists(creators_path):
                with open(creators_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar creadores: {e}")
            return {}
    
    def _load_brands(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de marcas."""
        try:
            brands_path = "data/marketplace/brands.json"
            if os.path.exists(brands_path):
                with open(brands_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar marcas: {e}")
            return {}
    
    def _load_deals(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de acuerdos."""
        try:
            deals_path = "data/marketplace/deals.json"
            if os.path.exists(deals_path):
                with open(deals_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar acuerdos: {e}")
            return {}
    
    def _load_proposals(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de propuestas."""
        try:
            proposals_path = "data/marketplace/proposals.json"
            if os.path.exists(proposals_path):
                with open(proposals_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar propuestas: {e}")
            return {}
    
        def _load_reviews(self) -> Dict[str, List[Dict[str, Any]]]:
        """Carga datos de reseñas."""
        try:
            reviews_path = "data/marketplace/reviews.json"
            if os.path.exists(reviews_path):
                with open(reviews_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return defaultdict(list)
        except Exception as e:
            logger.error(f"Error al cargar reseñas: {e}")
            return defaultdict(list)
    
    def _save_creators(self) -> None:
        """Guarda datos de creadores."""
        try:
            creators_path = "data/marketplace/creators.json"
            with open(creators_path, "w", encoding="utf-8") as f:
                json.dump(self.creators, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar creadores: {e}")
    
    def _save_brands(self) -> None:
        """Guarda datos de marcas."""
        try:
            brands_path = "data/marketplace/brands.json"
            with open(brands_path, "w", encoding="utf-8") as f:
                json.dump(self.brands, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar marcas: {e}")
    
    def _save_deals(self) -> None:
        """Guarda datos de acuerdos."""
        try:
            deals_path = "data/marketplace/deals.json"
            with open(deals_path, "w", encoding="utf-8") as f:
                json.dump(self.deals, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar acuerdos: {e}")
    
    def _save_proposals(self) -> None:
        """Guarda datos de propuestas."""
        try:
            proposals_path = "data/marketplace/proposals.json"
            with open(proposals_path, "w", encoding="utf-8") as f:
                json.dump(self.proposals, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar propuestas: {e}")
    
    def _save_reviews(self) -> None:
        """Guarda datos de reseñas."""
        try:
            reviews_path = "data/marketplace/reviews.json"
            with open(reviews_path, "w", encoding="utf-8") as f:
                json.dump(self.reviews, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar reseñas: {e}")
    
    def register_creator(self, name: str, channel_id: str, niche: str, 
                        platforms: List[str], audience_size: int,
                        rates: Dict[str, float], portfolio: Optional[List[str]] = None,
                        bio: Optional[str] = None) -> Dict[str, Any]:
        """
        Registra un creador de contenido en el marketplace.
        
        Args:
            name: Nombre del creador
            channel_id: ID del canal principal
            niche: Nicho o categoría
            platforms: Plataformas donde está presente
            audience_size: Tamaño de audiencia
            rates: Tarifas por tipo de colaboración
            portfolio: Enlaces a trabajos anteriores (opcional)
            bio: Biografía o descripción (opcional)
            
        Returns:
            Datos del creador registrado
        """
        # Generar ID único
        creator_id = str(uuid.uuid4())
        
        # Crear perfil de creador
        creator = {
            "creator_id": creator_id,
            "name": name,
            "channel_id": channel_id,
            "niche": niche,
            "platforms": platforms,
            "audience_size": audience_size,
            "rates": rates,
            "portfolio": portfolio or [],
            "bio": bio or "",
            "rating": 0.0,
            "reviews_count": 0,
            "deals_count": 0,
            "verified": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Guardar creador
        self.creators[creator_id] = creator
        self._save_creators()
        
        logger.info(f"Creador registrado: {name} ({creator_id})")
        
        return creator
    
    def register_brand(self, name: str, industry: str, website: str,
                     budget_range: List[float], target_audience: List[str],
                     contact_email: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Registra una marca en el marketplace.
        
        Args:
            name: Nombre de la marca
            industry: Industria o sector
            website: Sitio web
            budget_range: Rango de presupuesto [min, max]
            target_audience: Audiencia objetivo
            contact_email: Email de contacto
            description: Descripción de la marca (opcional)
            
        Returns:
            Datos de la marca registrada
        """
        # Generar ID único
        brand_id = str(uuid.uuid4())
        
        # Crear perfil de marca
        brand = {
            "brand_id": brand_id,
            "name": name,
            "industry": industry,
            "website": website,
            "budget_range": budget_range,
            "target_audience": target_audience,
            "contact_email": contact_email,
            "description": description or "",
            "rating": 0.0,
            "reviews_count": 0,
            "deals_count": 0,
            "verified": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Guardar marca
        self.brands[brand_id] = brand
        self._save_brands()
        
        logger.info(f"Marca registrada: {name} ({brand_id})")
        
        return brand
    
    def create_proposal(self, creator_id: str, brand_id: str, 
                      collaboration_type: str, description: str,
                      deliverables: List[str], timeline: int,
                      budget: float, terms: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Crea una propuesta de colaboración.
        
        Args:
            creator_id: ID del creador
            brand_id: ID de la marca
            collaboration_type: Tipo de colaboración
            description: Descripción de la propuesta
            deliverables: Entregables
            timeline: Plazo en días
            budget: Presupuesto
            terms: Términos adicionales (opcional)
            
        Returns:
            Datos de la propuesta creada
        """
        # Verificar que el creador existe
        if creator_id not in self.creators:
            logger.warning(f"Creador no encontrado: {creator_id}")
            return {
                "status": "error",
                "message": "Creador no encontrado"
            }
        
        # Verificar que la marca existe
        if brand_id not in self.brands:
            logger.warning(f"Marca no encontrada: {brand_id}")
            return {
                "status": "error",
                "message": "Marca no encontrada"
            }
        
        # Generar ID único
        proposal_id = str(uuid.uuid4())
        
        # Crear propuesta
        proposal = {
            "proposal_id": proposal_id,
            "creator_id": creator_id,
            "brand_id": brand_id,
            "collaboration_type": collaboration_type,
            "description": description,
            "deliverables": deliverables,
            "timeline": timeline,
            "budget": budget,
            "terms": terms or {},
            "status": "pending",  # pending, accepted, rejected, expired
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "expiration_date": (datetime.now() + timedelta(days=14)).isoformat()
        }
        
        # Guardar propuesta
        self.proposals[proposal_id] = proposal
        self._save_proposals()
        
        logger.info(f"Propuesta creada: {proposal_id} entre creador {creator_id} y marca {brand_id}")
        
        return proposal
    
    def update_proposal_status(self, proposal_id: str, status: str, 
                             notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Actualiza el estado de una propuesta.
        
        Args:
            proposal_id: ID de la propuesta
            status: Nuevo estado (pending, accepted, rejected, expired)
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos de la propuesta actualizada
        """
        # Verificar que la propuesta existe
        if proposal_id not in self.proposals:
            logger.warning(f"Propuesta no encontrada: {proposal_id}")
            return {
                "status": "error",
                "message": "Propuesta no encontrada"
            }
        
        # Validar estado
        valid_statuses = ["pending", "accepted", "rejected", "expired"]
        if status not in valid_statuses:
            logger.warning(f"Estado no válido: {status}")
            return {
                "status": "error",
                "message": f"Estado no válido. Debe ser uno de: {', '.join(valid_statuses)}"
            }
        
        # Actualizar estado
        self.proposals[proposal_id]["status"] = status
        self.proposals[proposal_id]["updated_at"] = datetime.now().isoformat()
        
        if notes:
            self.proposals[proposal_id]["notes"] = notes
        
        # Si se acepta la propuesta, crear un acuerdo
        if status == "accepted":
            self._create_deal_from_proposal(proposal_id)
        
        # Guardar propuesta
        self._save_proposals()
        
        logger.info(f"Estado de propuesta {proposal_id} actualizado a: {status}")
        
        return self.proposals[proposal_id]
    
    def _create_deal_from_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """
        Crea un acuerdo a partir de una propuesta aceptada.
        
        Args:
            proposal_id: ID de la propuesta
            
        Returns:
            Datos del acuerdo creado
        """
        proposal = self.proposals[proposal_id]
        
        # Generar ID único
        deal_id = str(uuid.uuid4())
        
        # Crear acuerdo
        deal = {
            "deal_id": deal_id,
            "proposal_id": proposal_id,
            "creator_id": proposal["creator_id"],
            "brand_id": proposal["brand_id"],
            "collaboration_type": proposal["collaboration_type"],
            "description": proposal["description"],
            "deliverables": proposal["deliverables"],
            "timeline": proposal["timeline"],
            "budget": proposal["budget"],
            "terms": proposal["terms"],
            "status": "active",  # active, completed, cancelled
            "progress": 0,
            "milestones": self._generate_milestones(proposal),
            "payments": {
                "total": proposal["budget"],
                "paid": 0.0,
                "pending": proposal["budget"],
                "next_payment": proposal["budget"] * 0.5,  # 50% adelanto
                "payment_schedule": [
                    {
                        "amount": proposal["budget"] * 0.5,
                        "due_date": datetime.now().isoformat(),
                        "status": "pending",
                        "description": "Pago inicial (50%)"
                    },
                    {
                        "amount": proposal["budget"] * 0.5,
                        "due_date": (datetime.now() + timedelta(days=proposal["timeline"])).isoformat(),
                        "status": "pending",
                        "description": "Pago final (50%)"
                    }
                ]
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=proposal["timeline"])).isoformat()
        }
        
        # Guardar acuerdo
        self.deals[deal_id] = deal
        self._save_deals()
        
        # Actualizar contadores de acuerdos
        self.creators[proposal["creator_id"]]["deals_count"] += 1
        self.brands[proposal["brand_id"]]["deals_count"] += 1
        
        # Guardar datos actualizados
        self._save_creators()
        self._save_brands()
        
        logger.info(f"Acuerdo creado: {deal_id} a partir de propuesta {proposal_id}")
        
        return deal
    
    def _generate_milestones(self, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera hitos para un acuerdo basado en la propuesta.
        
        Args:
            proposal: Datos de la propuesta
            
        Returns:
            Lista de hitos
        """
        timeline = proposal["timeline"]
        milestones = []
        
        # Hito 1: Inicio
        milestones.append({
            "milestone_id": str(uuid.uuid4()),
            "title": "Inicio del proyecto",
            "description": "Reunión inicial y planificación",
            "due_date": datetime.now().isoformat(),
            "status": "completed",
            "deliverables": ["Plan de trabajo"],
            "completion_percentage": 10
        })
        
        # Hito 2: Desarrollo
        milestones.append({
            "milestone_id": str(uuid.uuid4()),
            "title": "Desarrollo de contenido",
            "description": "Creación del contenido acordado",
            "due_date": (datetime.now() + timedelta(days=int(timeline * 0.6))).isoformat(),
            "status": "pending",
            "deliverables": ["Borrador de contenido"],
            "completion_percentage": 50
        })
        
        # Hito 3: Revisión
        milestones.append({
            "milestone_id": str(uuid.uuid4()),
            "title": "Revisión y ajustes",
            "description": "Revisión del contenido y ajustes finales",
            "due_date": (datetime.now() + timedelta(days=int(timeline * 0.8))).isoformat(),
            "status": "pending",
            "deliverables": ["Contenido revisado"],
            "completion_percentage": 80
        })
        
        # Hito 4: Entrega final
        milestones.append({
            "milestone_id": str(uuid.uuid4()),
            "title": "Entrega final",
            "description": "Publicación y entrega de informes",
            "due_date": (datetime.now() + timedelta(days=timeline)).isoformat(),
            "status": "pending",
            "deliverables": ["Contenido publicado", "Informe de resultados"],
            "completion_percentage": 100
        })
        
        return milestones
    
    def update_deal_progress(self, deal_id: str, progress: int, 
                           milestone_id: Optional[str] = None,
                           milestone_status: Optional[str] = None,
                           notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Actualiza el progreso de un acuerdo.
        
        Args:
            deal_id: ID del acuerdo
            progress: Porcentaje de progreso (0-100)
            milestone_id: ID del hito a actualizar (opcional)
            milestone_status: Estado del hito (opcional)
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos del acuerdo actualizado
        """
        # Verificar que el acuerdo existe
        if deal_id not in self.deals:
            logger.warning(f"Acuerdo no encontrado: {deal_id}")
            return {
                "status": "error",
                "message": "Acuerdo no encontrado"
            }
        
        # Validar progreso
        if progress < 0 or progress > 100:
            logger.warning(f"Progreso no válido: {progress}")
            return {
                "status": "error",
                "message": "El progreso debe estar entre 0 y 100"
            }
        
        # Actualizar progreso
        self.deals[deal_id]["progress"] = progress
        self.deals[deal_id]["updated_at"] = datetime.now().isoformat()
        
        if notes:
            if "notes" not in self.deals[deal_id]:
                self.deals[deal_id]["notes"] = []
            
            self.deals[deal_id]["notes"].append({
                "timestamp": datetime.now().isoformat(),
                "content": notes
            })
        
        # Actualizar hito si se proporciona
        if milestone_id and milestone_status:
            for milestone in self.deals[deal_id]["milestones"]:
                if milestone["milestone_id"] == milestone_id:
                    milestone["status"] = milestone_status
                    milestone["updated_at"] = datetime.now().isoformat()
                    break
        
        # Si el progreso es 100%, marcar como completado
        if progress == 100:
            self.deals[deal_id]["status"] = "completed"
            self.deals[deal_id]["end_date"] = datetime.now().isoformat()
        
        # Guardar acuerdo
        self._save_deals()
        
        logger.info(f"Progreso de acuerdo {deal_id} actualizado a: {progress}%")
        
        return self.deals[deal_id]
    
    def record_payment(self, deal_id: str, amount: float, 
                     payment_method: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Registra un pago para un acuerdo.
        
        Args:
            deal_id: ID del acuerdo
            amount: Monto del pago
            payment_method: Método de pago
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos del pago registrado
        """
        # Verificar que el acuerdo existe
        if deal_id not in self.deals:
            logger.warning(f"Acuerdo no encontrado: {deal_id}")
            return {
                "status": "error",
                "message": "Acuerdo no encontrado"
            }
        
        # Verificar monto
        if amount <= 0:
            logger.warning(f"Monto no válido: {amount}")
            return {
                "status": "error",
                "message": "El monto debe ser mayor que cero"
            }
        
        # Verificar que no exceda el monto pendiente
        pending = self.deals[deal_id]["payments"]["pending"]
        if amount > pending:
            logger.warning(f"Monto excede pendiente: {amount} > {pending}")
            return {
                "status": "error",
                "message": f"El monto excede el pago pendiente ({pending})"
            }
        
        # Registrar pago
        payment = {
            "payment_id": str(uuid.uuid4()),
            "deal_id": deal_id,
            "amount": amount,
            "payment_method": payment_method,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Actualizar pagos en el acuerdo
        self.deals[deal_id]["payments"]["paid"] += amount
        self.deals[deal_id]["payments"]["pending"] -= amount
        
        # Actualizar estado de pagos programados
        for scheduled_payment in self.deals[deal_id]["payments"]["payment_schedule"]:
            if scheduled_payment["status"] == "pending":
                scheduled_payment["status"] = "paid"
                scheduled_payment["paid_date"] = datetime.now().isoformat()
                scheduled_payment["payment_id"] = payment["payment_id"]
                break
        
        # Actualizar próximo pago
        if self.deals[deal_id]["payments"]["pending"] > 0:
            # Buscar el siguiente pago pendiente
            for scheduled_payment in self.deals[deal_id]["payments"]["payment_schedule"]:
                if scheduled_payment["status"] == "pending":
                    self.deals[deal_id]["payments"]["next_payment"] = scheduled_payment["amount"]
                    break
        else:
            self.deals[deal_id]["payments"]["next_payment"] = 0
        
        # Guardar historial de pagos
        if "payment_history" not in self.deals[deal_id]:
            self.deals[deal_id]["payment_history"] = []
        
        self.deals[deal_id]["payment_history"].append(payment)
        
        # Guardar acuerdo
        self._save_deals()
        
        logger.info(f"Pago registrado para acuerdo {deal_id}: {amount}")
        
        return payment
    
    def add_review(self, reviewer_type: str, reviewer_id: str, 
                 target_type: str, target_id: str, 
                 rating: float, comment: str) -> Dict[str, Any]:
        """
        Añade una reseña para un creador o marca.
        
        Args:
            reviewer_type: Tipo de revisor (creator, brand)
            reviewer_id: ID del revisor
            target_type: Tipo de objetivo (creator, brand)
            target_id: ID del objetivo
            rating: Calificación (1-5)
            comment: Comentario
            
        Returns:
            Datos de la reseña añadida
        """
        # Validar tipos
        valid_types = ["creator", "brand"]
        if reviewer_type not in valid_types or target_type not in valid_types:
            logger.warning(f"Tipo no válido: {reviewer_type} o {target_type}")
            return {
                "status": "error",
                "message": f"Tipo no válido. Debe ser uno de: {', '.join(valid_types)}"
            }
        
        # Validar IDs
        if (reviewer_type == "creator" and reviewer_id not in self.creators) or \
           (reviewer_type == "brand" and reviewer_id not in self.brands):
            logger.warning(f"Revisor no encontrado: {reviewer_id}")
            return {
                "status": "error",
                "message": "Revisor no encontrado"
            }
        
        if (target_type == "creator" and target_id not in self.creators) or \
           (target_type == "brand" and target_id not in self.brands):
            logger.warning(f"Objetivo no encontrado: {target_id}")
            return {
                "status": "error",
                "message": "Objetivo no encontrado"
            }
        
        # Validar calificación
        if rating < 1 or rating > 5:
            logger.warning(f"Calificación no válida: {rating}")
            return {
                "status": "error",
                "message": "La calificación debe estar entre 1 y 5"
            }
        
        # Crear reseña
        review = {
            "review_id": str(uuid.uuid4()),
            "reviewer_type": reviewer_type,
            "reviewer_id": reviewer_id,
            "target_type": target_type,
            "target_id": target_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now().isoformat(),
            "status": "active"  # active, hidden, reported
        }
        
        # Guardar reseña
        if target_id not in self.reviews:
            self.reviews[target_id] = []
        
        self.reviews[target_id].append(review)
        
        # Actualizar calificación promedio
        self._update_average_rating(target_type, target_id)
        
        # Guardar reseñas
        self._save_reviews()
        
        logger.info(f"Reseña añadida para {target_type} {target_id} por {reviewer_type} {reviewer_id}")
        
        return review
    
    def _update_average_rating(self, entity_type: str, entity_id: str) -> None:
        """
        Actualiza la calificación promedio de un creador o marca.
        
        Args:
            entity_type: Tipo de entidad (creator, brand)
            entity_id: ID de la entidad
        """
        if entity_id not in self.reviews:
            return
        
        # Filtrar reseñas activas
        active_reviews = [r for r in self.reviews[entity_id] if r["status"] == "active"]
        
        if not active_reviews:
            return
        
        # Calcular promedio
        total_rating = sum(r["rating"] for r in active_reviews)
        average_rating = total_rating / len(active_reviews)
        
        # Actualizar entidad
        if entity_type == "creator" and entity_id in self.creators:
            self.creators[entity_id]["rating"] = round(average_rating, 1)
            self.creators[entity_id]["reviews_count"] = len(active_reviews)
            self._save_creators()
        
        elif entity_type == "brand" and entity_id in self.brands:
            self.brands[entity_id]["rating"] = round(average_rating, 1)
            self.brands[entity_id]["reviews_count"] = len(active_reviews)
            self._save_brands()
    
    def search_creators(self, filters: Dict[str, Any] = None, 
                       sort_by: str = "rating", 
                       limit: int = 20) -> List[Dict[str, Any]]:
        """
        Busca creadores según filtros.
        
        Args:
            filters: Filtros de búsqueda (opcional)
            sort_by: Campo para ordenar (rating, audience_size, deals_count)
            limit: Límite de resultados
            
        Returns:
            Lista de creadores que coinciden con los filtros
        """
        filters = filters or {}
        
        # Filtrar creadores
        filtered_creators = []
        
        for creator_id, creator in self.creators.items():
            # Filtrar por estado
            if "status" in filters and creator["status"] != filters["status"]:
                continue
            
            # Filtrar por nicho
            if "niche" in filters and creator["niche"] != filters["niche"]:
                continue
            
            # Filtrar por plataforma
            if "platform" in filters and filters["platform"] not in creator["platforms"]:
                continue
            
            # Filtrar por tamaño de audiencia mínimo
            if "min_audience" in filters and creator["audience_size"] < filters["min_audience"]:
                continue
            
            # Filtrar por calificación mínima
            if "min_rating" in filters and creator["rating"] < filters["min_rating"]:
                continue
            
            # Filtrar por presupuesto máximo
            if "max_budget" in filters:
                # Verificar si alguna tarifa está dentro del presupuesto
                rates_within_budget = False
                for rate in creator["rates"].values():
                    if rate <= filters["max_budget"]:
                        rates_within_budget = True
                        break
                
                if not rates_within_budget:
                    continue
            
            # Añadir creador filtrado
            filtered_creators.append(creator)
        
        # Ordenar resultados
        if sort_by == "rating":
            filtered_creators.sort(key=lambda x: x["rating"], reverse=True)
        elif sort_by == "audience_size":
            filtered_creators.sort(key=lambda x: x["audience_size"], reverse=True)
        elif sort_by == "deals_count":
            filtered_creators.sort(key=lambda x: x["deals_count"], reverse=True)
        
        # Limitar resultados
        return filtered_creators[:limit]
    
    def search_brands(self, filters: Dict[str, Any] = None, 
                    sort_by: str = "rating", 
                    limit: int = 20) -> List[Dict[str, Any]]:
        """
        Busca marcas según filtros.
        
        Args:
            filters: Filtros de búsqueda (opcional)
            sort_by: Campo para ordenar (rating, deals_count)
            limit: Límite de resultados
            
        Returns:
            Lista de marcas que coinciden con los filtros
        """
        filters = filters or {}
        
        # Filtrar marcas
        filtered_brands = []
        
        for brand_id, brand in self.brands.items():
            # Filtrar por estado
            if "status" in filters and brand["status"] != filters["status"]:
                continue
            
            # Filtrar por industria
            if "industry" in filters and brand["industry"] != filters["industry"]:
                continue
            
            # Filtrar por presupuesto mínimo
            if "min_budget" in filters and brand["budget_range"][0] < filters["min_budget"]:
                continue
            
            # Filtrar por calificación mínima
            if "min_rating" in filters and brand["rating"] < filters["min_rating"]:
                continue
            
            # Filtrar por audiencia objetivo
            if "target_audience" in filters:
                if filters["target_audience"] not in brand["target_audience"]:
                    continue
            
            # Añadir marca filtrada
            filtered_brands.append(brand)
        
        # Ordenar resultados
        if sort_by == "rating":
            filtered_brands.sort(key=lambda x: x["rating"], reverse=True)
        elif sort_by == "deals_count":
            filtered_brands.sort(key=lambda x: x["deals_count"], reverse=True)
        
        # Limitar resultados
        return filtered_brands[:limit]
    
    def get_marketplace_analytics(self) -> Dict[str, Any]:
        """
        Obtiene análisis del marketplace.
        
        Returns:
            Estadísticas y análisis del marketplace
        """
        # Estadísticas generales
        stats = {
            "creators_count": len(self.creators),
            "brands_count": len(self.brands),
            "deals_count": len(self.deals),
            "proposals_count": len(self.proposals),
            "total_deal_value": sum(deal["budget"] for deal in self.deals.values()),
            "average_deal_value": 0,
            "conversion_rate": 0,  # Propuestas aceptadas / total
            "average_rating": {
                "creators": 0,
                "brands": 0
            }
        }
        
        # Calcular promedio de valor de acuerdos
        if stats["deals_count"] > 0:
            stats["average_deal_value"] = stats["total_deal_value"] / stats["deals_count"]
        
                # Calcular tasa de conversión
        accepted_proposals = sum(1 for p in self.proposals.values() if p["status"] == "accepted")
        if stats["proposals_count"] > 0:
            stats["conversion_rate"] = accepted_proposals / stats["proposals_count"]
        
        # Calcular calificaciones promedio
        if self.creators:
            total_creator_rating = sum(c["rating"] for c in self.creators.values())
            stats["average_rating"]["creators"] = round(total_creator_rating / len(self.creators), 1)
        
        if self.brands:
            total_brand_rating = sum(b["rating"] for b in self.brands.values())
            stats["average_rating"]["brands"] = round(total_brand_rating / len(self.brands), 1)
        
        # Análisis por categoría/nicho
        stats["by_niche"] = {}
        for creator in self.creators.values():
            niche = creator["niche"]
            if niche not in stats["by_niche"]:
                stats["by_niche"][niche] = {
                    "creators_count": 0,
                    "deals_count": 0,
                    "average_audience": 0,
                    "total_audience": 0
                }
            
            stats["by_niche"][niche]["creators_count"] += 1
            stats["by_niche"][niche]["deals_count"] += creator["deals_count"]
            stats["by_niche"][niche]["total_audience"] += creator["audience_size"]
        
        # Calcular audiencia promedio por nicho
        for niche, data in stats["by_niche"].items():
            if data["creators_count"] > 0:
                data["average_audience"] = data["total_audience"] // data["creators_count"]
        
        # Análisis por industria
        stats["by_industry"] = {}
        for brand in self.brands.values():
            industry = brand["industry"]
            if industry not in stats["by_industry"]:
                stats["by_industry"][industry] = {
                    "brands_count": 0,
                    "deals_count": 0,
                    "average_budget": 0,
                    "total_budget": 0
                }
            
            stats["by_industry"][industry]["brands_count"] += 1
            stats["by_industry"][industry]["deals_count"] += brand["deals_count"]
            stats["by_industry"][industry]["total_budget"] += sum(brand["budget_range"]) / 2  # Promedio del rango
        
        # Calcular presupuesto promedio por industria
        for industry, data in stats["by_industry"].items():
            if data["brands_count"] > 0:
                data["average_budget"] = data["total_budget"] / data["brands_count"]
        
        # Análisis temporal
        stats["time_analysis"] = {
            "deals_by_month": {},
            "proposals_by_month": {}
        }
        
        # Analizar acuerdos por mes
        for deal in self.deals.values():
            date = datetime.fromisoformat(deal["created_at"])
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in stats["time_analysis"]["deals_by_month"]:
                stats["time_analysis"]["deals_by_month"][month_key] = {
                    "count": 0,
                    "value": 0
                }
            
            stats["time_analysis"]["deals_by_month"][month_key]["count"] += 1
            stats["time_analysis"]["deals_by_month"][month_key]["value"] += deal["budget"]
        
        # Analizar propuestas por mes
        for proposal in self.proposals.values():
            date = datetime.fromisoformat(proposal["created_at"])
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in stats["time_analysis"]["proposals_by_month"]:
                stats["time_analysis"]["proposals_by_month"][month_key] = {
                    "count": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "pending": 0,
                    "expired": 0
                }
            
            stats["time_analysis"]["proposals_by_month"][month_key]["count"] += 1
            stats["time_analysis"]["proposals_by_month"][month_key][proposal["status"]] += 1
        
        return stats
    
    def get_creator_analytics(self, creator_id: str) -> Dict[str, Any]:
        """
        Obtiene análisis para un creador específico.
        
        Args:
            creator_id: ID del creador
            
        Returns:
            Estadísticas y análisis del creador
        """
        # Verificar que el creador existe
        if creator_id not in self.creators:
            logger.warning(f"Creador no encontrado: {creator_id}")
            return {
                "status": "error",
                "message": "Creador no encontrado"
            }
        
        creator = self.creators[creator_id]
        
        # Estadísticas generales
        stats = {
            "creator_id": creator_id,
            "name": creator["name"],
            "niche": creator["niche"],
            "platforms": creator["platforms"],
            "audience_size": creator["audience_size"],
            "rating": creator["rating"],
            "reviews_count": creator["reviews_count"],
            "deals_count": creator["deals_count"],
            "total_earnings": 0,
            "average_deal_value": 0,
            "proposal_stats": {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "pending": 0,
                "expired": 0,
                "conversion_rate": 0
            },
            "deals_by_status": {
                "active": 0,
                "completed": 0,
                "cancelled": 0
            },
            "reviews": []
        }
        
        # Obtener propuestas del creador
        creator_proposals = [p for p in self.proposals.values() if p["creator_id"] == creator_id]
        
        # Estadísticas de propuestas
        stats["proposal_stats"]["total"] = len(creator_proposals)
        
        for proposal in creator_proposals:
            stats["proposal_stats"][proposal["status"]] += 1
        
        if stats["proposal_stats"]["total"] > 0:
            stats["proposal_stats"]["conversion_rate"] = stats["proposal_stats"]["accepted"] / stats["proposal_stats"]["total"]
        
        # Obtener acuerdos del creador
        creator_deals = [d for d in self.deals.values() if d["creator_id"] == creator_id]
        
        # Estadísticas de acuerdos
        for deal in creator_deals:
            stats["total_earnings"] += deal["budget"]
            stats["deals_by_status"][deal["status"]] += 1
        
        if stats["deals_count"] > 0:
            stats["average_deal_value"] = stats["total_earnings"] / stats["deals_count"]
        
        # Obtener reseñas del creador
        if creator_id in self.reviews:
            stats["reviews"] = [r for r in self.reviews[creator_id] if r["status"] == "active"]
        
        # Análisis temporal
        stats["time_analysis"] = {
            "deals_by_month": {},
            "earnings_by_month": {}
        }
        
        # Analizar acuerdos por mes
        for deal in creator_deals:
            date = datetime.fromisoformat(deal["created_at"])
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in stats["time_analysis"]["deals_by_month"]:
                stats["time_analysis"]["deals_by_month"][month_key] = 0
                stats["time_analysis"]["earnings_by_month"][month_key] = 0
            
            stats["time_analysis"]["deals_by_month"][month_key] += 1
            stats["time_analysis"]["earnings_by_month"][month_key] += deal["budget"]
        
        return stats
    
    def get_brand_analytics(self, brand_id: str) -> Dict[str, Any]:
        """
        Obtiene análisis para una marca específica.
        
        Args:
            brand_id: ID de la marca
            
        Returns:
            Estadísticas y análisis de la marca
        """
                # Verificar que la marca existe
        if brand_id not in self.brands:
            logger.warning(f"Marca no encontrada: {brand_id}")
            return {
                "status": "error",
                "message": "Marca no encontrada"
            }
        
        brand = self.brands[brand_id]
        
        # Estadísticas generales
        stats = {
            "brand_id": brand_id,
            "name": brand["name"],
            "industry": brand["industry"],
            "website": brand["website"],
            "budget_range": brand["budget_range"],
            "target_audience": brand["target_audience"],
            "rating": brand["rating"],
            "reviews_count": brand["reviews_count"],
            "deals_count": brand["deals_count"],
            "total_spent": 0,
            "average_deal_value": 0,
            "proposal_stats": {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "pending": 0,
                "expired": 0,
                "conversion_rate": 0
            },
            "deals_by_status": {
                "active": 0,
                "completed": 0,
                "cancelled": 0
            },
            "reviews": []
        }
        
        # Obtener propuestas de la marca
        brand_proposals = [p for p in self.proposals.values() if p["brand_id"] == brand_id]
        
        # Estadísticas de propuestas
        stats["proposal_stats"]["total"] = len(brand_proposals)
        
        for proposal in brand_proposals:
            stats["proposal_stats"][proposal["status"]] += 1
        
        if stats["proposal_stats"]["total"] > 0:
            stats["proposal_stats"]["conversion_rate"] = stats["proposal_stats"]["accepted"] / stats["proposal_stats"]["total"]
        
        # Obtener acuerdos de la marca
        brand_deals = [d for d in self.deals.values() if d["brand_id"] == brand_id]
        
        # Estadísticas de acuerdos
        for deal in brand_deals:
            stats["total_spent"] += deal["budget"]
            stats["deals_by_status"][deal["status"]] += 1
        
        if stats["deals_count"] > 0:
            stats["average_deal_value"] = stats["total_spent"] / stats["deals_count"]
        
        # Obtener reseñas de la marca
        if brand_id in self.reviews:
            stats["reviews"] = [r for r in self.reviews[brand_id] if r["status"] == "active"]
        
        # Análisis temporal
        stats["time_analysis"] = {
            "deals_by_month": {},
            "spending_by_month": {}
        }
        
        # Analizar acuerdos por mes
        for deal in brand_deals:
            date = datetime.fromisoformat(deal["created_at"])
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in stats["time_analysis"]["deals_by_month"]:
                stats["time_analysis"]["deals_by_month"][month_key] = 0
                stats["time_analysis"]["spending_by_month"][month_key] = 0
            
            stats["time_analysis"]["deals_by_month"][month_key] += 1
            stats["time_analysis"]["spending_by_month"][month_key] += deal["budget"]
        
        # Análisis de creadores colaboradores
        stats["collaborators"] = {}
        
        for deal in brand_deals:
            creator_id = deal["creator_id"]
            if creator_id in self.creators:
                creator = self.creators[creator_id]
                
                if creator_id not in stats["collaborators"]:
                    stats["collaborators"][creator_id] = {
                        "name": creator["name"],
                        "niche": creator["niche"],
                        "audience_size": creator["audience_size"],
                        "deals_count": 0,
                        "total_spent": 0
                    }
                
                stats["collaborators"][creator_id]["deals_count"] += 1
                stats["collaborators"][creator_id]["total_spent"] += deal["budget"]
        
        return stats
    
    def get_deal_details(self, deal_id: str) -> Dict[str, Any]:
        """
        Obtiene detalles completos de un acuerdo.
        
        Args:
            deal_id: ID del acuerdo
            
        Returns:
            Detalles completos del acuerdo
        """
        # Verificar que el acuerdo existe
        if deal_id not in self.deals:
            logger.warning(f"Acuerdo no encontrado: {deal_id}")
            return {
                "status": "error",
                "message": "Acuerdo no encontrado"
            }
        
        deal = self.deals[deal_id]
        
        # Obtener información del creador
        creator_info = {}
        if deal["creator_id"] in self.creators:
            creator = self.creators[deal["creator_id"]]
            creator_info = {
                "creator_id": creator["creator_id"],
                "name": creator["name"],
                "niche": creator["niche"],
                "platforms": creator["platforms"],
                "audience_size": creator["audience_size"],
                "rating": creator["rating"]
            }
        
        # Obtener información de la marca
        brand_info = {}
        if deal["brand_id"] in self.brands:
            brand = self.brands[deal["brand_id"]]
            brand_info = {
                "brand_id": brand["brand_id"],
                "name": brand["name"],
                "industry": brand["industry"],
                "website": brand["website"],
                "rating": brand["rating"]
            }
        
        # Construir respuesta detallada
        deal_details = {
            **deal,
            "creator": creator_info,
            "brand": brand_info,
            "proposal": self.proposals.get(deal["proposal_id"], {})
        }
        
        return deal_details
    
    def cancel_deal(self, deal_id: str, reason: str) -> Dict[str, Any]:
        """
        Cancela un acuerdo activo.
        
        Args:
            deal_id: ID del acuerdo
            reason: Motivo de la cancelación
            
        Returns:
            Datos del acuerdo cancelado
        """
        # Verificar que el acuerdo existe
        if deal_id not in self.deals:
            logger.warning(f"Acuerdo no encontrado: {deal_id}")
            return {
                "status": "error",
                "message": "Acuerdo no encontrado"
            }
        
        # Verificar que el acuerdo está activo
        if self.deals[deal_id]["status"] != "active":
            logger.warning(f"Acuerdo no está activo: {deal_id}")
            return {
                "status": "error",
                "message": "Solo se pueden cancelar acuerdos activos"
            }
        
        # Actualizar estado
        self.deals[deal_id]["status"] = "cancelled"
        self.deals[deal_id]["updated_at"] = datetime.now().isoformat()
        
        # Registrar motivo de cancelación
        self.deals[deal_id]["cancellation"] = {
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar acuerdo
        self._save_deals()
        
        logger.info(f"Acuerdo {deal_id} cancelado: {reason}")
        
        return self.deals[deal_id]
    
    def update_creator_profile(self, creator_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza el perfil de un creador.
        
        Args:
            creator_id: ID del creador
            updates: Campos a actualizar
            
        Returns:
            Perfil actualizado del creador
        """
        # Verificar que el creador existe
        if creator_id not in self.creators:
            logger.warning(f"Creador no encontrado: {creator_id}")
            return {
                "status": "error",
                "message": "Creador no encontrado"
            }
        
        # Campos que no se pueden actualizar directamente
        protected_fields = ["creator_id", "rating", "reviews_count", "deals_count", 
                           "created_at", "verified"]
        
        # Actualizar campos permitidos
        for key, value in updates.items():
            if key not in protected_fields:
                self.creators[creator_id][key] = value
        
        # Actualizar timestamp
        self.creators[creator_id]["updated_at"] = datetime.now().isoformat()
        
        # Guardar creadores
        self._save_creators()
        
        logger.info(f"Perfil de creador {creator_id} actualizado")
        
        return self.creators[creator_id]
    
    def update_brand_profile(self, brand_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza el perfil de una marca.
        
        Args:
            brand_id: ID de la marca
            updates: Campos a actualizar
            
        Returns:
            Perfil actualizado de la marca
        """
        # Verificar que la marca existe
        if brand_id not in self.brands:
            logger.warning(f"Marca no encontrada: {brand_id}")
            return {
                "status": "error",
                "message": "Marca no encontrada"
            }
        
        # Campos que no se pueden actualizar directamente
        protected_fields = ["brand_id", "rating", "reviews_count", "deals_count", 
                           "created_at", "verified"]
        
        # Actualizar campos permitidos
        for key, value in updates.items():
            if key not in protected_fields:
                self.brands[brand_id][key] = value
        
        # Actualizar timestamp
        self.brands[brand_id]["updated_at"] = datetime.now().isoformat()
        
        # Guardar marcas
        self._save_brands()
        
        logger.info(f"Perfil de marca {brand_id} actualizado")
        
        return self.brands[brand_id]
    
    def verify_entity(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """
        Verifica un creador o marca.
        
        Args:
            entity_type: Tipo de entidad (creator, brand)
            entity_id: ID de la entidad
            
        Returns:
            Datos de la entidad verificada
        """
        # Validar tipo
        if entity_type not in ["creator", "brand"]:
            logger.warning(f"Tipo no válido: {entity_type}")
            return {
                "status": "error",
                "message": "Tipo no válido. Debe ser 'creator' o 'brand'"
            }
        
        # Verificar que la entidad existe
        if (entity_type == "creator" and entity_id not in self.creators) or \
           (entity_type == "brand" and entity_id not in self.brands):
            logger.warning(f"Entidad no encontrada: {entity_id}")
            return {
                "status": "error",
                "message": "Entidad no encontrada"
            }
        
        # Actualizar estado de verificación
        if entity_type == "creator":
            self.creators[entity_id]["verified"] = True
            self.creators[entity_id]["updated_at"] = datetime.now().isoformat()
            self._save_creators()
            logger.info(f"Creador {entity_id} verificado")
            return self.creators[entity_id]
        else:
            self.brands[entity_id]["verified"] = True
            self.brands[entity_id]["updated_at"] = datetime.now().isoformat()
            self._save_brands()
            logger.info(f"Marca {entity_id} verificada")
            return self.brands[entity_id]
    
    def report_review(self, review_id: str, target_id: str, reason: str) -> Dict[str, Any]:
        """
        Reporta una reseña como inapropiada.
        
        Args:
            review_id: ID de la reseña
            target_id: ID del objetivo de la reseña
            reason: Motivo del reporte
            
        Returns:
            Estado del reporte
        """
        # Verificar que el objetivo existe en las reseñas
        if target_id not in self.reviews:
            logger.warning(f"Objetivo no encontrado en reseñas: {target_id}")
            return {
                "status": "error",
                "message": "Objetivo no encontrado en reseñas"
            }
        
        # Buscar la reseña
        review = None
        for r in self.reviews[target_id]:
            if r["review_id"] == review_id:
                review = r
                break
        
        if not review:
            logger.warning(f"Reseña no encontrada: {review_id}")
            return {
                "status": "error",
                "message": "Reseña no encontrada"
            }
        
        # Registrar reporte
        if "reports" not in review:
            review["reports"] = []
        
        report = {
            "report_id": str(uuid.uuid4()),
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"  # pending, reviewed, dismissed
        }
        
        review["reports"].append(report)
        
        # Guardar reseñas
        self._save_reviews()
        
        logger.info(f"Reseña {review_id} reportada: {reason}")
        
        return {
            "status": "success",
            "message": "Reseña reportada correctamente",
            "report_id": report["report_id"]
        }
    
    def moderate_review(self, review_id: str, target_id: str, 
                       action: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Modera una reseña reportada.
        
        Args:
            review_id: ID de la reseña
            target_id: ID del objetivo de la reseña
            action: Acción a tomar (keep, hide, delete)
            notes: Notas de moderación (opcional)
            
        Returns:
            Estado de la moderación
        """
        # Verificar que el objetivo existe en las reseñas
        if target_id not in self.reviews:
            logger.warning(f"Objetivo no encontrado en reseñas: {target_id}")
            return {
                "status": "error",
                "message": "Objetivo no encontrado en reseñas"
            }
        
        # Buscar la reseña
        review_index = None
        for i, r in enumerate(self.reviews[target_id]):
            if r["review_id"] == review_id:
                review_index = i
                break
        
        if review_index is None:
            logger.warning(f"Reseña no encontrada: {review_id}")
            return {
                "status": "error",
                "message": "Reseña no encontrada"
            }
        
        review = self.reviews[target_id][review_index]
        
        # Validar acción
        valid_actions = ["keep", "hide", "delete"]
        if action not in valid_actions:
            logger.warning(f"Acción no válida: {action}")
            return {
                "status": "error",
                "message": f"Acción no válida. Debe ser una de: {', '.join(valid_actions)}"
            }
        
        # Aplicar acción
        if action == "keep":
            review["status"] = "active"
            
            # Actualizar reportes
            if "reports" in review:
                for report in review["reports"]:
                    if report["status"] == "pending":
                        report["status"] = "dismissed"
        
        elif action == "hide":
            review["status"] = "hidden"
            
            # Actualizar reportes
            if "reports" in review:
                for report in review["reports"]:
                    if report["status"] == "pending":
                        report["status"] = "reviewed"
        
        elif action == "delete":
            # Eliminar reseña
            self.reviews[target_id].pop(review_index)
        
        # Registrar moderación
        if action != "delete" and notes:
            if "moderation" not in review:
                review["moderation"] = []
            
            review["moderation"].append({
                "action": action,
                "notes": notes,
                "timestamp": datetime.now().isoformat()
            })
        
        # Actualizar calificación promedio si es necesario
        if action != "keep":
            target_type = review["target_type"]
            self._update_average_rating(target_type, target_id)
        
        # Guardar reseñas
        self._save_reviews()
        
        logger.info(f"Reseña {review_id} moderada: {action}")
        
        return {
            "status": "success",
            "message": f"Reseña {action}",
            "action": action
        }
    
    def generate_contract(self, deal_id: str) -> Dict[str, Any]:
        """
        Genera un contrato para un acuerdo.
        
        Args:
            deal_id: ID del acuerdo
            
        Returns:
            Datos del contrato generado
        """
        # Verificar que el acuerdo existe
        if deal_id not in self.deals:
            logger.warning(f"Acuerdo no encontrado: {deal_id}")
            return {
                "status": "error",
                "message": "Acuerdo no encontrado"
            }
        
        deal = self.deals[deal_id]
        
        # Obtener información del creador
        creator_info = {}
        if deal["creator_id"] in self.creators:
            creator = self.creators[deal["creator_id"]]
            creator_info = {
                "name": creator["name"],
                "id": creator["creator_id"]
            }
        
        # Obtener información de la marca
        brand_info = {}
        if deal["brand_id"] in self.brands:
            brand = self.brands[deal["brand_id"]]
            brand_info = {
                "name": brand["name"],
                "id": brand["brand_id"]
            }
        
        # Generar contrato
        contract = {
            "contract_id": str(uuid.uuid4()),
            "deal_id": deal_id,
            "title": f"Contrato de Colaboración - {creator_info.get('name', 'Creador')} y {brand_info.get('name', 'Marca')}",
            "parties": {
                "creator": creator_info,
                "brand": brand_info
            },
            "terms": {
                "collaboration_type": deal["collaboration_type"],
                "description": deal["description"],
                "deliverables": deal["deliverables"],
                "timeline": deal["timeline"],
                "budget": deal["budget"],
                "payment_schedule": deal["payments"]["payment_schedule"],
                "additional_terms": deal["terms"]
            },
            "milestones": deal["milestones"],
            "legal": {
                "confidentiality": "Toda la información compartida durante esta colaboración se considera confidencial y no debe ser divulgada a terceros sin consentimiento previo por escrito.",
                "intellectual_property": "El creador mantiene los derechos de autor sobre el contenido creado, pero otorga a la marca una licencia no exclusiva para usar el contenido con fines promocionales.",
                "termination": "Cualquiera de las partes puede terminar este acuerdo con un aviso previo de 7 días. Los pagos se realizarán por el trabajo completado hasta la fecha de terminación.",
                "dispute_resolution": "Cualquier disputa se resolverá primero mediante negociación de buena fe, y si no se llega a un acuerdo, mediante arbitraje."
            },
            "signatures": {
                "creator": {
                    "signed": False,
                    "date": None
                },
                "brand": {
                    "signed": False,
                    "date": None
                }
            },
            "created_at": datetime.now().isoformat(),
            "status": "draft"  # draft, active, completed, terminated
        }
        
        # Guardar contrato en el acuerdo
        self.deals[deal_id]["contract"] = contract
        self._save_deals()
        
        logger.info(f"Contrato generado para acuerdo {deal_id}")
        
        return contract
    
    def sign_contract(self, deal_id: str, signer_type: str, 
                    signer_id: str) -> Dict[str, Any]:
        """
        Firma un contrato.
        
        Args:
            deal_id: ID del acuerdo
            signer_type: Tipo de firmante (creator, brand)
            signer_id: ID del firmante
            
        Returns:
            Estado de la firma
        """
        # Verificar que el acuerdo existe
        if deal_id not in self.deals:
            logger.warning(f"Acuerdo no encontrado: {deal_id}")
            return {
                "status": "error",
                "message": "Acuerdo no encontrado"
            }
        
        # Verificar que el acuerdo tiene contrato
        if "contract" not in self.deals[deal_id]:
            logger.warning(f"Acuerdo sin contrato: {deal_id}")
            return {
                "status": "error",
                "message": "El acuerdo no tiene un contrato generado"
            }
        
        # Validar tipo de firmante
        if signer_type not in ["creator", "brand"]:
            logger.warning(f"Tipo de firmante no válido: {signer_type}")
            return {
                "status": "error",
                "message": "Tipo de firmante no válido. Debe ser 'creator' o 'brand'"
            }
        
        # Verificar que el firmante es parte del acuerdo
        deal = self.deals[deal_id]
        if (signer_type == "creator" and signer_id != deal["creator_id"]) or \
           (signer_type == "brand" and signer_id != deal["brand_id"]):
            logger.warning(f"Firmante no autorizado: {signer_id}")
            return {
                "status": "error",
                "message": "El firmante no es parte del acuerdo"
            }
        
        # Registrar firma
        deal["contract"]["signatures"][signer_type] = {
            "signed": True,
            "date": datetime.now().isoformat()
        }
        
        # Verificar si ambas partes han firmado
        if deal["contract"]["signatures"]["creator"]["signed"] and \
           deal["contract"]["signatures"]["brand"]["signed"]:
            deal["contract"]["status"] = "active"
        
        # Guardar acuerdo
        self._save_deals()
        
        logger.info(f"Contrato firmado por {signer_type} {signer_id} para acuerdo {deal_id}")
        
        return {
            "status": "success",
            "message": "Contrato firmado correctamente",
            "contract_status": deal["contract"]["status"]
        }
    
    def export_marketplace_data(self, data_type: str = "all") -> Dict[str, Any]:
        """
        Exporta datos del marketplace para análisis.
        
        Args:
            data_type: Tipo de datos a exportar (all, creators, brands, deals, proposals, reviews)
            
        Returns:
            Datos exportados
        """
        valid_types = ["all", "creators", "brands", "deals", "proposals", "reviews"]
        
        if data_type not in valid_types:
            logger.warning(f"Tipo de datos no válido: {data_type}")
            return {
                "status": "error",
                "message": f"Tipo de datos no válido. Debe ser uno de: {', '.join(valid_types)}"
            }
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "export_type": data_type
        }
        
        if data_type == "all" or data_type == "creators":
            export_data["creators"] = self.creators
        
        if data_type == "all" or data_type == "brands":
            export_data["brands"] = self.brands
        
        if data_type == "all" or data_type == "deals":
            export_data["deals"] = self.deals
        
        if data_type == "all" or data_type == "proposals":
            export_data["proposals"] = self.proposals
        
        if data_type == "all" or data_type == "reviews":
            export_data["reviews"] = self.reviews
        
        logger.info(f"Datos del marketplace exportados: {data_type}")
        
        return export_data
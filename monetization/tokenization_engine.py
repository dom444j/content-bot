"""
Módulo para la tokenización de contenido y personajes.

Este módulo permite crear, gestionar y monetizar tokens digitales
asociados a personajes, contenido y canales del sistema.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/tokenization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TokenizationEngine")

class TokenizationEngine:
    """
    Motor de tokenización para monetizar personajes y contenido.
    
    Permite crear tokens digitales que representan propiedad o participación
    en personajes virtuales, canales o contenido específico, facilitando
    la monetización a través de la venta de tokens y la distribución
    de ingresos entre poseedores de tokens.
    """
    
    def __init__(self):
        """Inicializa el motor de tokenización."""
        self.tokens = self._load_tokens()
        self.token_holders = self._load_token_holders()
        self.token_transactions = self._load_token_transactions()
        self.revenue_distributions = self._load_revenue_distributions()
        
        # Asegurar que existan los directorios necesarios
        os.makedirs("data/tokenization", exist_ok=True)
        
        logger.info("TokenizationEngine inicializado")
    
    def _load_tokens(self) -> Dict[str, Dict[str, Any]]:
        """Carga datos de tokens."""
        try:
            tokens_path = "data/tokenization/tokens.json"
            if os.path.exists(tokens_path):
                with open(tokens_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error al cargar tokens: {e}")
            return {}
    
    def _load_token_holders(self) -> Dict[str, Dict[str, float]]:
        """Carga datos de poseedores de tokens."""
        try:
            holders_path = "data/tokenization/token_holders.json"
            if os.path.exists(holders_path):
                with open(holders_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return defaultdict(dict)
        except Exception as e:
            logger.error(f"Error al cargar poseedores de tokens: {e}")
            return defaultdict(dict)
    
    def _load_token_transactions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Carga historial de transacciones de tokens."""
        try:
            transactions_path = "data/tokenization/token_transactions.json"
            if os.path.exists(transactions_path):
                with open(transactions_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return defaultdict(list)
        except Exception as e:
            logger.error(f"Error al cargar transacciones de tokens: {e}")
            return defaultdict(list)
    
    def _load_revenue_distributions(self) -> List[Dict[str, Any]]:
        """Carga historial de distribuciones de ingresos."""
        try:
            distributions_path = "data/tokenization/revenue_distributions.json"
            if os.path.exists(distributions_path):
                with open(distributions_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error al cargar distribuciones de ingresos: {e}")
            return []
    
    def _save_tokens(self) -> None:
        """Guarda datos de tokens."""
        try:
            tokens_path = "data/tokenization/tokens.json"
            with open(tokens_path, "w", encoding="utf-8") as f:
                json.dump(self.tokens, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar tokens: {e}")
    
    def _save_token_holders(self) -> None:
        """Guarda datos de poseedores de tokens."""
        try:
            holders_path = "data/tokenization/token_holders.json"
            with open(holders_path, "w", encoding="utf-8") as f:
                json.dump(self.token_holders, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar poseedores de tokens: {e}")
    
    def _save_token_transactions(self) -> None:
        """Guarda historial de transacciones de tokens."""
        try:
            transactions_path = "data/tokenization/token_transactions.json"
            with open(transactions_path, "w", encoding="utf-8") as f:
                json.dump(self.token_transactions, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar transacciones de tokens: {e}")
    
    def _save_revenue_distributions(self) -> None:
        """Guarda historial de distribuciones de ingresos."""
        try:
            distributions_path = "data/tokenization/revenue_distributions.json"
            with open(distributions_path, "w", encoding="utf-8") as f:
                json.dump(self.revenue_distributions, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar distribuciones de ingresos: {e}")
    
    def create_token(self, 
                    name: str, 
                    symbol: str, 
                    asset_type: str, 
                    asset_id: str,
                    total_supply: int,
                    initial_price: float,
                    creator_id: str,
                    description: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    revenue_share: Optional[float] = 0.5) -> Dict[str, Any]:
        """
        Crea un nuevo token para un activo (personaje, canal, contenido).
        
        Args:
            name: Nombre del token
            symbol: Símbolo del token (3-5 caracteres)
            asset_type: Tipo de activo (character, channel, content)
            asset_id: ID del activo
            total_supply: Suministro total de tokens
            initial_price: Precio inicial por token
            creator_id: ID del creador
            description: Descripción del token (opcional)
            metadata: Metadatos adicionales (opcional)
            revenue_share: Porcentaje de ingresos compartidos (0-1)
            
        Returns:
            Datos del token creado
        """
        # Validar parámetros
        if not name or not symbol or not asset_type or not asset_id:
            logger.error("Parámetros incompletos para crear token")
            return {
                "status": "error",
                "message": "Parámetros incompletos"
            }
        
        # Validar tipo de activo
        valid_asset_types = ["character", "channel", "content", "collection"]
        if asset_type not in valid_asset_types:
            logger.error(f"Tipo de activo no válido: {asset_type}")
            return {
                "status": "error",
                "message": f"Tipo de activo no válido. Debe ser uno de: {', '.join(valid_asset_types)}"
            }
        
        # Validar suministro y precio
        if total_supply <= 0 or initial_price <= 0:
            logger.error(f"Valores no válidos: supply={total_supply}, price={initial_price}")
            return {
                "status": "error",
                "message": "El suministro y precio deben ser mayores que cero"
            }
        
        # Validar revenue share
        if revenue_share < 0 or revenue_share > 1:
            logger.error(f"Revenue share no válido: {revenue_share}")
            return {
                "status": "error",
                "message": "El revenue share debe estar entre 0 y 1"
            }
        
        # Generar ID único
        token_id = str(uuid.uuid4())
        
        # Crear token
        token = {
            "token_id": token_id,
            "name": name,
            "symbol": symbol.upper(),
            "asset_type": asset_type,
            "asset_id": asset_id,
            "total_supply": total_supply,
            "circulating_supply": 0,  # Inicialmente 0
            "reserved_supply": total_supply,  # Todo el suministro reservado
            "initial_price": initial_price,
            "current_price": initial_price,
            "creator_id": creator_id,
            "description": description or f"Token for {asset_type} {name}",
            "metadata": metadata or {},
            "revenue_share": revenue_share,
            "total_revenue_generated": 0.0,
            "total_revenue_distributed": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active"  # active, paused, deprecated
        }
        
        # Guardar token
        self.tokens[token_id] = token
        self._save_tokens()
        
        # Asignar tokens iniciales al creador (50% del suministro)
        creator_allocation = int(total_supply * 0.5)
        self.allocate_tokens(token_id, creator_id, creator_allocation, 0.0, "initial_allocation")
        
        logger.info(f"Token creado: {name} ({symbol}) para {asset_type} {asset_id}")
        
        return token
    
    def allocate_tokens(self, 
                       token_id: str, 
                       holder_id: str, 
                       amount: int,
                       price_per_token: float,
                       transaction_type: str,
                       notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Asigna tokens a un poseedor.
        
        Args:
            token_id: ID del token
            holder_id: ID del poseedor
            amount: Cantidad de tokens
            price_per_token: Precio por token
            transaction_type: Tipo de transacción (initial_allocation, purchase, reward, airdrop)
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos de la transacción
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        token = self.tokens[token_id]
        
        # Verificar que hay suficientes tokens disponibles
        if amount > token["reserved_supply"]:
            logger.error(f"Suministro insuficiente: {amount} > {token['reserved_supply']}")
            return {
                "status": "error",
                "message": "Suministro insuficiente"
            }
        
        # Actualizar suministros
        token["reserved_supply"] -= amount
        token["circulating_supply"] += amount
        token["updated_at"] = datetime.now().isoformat()
        
        # Actualizar poseedores
        if token_id not in self.token_holders:
            self.token_holders[token_id] = {}
        
        if holder_id not in self.token_holders[token_id]:
            self.token_holders[token_id][holder_id] = 0
        
        self.token_holders[token_id][holder_id] += amount
        
        # Registrar transacción
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "token_id": token_id,
            "holder_id": holder_id,
            "amount": amount,
            "price_per_token": price_per_token,
            "total_value": amount * price_per_token,
            "transaction_type": transaction_type,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        if token_id not in self.token_transactions:
            self.token_transactions[token_id] = []
        
        self.token_transactions[token_id].append(transaction)
        
        # Guardar cambios
        self._save_tokens()
        self._save_token_holders()
        self._save_token_transactions()
        
        logger.info(f"Tokens asignados: {amount} de {token['symbol']} a {holder_id}")
        
        return transaction
    
    def update_token_price(self, token_id: str, new_price: float) -> Dict[str, Any]:
        """
        Actualiza el precio de un token.
        
        Args:
            token_id: ID del token
            new_price: Nuevo precio
            
        Returns:
            Datos del token actualizado
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Validar precio
        if new_price <= 0:
            logger.error(f"Precio no válido: {new_price}")
            return {
                "status": "error",
                "message": "El precio debe ser mayor que cero"
            }
        
        # Actualizar precio
        old_price = self.tokens[token_id]["current_price"]
        self.tokens[token_id]["current_price"] = new_price
        self.tokens[token_id]["updated_at"] = datetime.now().isoformat()
        
        # Calcular cambio porcentual
        price_change = ((new_price - old_price) / old_price) * 100
        
        # Guardar token
        self._save_tokens()
        
        logger.info(f"Precio de token actualizado: {token_id} de {old_price} a {new_price} ({price_change:.2f}%)")
        
        return {
            "token_id": token_id,
            "symbol": self.tokens[token_id]["symbol"],
            "old_price": old_price,
            "new_price": new_price,
            "price_change_percent": price_change,
            "updated_at": self.tokens[token_id]["updated_at"]
        }
    
    def transfer_tokens(self, 
                       token_id: str, 
                       from_holder_id: str, 
                       to_holder_id: str, 
                       amount: int,
                       price_per_token: float,
                       notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Transfiere tokens entre poseedores.
        
        Args:
            token_id: ID del token
            from_holder_id: ID del poseedor origen
            to_holder_id: ID del poseedor destino
            amount: Cantidad de tokens
            price_per_token: Precio por token
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos de la transacción
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Verificar que el poseedor origen tiene suficientes tokens
        if token_id not in self.token_holders or \
           from_holder_id not in self.token_holders[token_id] or \
           self.token_holders[token_id][from_holder_id] < amount:
            logger.error(f"Tokens insuficientes: {from_holder_id} no tiene suficientes {token_id}")
            return {
                "status": "error",
                "message": "Tokens insuficientes"
            }
        
        # Actualizar poseedores
        self.token_holders[token_id][from_holder_id] -= amount
        
        if to_holder_id not in self.token_holders[token_id]:
            self.token_holders[token_id][to_holder_id] = 0
        
        self.token_holders[token_id][to_holder_id] += amount
        
        # Registrar transacción
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "token_id": token_id,
            "from_holder_id": from_holder_id,
            "to_holder_id": to_holder_id,
            "amount": amount,
            "price_per_token": price_per_token,
            "total_value": amount * price_per_token,
            "transaction_type": "transfer",
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        if token_id not in self.token_transactions:
            self.token_transactions[token_id] = []
        
        self.token_transactions[token_id].append(transaction)
        
        # Actualizar precio del token si es necesario
        if price_per_token != self.tokens[token_id]["current_price"]:
            self.update_token_price(token_id, price_per_token)
        
        # Guardar cambios
        self._save_token_holders()
        self._save_token_transactions()
        
        logger.info(f"Tokens transferidos: {amount} de {self.tokens[token_id]['symbol']} de {from_holder_id} a {to_holder_id}")
        
        return transaction
    
    def distribute_revenue(self, 
                          token_id: str, 
                          revenue_amount: float,
                          revenue_source: str,
                          source_id: Optional[str] = None,
                          notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Distribuye ingresos entre poseedores de tokens.
        
        Args:
            token_id: ID del token
            revenue_amount: Cantidad de ingresos
            revenue_source: Fuente de ingresos (ads, affiliate, sponsorship, etc.)
            source_id: ID de la fuente (opcional)
            notes: Notas adicionales (opcional)
            
        Returns:
            Datos de la distribución
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        token = self.tokens[token_id]
        
        # Verificar que hay poseedores
        if token_id not in self.token_holders or not self.token_holders[token_id]:
            logger.error(f"No hay poseedores para el token: {token_id}")
            return {
                "status": "error",
                "message": "No hay poseedores para distribuir ingresos"
            }
        
        # Calcular cantidad a distribuir según revenue_share
        distributable_amount = revenue_amount * token["revenue_share"]
        creator_amount = revenue_amount - distributable_amount
        
        # Calcular distribución por poseedor
        total_tokens = token["circulating_supply"]
        distributions = []
        
        for holder_id, token_amount in self.token_holders[token_id].items():
            if token_amount <= 0:
                continue
                
            # Calcular porcentaje de propiedad y cantidad a recibir
            ownership_percentage = token_amount / total_tokens
            holder_amount = distributable_amount * ownership_percentage
            
            # Registrar distribución
            distribution = {
                "holder_id": holder_id,
                "token_amount": token_amount,
                "ownership_percentage": ownership_percentage,
                "amount_received": holder_amount
            }
            
            distributions.append(distribution)
        
        # Crear registro de distribución
        distribution_record = {
            "distribution_id": str(uuid.uuid4()),
            "token_id": token_id,
            "token_symbol": token["symbol"],
            "revenue_amount": revenue_amount,
            "distributable_amount": distributable_amount,
            "creator_amount": creator_amount,
            "creator_id": token["creator_id"],
            "revenue_source": revenue_source,
            "source_id": source_id,
            "distributions": distributions,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        # Actualizar estadísticas del token
        token["total_revenue_generated"] += revenue_amount
        token["total_revenue_distributed"] += distributable_amount
        token["updated_at"] = datetime.now().isoformat()
        
        # Guardar distribución y actualizar token
        self.revenue_distributions.append(distribution_record)
        self._save_revenue_distributions()
        self._save_tokens()
        
        logger.info(f"Ingresos distribuidos: {distributable_amount} para token {token['symbol']}")
        
        return distribution_record
    
    def get_token_details(self, token_id: str) -> Dict[str, Any]:
        """
        Obtiene detalles completos de un token.
        
        Args:
            token_id: ID del token
            
        Returns:
            Detalles completos del token
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        token = self.tokens[token_id]
        
        # Obtener poseedores
        holders = []
        if token_id in self.token_holders:
            for holder_id, amount in self.token_holders[token_id].items():
                if amount <= 0:
                    continue
                    
                ownership_percentage = (amount / token["circulating_supply"]) * 100
                holders.append({
                    "holder_id": holder_id,
                    "amount": amount,
                    "ownership_percentage": ownership_percentage
                })
        
        # Ordenar poseedores por cantidad
        holders.sort(key=lambda x: x["amount"], reverse=True)
        
        # Obtener transacciones recientes
        recent_transactions = []
        if token_id in self.token_transactions:
            # Tomar las 10 transacciones más recientes
            transactions = sorted(
                self.token_transactions[token_id], 
                key=lambda x: x["timestamp"], 
                reverse=True
            )[:10]
            
            recent_transactions = transactions
        
        # Obtener distribuciones recientes
        recent_distributions = []
        for distribution in sorted(
            self.revenue_distributions, 
            key=lambda x: x["timestamp"], 
            reverse=True
        ):
            if distribution["token_id"] == token_id:
                recent_distributions.append(distribution)
                
                if len(recent_distributions) >= 5:
                    break
        
        # Construir respuesta detallada
        token_details = {
            **token,
            "holders": holders,
            "recent_transactions": recent_transactions,
            "recent_distributions": recent_distributions,
            "market_cap": token["current_price"] * token["circulating_supply"],
            "holder_count": len(holders)
        }
        
        return token_details
    
    def get_holder_portfolio(self, holder_id: str) -> Dict[str, Any]:
        """
        Obtiene el portafolio de tokens de un poseedor.
        
        Args:
            holder_id: ID del poseedor
            
        Returns:
            Portafolio de tokens
        """
        portfolio = {
            "holder_id": holder_id,
            "tokens": [],
            "total_value": 0.0,
            "total_tokens": 0
        }
        
        # Buscar tokens del poseedor
        for token_id, holders in self.token_holders.items():
            if holder_id in holders and holders[holder_id] > 0:
                token = self.tokens[token_id]
                
                token_amount = holders[holder_id]
                token_value = token_amount * token["current_price"]
                ownership_percentage = (token_amount / token["circulating_supply"]) * 100
                
                token_info = {
                    "token_id": token_id,
                    "name": token["name"],
                    "symbol": token["symbol"],
                    "amount": token_amount,
                    "price": token["current_price"],
                    "value": token_value,
                    "ownership_percentage": ownership_percentage,
                    "asset_type": token["asset_type"],
                    "asset_id": token["asset_id"]
                }
                
                portfolio["tokens"].append(token_info)
                portfolio["total_value"] += token_value
                portfolio["total_tokens"] += 1
        
        # Ordenar tokens por valor
        portfolio["tokens"].sort(key=lambda x: x["value"], reverse=True)
        
        return portfolio
    
    def get_token_price_history(self, token_id: str) -> Dict[str, Any]:
        """
        Obtiene el historial de precios de un token.
        
        Args:
            token_id: ID del token
            
        Returns:
            Historial de precios
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        token = self.tokens[token_id]
        
        # Inicializar con el precio inicial
        price_history = [{
            "price": token["initial_price"],
            "timestamp": token["created_at"]
        }]
        
        # Obtener cambios de precio de las transacciones
        if token_id in self.token_transactions:
            # Ordenar transacciones por fecha
            transactions = sorted(
                self.token_transactions[token_id], 
                key=lambda x: x["timestamp"]
            )
            
            # Extraer precios únicos
            current_price = token["initial_price"]
            
            for transaction in transactions:
                if "price_per_token" in transaction and transaction["price_per_token"] != current_price:
                    current_price = transaction["price_per_token"]
                    
                    price_history.append({
                        "price": current_price,
                        "timestamp": transaction["timestamp"]
                    })
        
        # Añadir precio actual si es diferente del último registrado
        if price_history[-1]["price"] != token["current_price"]:
            price_history.append({
                "price": token["current_price"],
                "timestamp": token["updated_at"]
            })
        
        return {
            "token_id": token_id,
            "symbol": token["symbol"],
            "name": token["name"],
            "initial_price": token["initial_price"],
            "current_price": token["current_price"],
            "price_change": ((token["current_price"] - token["initial_price"]) / token["initial_price"]) * 100,
            "price_history": price_history
        }
    
    def get_token_revenue_stats(self, token_id: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de ingresos de un token.
        
        Args:
            token_id: ID del token
            
        Returns:
            Estadísticas de ingresos
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        token = self.tokens[token_id]
        
        # Inicializar estadísticas
        stats = {
            "token_id": token_id,
            "symbol": token["symbol"],
            "name": token["name"],
            "total_revenue_generated": token["total_revenue_generated"],
            "total_revenue_distributed": token["total_revenue_distributed"],
            "creator_revenue": token["total_revenue_generated"] - token["total_revenue_distributed"],
            "revenue_share_percentage": token["revenue_share"] * 100,
            "distributions_count": 0,
            "revenue_by_source": defaultdict(float),
            "monthly_revenue": defaultdict(float),
            "recent_distributions": []
        }
        
        # Analizar distribuciones
        for distribution in self.revenue_distributions:
            if distribution["token_id"] == token_id:
                stats["distributions_count"] += 1
                
                # Agrupar por fuente
                source = distribution["revenue_source"]
                stats["revenue_by_source"][source] += distribution["revenue_amount"]
                
                # Agrupar por mes
                date = datetime.fromisoformat(distribution["timestamp"])
                month_key = f"{date.year}-{date.month:02d}"
                stats["monthly_revenue"][month_key] += distribution["revenue_amount"]
                
                # Añadir a recientes
                if len(stats["recent_distributions"]) < 5:
                    stats["recent_distributions"].append({
                        "distribution_id": distribution["distribution_id"],
                        "revenue_amount": distribution["revenue_amount"],
                        "distributable_amount": distribution["distributable_amount"],
                        "revenue_source": distribution["revenue_source"],
                        "timestamp": distribution["timestamp"]
                    })
        
        # Convertir defaultdicts a diccionarios normales
        stats["revenue_by_source"] = dict(stats["revenue_by_source"])
        stats["monthly_revenue"] = dict(stats["monthly_revenue"])
        
        # Ordenar distribuciones recientes
        stats["recent_distributions"].sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Calcular ROI para poseedores iniciales
        if token["initial_price"] > 0:
            stats["token_roi"] = ((token["current_price"] - token["initial_price"]) / token["initial_price"]) * 100
        else:
            stats["token_roi"] = 0
        
        return stats
    
    def search_tokens(self, 
                     filters: Dict[str, Any] = None, 
                     sort_by: str = "current_price", 
                     limit: int = 20) -> List[Dict[str, Any]]:
        """
        Busca tokens según filtros.
        
        Args:
            filters: Filtros de búsqueda (opcional)
            sort_by: Campo para ordenar (current_price, circulating_supply, total_revenue_generated)
            limit: Límite de resultados
            
        Returns:
            Lista de tokens que coinciden con los filtros
        """
        filters = filters or {}
        
        # Filtrar tokens
        filtered_tokens = []
        
        for token_id, token in self.tokens.items():
            # Filtrar por estado
                        if "status" in filters and token["status"] != filters["status"]:
                continue
            
            # Filtrar por tipo de activo
            if "asset_type" in filters and token["asset_type"] != filters["asset_type"]:
                continue
            
            # Filtrar por creador
            if "creator_id" in filters and token["creator_id"] != filters["creator_id"]:
                continue
            
            # Filtrar por rango de precio
            if "min_price" in filters and token["current_price"] < filters["min_price"]:
                continue
            
            if "max_price" in filters and token["current_price"] > filters["max_price"]:
                continue
            
            # Filtrar por texto en nombre o símbolo
            if "search_text" in filters and filters["search_text"]:
                search_text = filters["search_text"].lower()
                if search_text not in token["name"].lower() and search_text not in token["symbol"].lower():
                    continue
            
            # Añadir token simplificado a resultados
            filtered_tokens.append({
                "token_id": token_id,
                "name": token["name"],
                "symbol": token["symbol"],
                "asset_type": token["asset_type"],
                "current_price": token["current_price"],
                "circulating_supply": token["circulating_supply"],
                "market_cap": token["current_price"] * token["circulating_supply"],
                "total_revenue_generated": token["total_revenue_generated"],
                "creator_id": token["creator_id"],
                "created_at": token["created_at"],
                "status": token["status"]
            })
        
        # Validar campo de ordenación
        valid_sort_fields = ["current_price", "circulating_supply", "total_revenue_generated", 
                            "market_cap", "created_at"]
        
        if sort_by not in valid_sort_fields:
            sort_by = "current_price"
        
        # Ordenar resultados
        if sort_by == "market_cap":
            filtered_tokens.sort(key=lambda x: x["current_price"] * x["circulating_supply"], reverse=True)
        else:
            filtered_tokens.sort(key=lambda x: x[sort_by], reverse=True)
        
        # Limitar resultados
        return filtered_tokens[:limit]
    
    def pause_token(self, token_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Pausa un token para evitar nuevas transacciones.
        
        Args:
            token_id: ID del token
            reason: Motivo de la pausa (opcional)
            
        Returns:
            Estado del token
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Actualizar estado
        self.tokens[token_id]["status"] = "paused"
        self.tokens[token_id]["updated_at"] = datetime.now().isoformat()
        
        # Registrar motivo si se proporciona
        if reason:
            if "status_history" not in self.tokens[token_id]:
                self.tokens[token_id]["status_history"] = []
            
            self.tokens[token_id]["status_history"].append({
                "status": "paused",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })
        
        # Guardar token
        self._save_tokens()
        
        logger.info(f"Token pausado: {token_id} - {reason or 'Sin motivo especificado'}")
        
        return {
            "token_id": token_id,
            "name": self.tokens[token_id]["name"],
            "symbol": self.tokens[token_id]["symbol"],
            "status": "paused",
            "updated_at": self.tokens[token_id]["updated_at"]
        }
    
    def resume_token(self, token_id: str) -> Dict[str, Any]:
        """
        Reanuda un token pausado.
        
        Args:
            token_id: ID del token
            
        Returns:
            Estado del token
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Verificar que el token está pausado
        if self.tokens[token_id]["status"] != "paused":
            logger.warning(f"Token no está pausado: {token_id}")
            return {
                "status": "error",
                "message": "El token no está pausado"
            }
        
        # Actualizar estado
        self.tokens[token_id]["status"] = "active"
        self.tokens[token_id]["updated_at"] = datetime.now().isoformat()
        
        # Registrar cambio de estado
        if "status_history" not in self.tokens[token_id]:
            self.tokens[token_id]["status_history"] = []
        
        self.tokens[token_id]["status_history"].append({
            "status": "active",
            "reason": "Token reactivado",
            "timestamp": datetime.now().isoformat()
        })
        
        # Guardar token
        self._save_tokens()
        
        logger.info(f"Token reactivado: {token_id}")
        
        return {
            "token_id": token_id,
            "name": self.tokens[token_id]["name"],
            "symbol": self.tokens[token_id]["symbol"],
            "status": "active",
            "updated_at": self.tokens[token_id]["updated_at"]
        }
    
    def deprecate_token(self, token_id: str, reason: str) -> Dict[str, Any]:
        """
        Marca un token como obsoleto.
        
        Args:
            token_id: ID del token
            reason: Motivo de la obsolescencia
            
        Returns:
            Estado del token
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Actualizar estado
        self.tokens[token_id]["status"] = "deprecated"
        self.tokens[token_id]["updated_at"] = datetime.now().isoformat()
        
        # Registrar motivo
        if "status_history" not in self.tokens[token_id]:
            self.tokens[token_id]["status_history"] = []
        
        self.tokens[token_id]["status_history"].append({
            "status": "deprecated",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Guardar token
        self._save_tokens()
        
        logger.info(f"Token obsoleto: {token_id} - {reason}")
        
        return {
            "token_id": token_id,
            "name": self.tokens[token_id]["name"],
            "symbol": self.tokens[token_id]["symbol"],
            "status": "deprecated",
            "reason": reason,
            "updated_at": self.tokens[token_id]["updated_at"]
        }
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas generales del marketplace de tokens.
        
        Returns:
            Estadísticas del marketplace
        """
        stats = {
            "total_tokens": len(self.tokens),
            "active_tokens": 0,
            "paused_tokens": 0,
            "deprecated_tokens": 0,
            "total_market_cap": 0.0,
            "total_revenue_generated": 0.0,
            "total_revenue_distributed": 0.0,
            "total_transactions": 0,
            "tokens_by_asset_type": defaultdict(int),
            "top_tokens_by_market_cap": [],
            "top_tokens_by_revenue": [],
            "recent_transactions": []
        }
        
        # Analizar tokens
        for token_id, token in self.tokens.items():
            # Contar por estado
            if token["status"] == "active":
                stats["active_tokens"] += 1
            elif token["status"] == "paused":
                stats["paused_tokens"] += 1
            elif token["status"] == "deprecated":
                stats["deprecated_tokens"] += 1
            
            # Sumar capitalización de mercado
            market_cap = token["current_price"] * token["circulating_supply"]
            stats["total_market_cap"] += market_cap
            
            # Sumar ingresos
            stats["total_revenue_generated"] += token["total_revenue_generated"]
            stats["total_revenue_distributed"] += token["total_revenue_distributed"]
            
            # Contar por tipo de activo
            stats["tokens_by_asset_type"][token["asset_type"]] += 1
            
            # Añadir a top por capitalización de mercado
            stats["top_tokens_by_market_cap"].append({
                "token_id": token_id,
                "name": token["name"],
                "symbol": token["symbol"],
                "market_cap": market_cap,
                "current_price": token["current_price"],
                "circulating_supply": token["circulating_supply"]
            })
            
            # Añadir a top por ingresos
            stats["top_tokens_by_revenue"].append({
                "token_id": token_id,
                "name": token["name"],
                "symbol": token["symbol"],
                "total_revenue": token["total_revenue_generated"],
                "revenue_distributed": token["total_revenue_distributed"]
            })
        
        # Ordenar y limitar tops
        stats["top_tokens_by_market_cap"].sort(key=lambda x: x["market_cap"], reverse=True)
        stats["top_tokens_by_market_cap"] = stats["top_tokens_by_market_cap"][:10]
        
        stats["top_tokens_by_revenue"].sort(key=lambda x: x["total_revenue"], reverse=True)
        stats["top_tokens_by_revenue"] = stats["top_tokens_by_revenue"][:10]
        
        # Contar transacciones y obtener recientes
        all_transactions = []
        for token_id, transactions in self.token_transactions.items():
            stats["total_transactions"] += len(transactions)
            
            for transaction in transactions:
                transaction_with_token = {
                    **transaction,
                    "token_symbol": self.tokens[token_id]["symbol"] if token_id in self.tokens else "UNKNOWN"
                }
                all_transactions.append(transaction_with_token)
        
        # Ordenar transacciones por fecha y limitar
        all_transactions.sort(key=lambda x: x["timestamp"], reverse=True)
        stats["recent_transactions"] = all_transactions[:20]
        
        # Convertir defaultdicts a diccionarios normales
        stats["tokens_by_asset_type"] = dict(stats["tokens_by_asset_type"])
        
        return stats
    
    def airdrop_tokens(self, 
                      token_id: str, 
                      recipients: List[Dict[str, Any]],
                      notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Distribuye tokens a múltiples destinatarios (airdrop).
        
        Args:
            token_id: ID del token
            recipients: Lista de destinatarios con sus cantidades
                        [{"recipient_id": "id1", "amount": 10}, ...]
            notes: Notas adicionales (opcional)
            
        Returns:
            Resultados del airdrop
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        token = self.tokens[token_id]
        
        # Verificar que el token está activo
        if token["status"] != "active":
            logger.error(f"Token no está activo: {token_id}")
            return {
                "status": "error",
                "message": f"El token no está activo, estado actual: {token['status']}"
            }
        
        # Calcular total de tokens a distribuir
        total_tokens = sum(recipient["amount"] for recipient in recipients)
        
        # Verificar que hay suficientes tokens disponibles
        if total_tokens > token["reserved_supply"]:
            logger.error(f"Suministro insuficiente: {total_tokens} > {token['reserved_supply']}")
            return {
                "status": "error",
                "message": "Suministro insuficiente para el airdrop"
            }
        
        # Realizar airdrop
        airdrop_results = {
            "airdrop_id": str(uuid.uuid4()),
            "token_id": token_id,
            "token_symbol": token["symbol"],
            "total_tokens": total_tokens,
            "recipients_count": len(recipients),
            "successful_transfers": 0,
            "failed_transfers": 0,
            "transfers": []
        }
        
        for recipient in recipients:
            recipient_id = recipient["amount"]
            amount = recipient["amount"]
            
            # Asignar tokens
            result = self.allocate_tokens(
                token_id=token_id,
                holder_id=recipient_id,
                amount=amount,
                price_per_token=0.0,  # Airdrop es gratis
                transaction_type="airdrop",
                notes=notes
            )
            
            # Registrar resultado
            if "status" in result and result["status"] == "error":
                airdrop_results["failed_transfers"] += 1
                transfer_result = {
                    "recipient_id": recipient_id,
                    "amount": amount,
                    "success": False,
                    "error": result["message"]
                }
            else:
                airdrop_results["successful_transfers"] += 1
                transfer_result = {
                    "recipient_id": recipient_id,
                    "amount": amount,
                    "success": True,
                    "transaction_id": result["transaction_id"]
                }
            
            airdrop_results["transfers"].append(transfer_result)
        
        logger.info(f"Airdrop completado: {token_id} - {total_tokens} tokens a {len(recipients)} destinatarios")
        
        return airdrop_results
    
    def burn_tokens(self, 
                   token_id: str, 
                   holder_id: str, 
                   amount: int,
                   reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Quema (destruye) tokens de un poseedor.
        
        Args:
            token_id: ID del token
            holder_id: ID del poseedor
            amount: Cantidad de tokens a quemar
            reason: Motivo de la quema (opcional)
            
        Returns:
            Resultado de la quema
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Verificar que el poseedor tiene suficientes tokens
        if token_id not in self.token_holders or \
           holder_id not in self.token_holders[token_id] or \
           self.token_holders[token_id][holder_id] < amount:
            logger.error(f"Tokens insuficientes: {holder_id} no tiene suficientes {token_id}")
            return {
                "status": "error",
                "message": "Tokens insuficientes para quemar"
            }
        
        # Actualizar poseedores
        self.token_holders[token_id][holder_id] -= amount
        
        # Actualizar suministros
        token = self.tokens[token_id]
        token["circulating_supply"] -= amount
        token["updated_at"] = datetime.now().isoformat()
        
        # Registrar transacción
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "token_id": token_id,
            "holder_id": holder_id,
            "amount": amount,
            "price_per_token": 0.0,  # No hay precio en quema
            "total_value": 0.0,
            "transaction_type": "burn",
            "notes": reason or "Token burn",
            "timestamp": datetime.now().isoformat()
        }
        
        if token_id not in self.token_transactions:
            self.token_transactions[token_id] = []
        
        self.token_transactions[token_id].append(transaction)
        
        # Guardar cambios
        self._save_tokens()
        self._save_token_holders()
        self._save_token_transactions()
        
        logger.info(f"Tokens quemados: {amount} de {token['symbol']} de {holder_id}")
        
        return {
            "status": "success",
            "token_id": token_id,
            "symbol": token["symbol"],
            "holder_id": holder_id,
            "amount_burned": amount,
            "transaction_id": transaction["transaction_id"],
            "new_balance": self.token_holders[token_id][holder_id],
            "new_circulating_supply": token["circulating_supply"]
        }
    
    def mint_additional_tokens(self, 
                              token_id: str, 
                              amount: int,
                              reason: str) -> Dict[str, Any]:
        """
        Crea tokens adicionales (aumenta el suministro).
        
        Args:
            token_id: ID del token
            amount: Cantidad de tokens a crear
            reason: Motivo de la creación adicional
            
        Returns:
            Resultado de la creación
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Verificar que el token está activo
        token = self.tokens[token_id]
        if token["status"] != "active":
            logger.error(f"Token no está activo: {token_id}")
            return {
                "status": "error",
                "message": f"El token no está activo, estado actual: {token['status']}"
            }
        
        # Validar cantidad
        if amount <= 0:
            logger.error(f"Cantidad no válida: {amount}")
            return {
                "status": "error",
                "message": "La cantidad debe ser mayor que cero"
            }
        
        # Actualizar suministros
        token["total_supply"] += amount
        token["reserved_supply"] += amount
        token["updated_at"] = datetime.now().isoformat()
        
        # Registrar evento de creación
        if "supply_changes" not in token:
            token["supply_changes"] = []
        
        token["supply_changes"].append({
            "change_type": "mint",
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Guardar token
        self._save_tokens()
        
        logger.info(f"Tokens adicionales creados: {amount} para {token['symbol']}")
        
        return {
            "status": "success",
            "token_id": token_id,
            "symbol": token["symbol"],
            "amount_minted": amount,
            "new_total_supply": token["total_supply"],
            "new_reserved_supply": token["reserved_supply"]
        }
    
    def export_token_data(self, token_id: str) -> Dict[str, Any]:
        """
        Exporta todos los datos relacionados con un token.
        
        Args:
            token_id: ID del token
            
        Returns:
            Datos completos del token
        """
        # Verificar que el token existe
        if token_id not in self.tokens:
            logger.error(f"Token no encontrado: {token_id}")
            return {
                "status": "error",
                "message": "Token no encontrado"
            }
        
        # Recopilar datos
        export_data = {
            "token": self.tokens[token_id],
            "holders": self.token_holders.get(token_id, {}),
            "transactions": self.token_transactions.get(token_id, []),
            "distributions": []
        }
        
        # Filtrar distribuciones
        for distribution in self.revenue_distributions:
            if distribution["token_id"] == token_id:
                export_data["distributions"].append(distribution)
        
        # Añadir metadatos de exportación
        export_data["export_metadata"] = {
            "export_timestamp": datetime.now().isoformat(),
            "token_id": token_id,
            "token_name": self.tokens[token_id]["name"],
            "token_symbol": self.tokens[token_id]["symbol"]
        }
        
        return export_data
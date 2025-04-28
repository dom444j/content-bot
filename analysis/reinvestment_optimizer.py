import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import seaborn as sns
from scipy.optimize import minimize
import math

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/reinvestment_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reinvestment_optimizer")

class ReinvestmentOptimizer:
    """
    Optimiza la estrategia de reinversión de ingresos para maximizar el crecimiento
    y la rentabilidad a largo plazo.
    """
    
    def __init__(self, data_path: str = "data/analysis"):
        """
        Inicializa el optimizador de reinversión.
        
        Args:
            data_path: Ruta para almacenar datos de análisis
        """
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Cargar configuración
        self.config_path = "config/platforms.json"
        self.config = self._load_config()
        
        # Inicializar almacenamiento de datos
        self.revenue_data = self._load_data("revenue_data.json", {})
        self.cost_data = self._load_data("cost_data.json", {})
        self.investment_history = self._load_data("investment_history.json", {})
        self.optimization_results = self._load_data("optimization_results.json", {})
        
        # Importar adaptadores de plataforma
        self.platform_adapters = {}
        self._load_platform_adapters()
        
        # Configurar visualización
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Categorías de inversión predefinidas
        self.investment_categories = [
            "content_production",
            "advertising",
            "equipment",
            "staff",
            "software_tools",
            "education",
            "community_building",
            "collaborations"
        ]
        
        # Factores de impacto predeterminados (se pueden ajustar con datos reales)
        self.default_impact_factors = {
            "content_production": {
                "growth_impact": 0.4,
                "revenue_impact": 0.3,
                "time_to_effect": 7,  # días
                "diminishing_returns": 0.8
            },
            "advertising": {
                "growth_impact": 0.5,
                "revenue_impact": 0.2,
                "time_to_effect": 3,  # días
                "diminishing_returns": 0.7
            },
            "equipment": {
                "growth_impact": 0.2,
                "revenue_impact": 0.1,
                "time_to_effect": 30,  # días
                "diminishing_returns": 0.9
            },
            "staff": {
                "growth_impact": 0.3,
                "revenue_impact": 0.3,
                "time_to_effect": 14,  # días
                "diminishing_returns": 0.85
            },
            "software_tools": {
                "growth_impact": 0.2,
                "revenue_impact": 0.2,
                "time_to_effect": 7,  # días
                "diminishing_returns": 0.8
            },
            "education": {
                "growth_impact": 0.3,
                "revenue_impact": 0.2,
                "time_to_effect": 30,  # días
                "diminishing_returns": 0.9
            },
            "community_building": {
                "growth_impact": 0.4,
                "revenue_impact": 0.1,
                "time_to_effect": 14,  # días
                "diminishing_returns": 0.75
            },
            "collaborations": {
                "growth_impact": 0.5,
                "revenue_impact": 0.3,
                "time_to_effect": 7,  # días
                "diminishing_returns": 0.8
            }
                }
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de plataformas"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _load_data(self, filename: str, default: Any) -> Any:
        """Carga datos desde un archivo JSON"""
        filepath = os.path.join(self.data_path, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error al cargar {filename}: {str(e)}")
            return default
    
    def _save_data(self, filename: str, data: Any) -> bool:
        """Guarda datos en un archivo JSON"""
        filepath = os.path.join(self.data_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error al guardar {filename}: {str(e)}")
            return False
    
    def _load_platform_adapters(self):
        """Carga los adaptadores de plataforma disponibles"""
        try:
            # Importar adaptadores dinámicamente
            from platform_adapters.youtube_adapter import YouTubeAdapter
            from platform_adapters.tiktok_adapter import TikTokAdapter
            from platform_adapters.instagram_adapter import InstagramAdapter
            
            # Inicializar adaptadores con configuración
            self.platform_adapters["youtube"] = YouTubeAdapter(self.config.get("youtube", {}))
            self.platform_adapters["tiktok"] = TikTokAdapter(self.config.get("tiktok", {}))
            self.platform_adapters["instagram"] = InstagramAdapter(self.config.get("instagram", {}))
            
            logger.info(f"Adaptadores de plataforma cargados: {list(self.platform_adapters.keys())}")
        except ImportError as e:
            logger.warning(f"No se pudieron cargar todos los adaptadores: {str(e)}")
        except Exception as e:
            logger.error(f"Error al inicializar adaptadores: {str(e)}")
    
    def add_revenue_data(self, platform: str, channel_id: str, 
                        start_date: str, end_date: str, 
                        revenue_sources: Dict[str, float]) -> Dict[str, Any]:
        """
        Añade datos de ingresos para un canal y período específico.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram)
            channel_id: ID del canal
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            revenue_sources: Diccionario con fuentes de ingresos y montos
            
        Returns:
            Resultado de la operación
        """
        try:
            # Validar fechas
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            if end_dt < start_dt:
                return {
                    "status": "error",
                    "message": "La fecha de fin debe ser posterior a la fecha de inicio"
                }
            
            # Crear ID único para el registro
            record_id = f"{platform}_{channel_id}_{start_date}_{end_date}"
            
            # Calcular total de ingresos
            total_revenue = sum(revenue_sources.values())
            
            # Crear registro de ingresos
            revenue_record = {
                "record_id": record_id,
                "platform": platform,
                "channel_id": channel_id,
                "start_date": start_date,
                "end_date": end_date,
                "revenue_sources": revenue_sources,
                "total_revenue": total_revenue,
                "created_at": datetime.now().isoformat()
            }
            
            # Guardar registro
            self.revenue_data[record_id] = revenue_record
            self._save_data("revenue_data.json", self.revenue_data)
            
            logger.info(f"Datos de ingresos añadidos: {record_id}, Total: {total_revenue}")
            
            return {
                "status": "success",
                "record_id": record_id,
                "message": "Datos de ingresos añadidos correctamente",
                "total_revenue": total_revenue
            }
            
        except Exception as e:
            logger.error(f"Error al añadir datos de ingresos: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al añadir datos de ingresos: {str(e)}"
            }
    
    def add_cost_data(self, platform: str, channel_id: str, 
                     start_date: str, end_date: str, 
                     cost_categories: Dict[str, float]) -> Dict[str, Any]:
        """
        Añade datos de costos para un canal y período específico.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram)
            channel_id: ID del canal
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            cost_categories: Diccionario con categorías de costos y montos
            
        Returns:
            Resultado de la operación
        """
        try:
            # Validar fechas
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            if end_dt < start_dt:
                return {
                    "status": "error",
                    "message": "La fecha de fin debe ser posterior a la fecha de inicio"
                }
            
            # Crear ID único para el registro
            record_id = f"{platform}_{channel_id}_{start_date}_{end_date}"
            
            # Calcular total de costos
            total_cost = sum(cost_categories.values())
            
            # Crear registro de costos
            cost_record = {
                "record_id": record_id,
                "platform": platform,
                "channel_id": channel_id,
                "start_date": start_date,
                "end_date": end_date,
                "cost_categories": cost_categories,
                "total_cost": total_cost,
                "created_at": datetime.now().isoformat()
            }
            
            # Guardar registro
            self.cost_data[record_id] = cost_record
            self._save_data("cost_data.json", self.cost_data)
            
            logger.info(f"Datos de costos añadidos: {record_id}, Total: {total_cost}")
            
            return {
                "status": "success",
                "record_id": record_id,
                "message": "Datos de costos añadidos correctamente",
                "total_cost": total_cost
            }
            
        except Exception as e:
            logger.error(f"Error al añadir datos de costos: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al añadir datos de costos: {str(e)}"
            }
    
    def calculate_profit_margin(self, platform: str, channel_id: str, 
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Calcula el margen de beneficio para un canal y período específico.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram)
            channel_id: ID del canal
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            
        Returns:
            Análisis de margen de beneficio
        """
        try:
            # Validar fechas
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            if end_dt < start_dt:
                return {
                    "status": "error",
                    "message": "La fecha de fin debe ser posterior a la fecha de inicio"
                }
            
            # Crear ID para el período
            period_id = f"{platform}_{channel_id}_{start_date}_{end_date}"
            
            # Buscar datos de ingresos y costos
            revenue_record = self.revenue_data.get(period_id)
            cost_record = self.cost_data.get(period_id)
            
            if not revenue_record:
                return {
                    "status": "error",
                    "message": f"No se encontraron datos de ingresos para el período: {period_id}"
                }
            
            if not cost_record:
                return {
                    "status": "error",
                    "message": f"No se encontraron datos de costos para el período: {period_id}"
                }
            
            # Calcular margen de beneficio
            total_revenue = revenue_record["total_revenue"]
            total_cost = cost_record["total_cost"]
            
            profit = total_revenue - total_cost
            profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
            roi = (profit / total_cost) * 100 if total_cost > 0 else 0
            
            # Crear análisis
            profit_analysis = {
                "period_id": period_id,
                "platform": platform,
                "channel_id": channel_id,
                "start_date": start_date,
                "end_date": end_date,
                "total_revenue": total_revenue,
                "total_cost": total_cost,
                "profit": profit,
                "profit_margin": round(profit_margin, 2),
                "roi": round(roi, 2),
                "analysis_date": datetime.now().isoformat()
            }
            
            logger.info(f"Análisis de beneficio calculado: {period_id}, Margen: {profit_margin:.2f}%, ROI: {roi:.2f}%")
            
            return {
                "status": "success",
                "analysis": profit_analysis
            }
            
        except Exception as e:
            logger.error(f"Error al calcular margen de beneficio: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al calcular margen de beneficio: {str(e)}"
            }
    
    def optimize_reinvestment(self, platform: str, channel_id: str, 
                             available_amount: float, 
                             optimization_goal: str = "growth", 
                             time_horizon: int = 90,
                             constraints: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimiza la distribución de reinversión para maximizar el objetivo especificado.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram)
            channel_id: ID del canal
            available_amount: Cantidad disponible para reinversión
            optimization_goal: Objetivo de optimización ('growth', 'revenue', 'balanced')
            time_horizon: Horizonte temporal en días
            constraints: Restricciones por categoría (min, max)
            
        Returns:
            Plan de reinversión optimizado
        """
        if available_amount <= 0:
            return {
                "status": "error",
                "message": "La cantidad disponible debe ser mayor que cero"
            }
        
        try:
            # Obtener factores de impacto ajustados con datos históricos
            impact_factors = self._get_adjusted_impact_factors(platform, channel_id)
            
            # Establecer restricciones por defecto si no se proporcionan
            if constraints is None:
                constraints = {}
                for category in self.investment_categories:
                    constraints[category] = {
                        "min": 0.0,
                        "max": available_amount
                    }
            
            # Definir función objetivo según el objetivo de optimización
            if optimization_goal == "growth":
                def objective_function(allocations):
                    return -self._calculate_growth_impact(allocations, impact_factors, time_horizon)
            elif optimization_goal == "revenue":
                def objective_function(allocations):
                    return -self._calculate_revenue_impact(allocations, impact_factors, time_horizon)
            else:  # balanced
                def objective_function(allocations):
                    growth_impact = self._calculate_growth_impact(allocations, impact_factors, time_horizon)
                    revenue_impact = self._calculate_revenue_impact(allocations, impact_factors, time_horizon)
                    return -(growth_impact * 0.5 + revenue_impact * 0.5)
            
            # Definir restricciones para la optimización
            constraint_functions = []
            
            # Restricción de presupuesto total
            def budget_constraint(allocations):
                return available_amount - sum(allocations)
            
            constraint_functions.append({
                "type": "eq",
                "fun": budget_constraint
            })
            
            # Restricciones de límites por categoría
            bounds = []
            for i, category in enumerate(self.investment_categories):
                category_constraints = constraints.get(category, {"min": 0.0, "max": available_amount})
                bounds.append((category_constraints["min"], category_constraints["max"]))
            
            # Punto inicial: distribución uniforme
            initial_allocation = [available_amount / len(self.investment_categories)] * len(self.investment_categories)
            
            # Ejecutar optimización
            result = minimize(
                objective_function,
                initial_allocation,
                method="SLSQP",
                bounds=bounds,
                constraints=constraint_functions,
                options={"disp": True}
            )
            
            # Procesar resultados
            if result.success:
                # Redondear asignaciones a 2 decimales
                optimized_allocations = [round(alloc, 2) for alloc in result.x]
                
                # Crear plan de reinversión
                reinvestment_plan = {}
                for i, category in enumerate(self.investment_categories):
                    reinvestment_plan[category] = optimized_allocations[i]
                
                # Calcular impactos esperados
                expected_growth_impact = self._calculate_growth_impact(optimized_allocations, impact_factors, time_horizon)
                expected_revenue_impact = self._calculate_revenue_impact(optimized_allocations, impact_factors, time_horizon)
                
                # Crear registro de optimización
                optimization_id = f"{platform}_{channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                optimization_record = {
                    "optimization_id": optimization_id,
                    "platform": platform,
                    "channel_id": channel_id,
                    "available_amount": available_amount,
                    "optimization_goal": optimization_goal,
                    "time_horizon": time_horizon,
                    "constraints": constraints,
                    "reinvestment_plan": reinvestment_plan,
                    "expected_growth_impact": round(expected_growth_impact, 2),
                    "expected_revenue_impact": round(expected_revenue_impact, 2),
                    "optimization_date": datetime.now().isoformat()
                }
                
                # Guardar resultado
                self.optimization_results[optimization_id] = optimization_record
                self._save_data("optimization_results.json", self.optimization_results)
                
                # Generar visualización
                visualization_path = self._generate_optimization_charts(optimization_record)
                optimization_record["visualization_path"] = visualization_path
                
                logger.info(f"Plan de reinversión optimizado: {optimization_id}")
                
                return {
                    "status": "success",
                    "optimization_id": optimization_id,
                    "reinvestment_plan": reinvestment_plan,
                    "expected_impacts": {
                        "growth": round(expected_growth_impact, 2),
                        "revenue": round(expected_revenue_impact, 2)
                    },
                    "visualization_path": visualization_path
                }
            else:
                return {
                    "status": "error",
                    "message": f"La optimización no convergió: {result.message}"
                }
            
        except Exception as e:
            logger.error(f"Error al optimizar reinversión: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al optimizar reinversión: {str(e)}"
            }
    
    def _get_adjusted_impact_factors(self, platform: str, channel_id: str) -> Dict[str, Dict[str, float]]:
        """
        Obtiene factores de impacto ajustados con datos históricos.
        
        Args:
            platform: Plataforma
            channel_id: ID del canal
            
        Returns:
            Factores de impacto ajustados
        """
        # Copiar factores predeterminados
        adjusted_factors = self.default_impact_factors.copy()
        
        # Buscar inversiones históricas para este canal
        channel_investments = {}
        for inv_id, inv_data in self.investment_history.items():
            if inv_data["platform"] == platform and inv_data["channel_id"] == channel_id:
                channel_investments[inv_id] = inv_data
        
        # Si no hay suficientes datos históricos, usar valores predeterminados
        if len(channel_investments) < 3:
            logger.info(f"Usando factores de impacto predeterminados para {platform}_{channel_id}")
            return adjusted_factors
        
        try:
            # Calcular ajustes basados en resultados históricos
            for category in self.investment_categories:
                category_investments = []
                category_growth_impacts = []
                category_revenue_impacts = []
                
                for inv_data in channel_investments.values():
                    if category in inv_data.get("allocation", {}) and inv_data.get("measured_impacts"):
                        amount = inv_data["allocation"][category]
                        growth_impact = inv_data["measured_impacts"].get("growth", 0)
                        revenue_impact = inv_data["measured_impacts"].get("revenue", 0)
                        
                        if amount > 0:
                            category_investments.append(amount)
                            category_growth_impacts.append(growth_impact)
                            category_revenue_impacts.append(revenue_impact)
                
                # Ajustar factores si hay suficientes datos
                if len(category_investments) >= 3:
                    # Calcular correlaciones
                    growth_correlation = np.corrcoef(category_investments, category_growth_impacts)[0, 1]
                    revenue_correlation = np.corrcoef(category_investments, category_revenue_impacts)[0, 1]
                    
                    # Ajustar factores basados en correlaciones
                    if not np.isnan(growth_correlation):
                        adjusted_factors[category]["growth_impact"] = (
                            adjusted_factors[category]["growth_impact"] * 0.7 + 
                            abs(growth_correlation) * 0.3
                        )
                    
                    if not np.isnan(revenue_correlation):
                        adjusted_factors[category]["revenue_impact"] = (
                            adjusted_factors[category]["revenue_impact"] * 0.7 + 
                            abs(revenue_correlation) * 0.3
                        )
                    
                    logger.info(f"Factores ajustados para {category} en {platform}_{channel_id}")
            
            return adjusted_factors
            
        except Exception as e:
            logger.warning(f"Error al ajustar factores de impacto: {str(e)}")
            return self.default_impact_factors
    
    def _calculate_growth_impact(self, allocations: List[float], 
                                impact_factors: Dict[str, Dict[str, float]], 
                                time_horizon: int) -> float:
        """
        Calcula el impacto esperado en crecimiento de una asignación.
        
        Args:
            allocations: Lista de asignaciones por categoría
            impact_factors: Factores de impacto
            time_horizon: Horizonte temporal en días
            
        Returns:
            Impacto esperado en crecimiento
        """
        total_impact = 0.0
        
        for i, category in enumerate(self.investment_categories):
            amount = allocations[i]
            factor = impact_factors[category]
            
            # Aplicar rendimientos decrecientes
            diminishing_return = factor["diminishing_returns"]
            effective_amount = amount ** diminishing_return
            
            # Calcular impacto base
            base_impact = effective_amount * factor["growth_impact"]
            
            # Ajustar por tiempo hasta efecto
            time_factor = min(1.0, time_horizon / (factor["time_to_effect"] * 3))
            
            # Impacto total para esta categoría
            category_impact = base_impact * time_factor
            total_impact += category_impact
        
        return total_impact
    
    def _calculate_revenue_impact(self, allocations: List[float], 
                                 impact_factors: Dict[str, Dict[str, float]], 
                                 time_horizon: int) -> float:
        """
        Calcula el impacto esperado en ingresos de una asignación.
        
        Args:
            allocations: Lista de asignaciones por categoría
            impact_factors: Factores de impacto
            time_horizon: Horizonte temporal en días
            
        Returns:
            Impacto esperado en ingresos
        """
        total_impact = 0.0
        
        for i, category in enumerate(self.investment_categories):
            amount = allocations[i]
            factor = impact_factors[category]
            
            # Aplicar rendimientos decrecientes
            diminishing_return = factor["diminishing_returns"]
            effective_amount = amount ** diminishing_return
            
            # Calcular impacto base
            base_impact = effective_amount * factor["revenue_impact"]
            
            # Ajustar por tiempo hasta efecto
            time_factor = min(1.0, time_horizon / (factor["time_to_effect"] * 3))
            
            # Impacto total para esta categoría
            category_impact = base_impact * time_factor
            total_impact += category_impact
        
        return total_impact
    
    def _generate_optimization_charts(self, optimization_record: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera gráficos para visualizar el plan de reinversión optimizado.
        
        Args:
            optimization_record: Registro de optimización
            
        Returns:
            Rutas de los archivos de visualización
        """
        # Crear directorio para visualizaciones
        viz_dir = os.path.join(self.data_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Timestamp para nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimization_id = optimization_record["optimization_id"]
        
        # Diccionario para almacenar rutas
        visualization_paths = {}
        
        # Gráfico de pastel para distribución de reinversión
        plt.figure(figsize=(12, 8))
        
        reinvestment_plan = optimization_record["reinvestment_plan"]
        categories = list(reinvestment_plan.keys())
        amounts = list(reinvestment_plan.values())
        
        # Filtrar categorías con asignación cero
        non_zero_categories = []
        non_zero_amounts = []
        for i, amount in enumerate(amounts):
            if amount > 0:
                non_zero_categories.append(categories[i])
                non_zero_amounts.append(amount)
        
        # Crear gráfico de pastel
        plt.pie(non_zero_amounts, labels=non_zero_categories, autopct='%1.1f%%', 
                startangle=90, shadow=True)
        plt.axis('equal')
        plt.title(f"Optimal Reinvestment Distribution - {optimization_id}", fontsize=16)
        
        # Guardar gráfico
        pie_path = os.path.join(viz_dir, f"reinvestment_pie_{optimization_id}_{timestamp}.png")
        plt.savefig(pie_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["pie_chart"] = pie_path
        
        # Gráfico de barras para comparar impactos esperados
        plt.figure(figsize=(10, 6))
        
        impact_types = ["Growth Impact", "Revenue Impact"]
        impact_values = [
            optimization_record["expected_growth_impact"],
            optimization_record["expected_revenue_impact"]
        ]
        
        bars = plt.bar(impact_types, impact_values, color=['blue', 'green'])
        
        plt.title(f"Expected Impacts - {optimization_id}", fontsize=16)
        plt.ylabel("Impact Score", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Guardar gráfico
        impact_path = os.path.join(viz_dir, f"expected_impacts_{optimization_id}_{timestamp}.png")
        plt.savefig(impact_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["impact_chart"] = impact_path
        
        # Gráfico de barras horizontales para mostrar asignaciones por categoría
        plt.figure(figsize=(12, 8))
        
        # Ordenar categorías por monto asignado
        sorted_indices = np.argsort(amounts)
        sorted_categories = [categories[i] for i in sorted_indices]
        sorted_amounts = [amounts[i] for i in sorted_indices]
        
        # Crear gráfico de barras horizontales
        bars = plt.barh(sorted_categories, sorted_amounts, color='skyblue')
        
        plt.title(f"Reinvestment Allocation by Category - {optimization_id}", fontsize=16)
        plt.xlabel("Amount", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'${width:.2f}', ha='left', va='center')
        
        # Guardar gráfico
        alloc_path = os.path.join(viz_dir, f"allocation_bars_{optimization_id}_{timestamp}.png")
        plt.savefig(alloc_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["allocation_chart"] = alloc_path
        
        return visualization_paths
    
    def record_investment(self, optimization_id: str, 
                         actual_allocation: Dict[str, float],
                         notes: str = "") -> Dict[str, Any]:
        """
        Registra una inversión realizada basada en un plan de optimización.
        
        Args:
            optimization_id: ID del plan de optimización
            actual_allocation: Asignación real por categoría
            notes: Notas adicionales
            
        Returns:
            Resultado del registro
        """
        if optimization_id not in self.optimization_results:
            return {
                "status": "error",
                "message": f"Plan de optimización no encontrado: {optimization_id}"
            }
        
        try:
            # Obtener plan de optimización
            optimization_record = self.optimization_results[optimization_id]
            
            # Crear ID único para la inversión
            investment_id = f"inv_{optimization_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calcular total invertido
            total_invested = sum(actual_allocation.values())
            
            # Crear registro de inversión
            investment_record = {
                "investment_id": investment_id,
                "optimization_id": optimization_id,
                "platform": optimization_record["platform"],
                "channel_id": optimization_record["channel_id"],
                "allocation": actual_allocation,
                "total_invested": total_invested,
                "investment_date": datetime.now().isoformat(),
                "notes": notes,
                "measured_impacts": None  # Se actualizará más tarde
            }
            
            # Guardar registro
            self.investment_history[investment_id] = investment_record
            self._save_data("investment_history.json", self.investment_history)
            
            logger.info(f"Inversión registrada: {investment_id}, Total: {total_invested}")
            
            return {
                "status": "success",
                "investment_id": investment_id,
                "message": "Inversión registrada correctamente"
            }
            
        except Exception as e:
            logger.error(f"Error al registrar inversión: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al registrar inversión: {str(e)}"
            }
    
    def record_investment_impact(self, investment_id: str, 
                               measured_impacts: Dict[str, float]) -> Dict[str, Any]:
        """
        Registra el impacto medido de una inversión.
        
        Args:
            investment_id: ID de la inversión
            measured_impacts: Impactos medidos (growth, revenue)
            
        Returns:
            Resultado del registro
        """
        if investment_id not in self.investment_history:
            return {
                "status": "error",
                "message": f"Inversión no encontrada: {investment_id}"
            }
        
        try:
            # Actualizar registro de inversión
            self.investment_history[investment_id]["measured_impacts"] = measured_impacts
            self._save_data("investment_history.json", self.investment_history)
            
            logger.info(f"Impacto de inversión registrado: {investment_id}")
            
            return {
                "status": "success",
                "investment_id": investment_id,
                "message": "Impacto de inversión registrado correctamente"
            }
            
        except Exception as e:
            logger.error(f"Error al registrar impacto de inversión: {str(e)}")
                        return {
                "status": "error",
                "message": f"Error al registrar impacto de inversión: {str(e)}"
            }
    
    def get_investment_history(self, platform: str = None, channel_id: str = None, 
                              start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Obtiene el historial de inversiones con filtros opcionales.
        
        Args:
            platform: Filtrar por plataforma
            channel_id: Filtrar por ID de canal
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            
        Returns:
            Historial de inversiones filtrado
        """
        try:
            filtered_history = {}
            
            # Convertir fechas si se proporcionan
            start_dt = datetime.fromisoformat(start_date) if start_date else None
            end_dt = datetime.fromisoformat(end_date) if end_date else None
            
            # Filtrar inversiones
            for inv_id, inv_data in self.investment_history.items():
                # Filtrar por plataforma
                if platform and inv_data["platform"] != platform:
                    continue
                
                # Filtrar por canal
                if channel_id and inv_data["channel_id"] != channel_id:
                    continue
                
                # Filtrar por fecha
                if start_dt or end_dt:
                    inv_date = datetime.fromisoformat(inv_data["investment_date"].split("T")[0])
                    
                    if start_dt and inv_date < start_dt:
                        continue
                    
                    if end_dt and inv_date > end_dt:
                        continue
                
                # Añadir a resultados filtrados
                filtered_history[inv_id] = inv_data
            
            return {
                "status": "success",
                "count": len(filtered_history),
                "investments": filtered_history
            }
            
        except Exception as e:
            logger.error(f"Error al obtener historial de inversiones: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener historial de inversiones: {str(e)}"
            }
    
    def analyze_investment_performance(self, platform: str = None, 
                                      channel_id: str = None,
                                      time_period: int = 90) -> Dict[str, Any]:
        """
        Analiza el rendimiento de las inversiones realizadas.
        
        Args:
            platform: Filtrar por plataforma
            channel_id: Filtrar por ID de canal
            time_period: Período de análisis en días
            
        Returns:
            Análisis de rendimiento de inversiones
        """
        try:
            # Obtener inversiones relevantes
            history_result = self.get_investment_history(platform, channel_id)
            
            if history_result["status"] != "success":
                return history_result
            
            investments = history_result["investments"]
            
            if not investments:
                return {
                    "status": "warning",
                    "message": "No se encontraron inversiones para analizar"
                }
            
            # Filtrar inversiones con impactos medidos
            measured_investments = {}
            for inv_id, inv_data in investments.items():
                if inv_data.get("measured_impacts"):
                    measured_investments[inv_id] = inv_data
            
            if not measured_investments:
                return {
                    "status": "warning",
                    "message": "No se encontraron inversiones con impactos medidos"
                }
            
            # Analizar rendimiento por categoría
            category_performance = {}
            for category in self.investment_categories:
                category_investments = []
                category_growth_impacts = []
                category_revenue_impacts = []
                
                for inv_data in measured_investments.values():
                    if category in inv_data.get("allocation", {}) and inv_data["allocation"][category] > 0:
                        amount = inv_data["allocation"][category]
                        growth_impact = inv_data["measured_impacts"].get("growth", 0)
                        revenue_impact = inv_data["measured_impacts"].get("revenue", 0)
                        
                        category_investments.append(amount)
                        category_growth_impacts.append(growth_impact)
                        category_revenue_impacts.append(revenue_impact)
                
                if category_investments:
                    # Calcular métricas de rendimiento
                    avg_amount = sum(category_investments) / len(category_investments)
                    avg_growth_impact = sum(category_growth_impacts) / len(category_growth_impacts)
                    avg_revenue_impact = sum(category_revenue_impacts) / len(category_revenue_impacts)
                    
                    # Calcular ROI
                    growth_roi = avg_growth_impact / avg_amount if avg_amount > 0 else 0
                    revenue_roi = avg_revenue_impact / avg_amount if avg_amount > 0 else 0
                    
                    category_performance[category] = {
                        "investment_count": len(category_investments),
                        "total_invested": sum(category_investments),
                        "avg_investment": avg_amount,
                        "avg_growth_impact": avg_growth_impact,
                        "avg_revenue_impact": avg_revenue_impact,
                        "growth_roi": growth_roi,
                        "revenue_roi": revenue_roi
                    }
            
            # Identificar categorías de mejor y peor rendimiento
            if category_performance:
                # Mejor categoría para crecimiento
                best_growth_category = max(
                    category_performance.items(),
                    key=lambda x: x[1]["growth_roi"]
                )[0]
                
                # Mejor categoría para ingresos
                best_revenue_category = max(
                    category_performance.items(),
                    key=lambda x: x[1]["revenue_roi"]
                )[0]
                
                # Peor categoría para crecimiento
                worst_growth_category = min(
                    category_performance.items(),
                    key=lambda x: x[1]["growth_roi"]
                )[0]
                
                # Peor categoría para ingresos
                worst_revenue_category = min(
                    category_performance.items(),
                    key=lambda x: x[1]["revenue_roi"]
                )[0]
                
                # Crear resumen
                performance_summary = {
                    "best_performers": {
                        "growth": best_growth_category,
                        "revenue": best_revenue_category
                    },
                    "worst_performers": {
                        "growth": worst_growth_category,
                        "revenue": worst_revenue_category
                    }
                }
            else:
                performance_summary = {}
            
            # Generar visualizaciones
            visualization_paths = self._generate_performance_charts(category_performance)
            
            return {
                "status": "success",
                "analysis_date": datetime.now().isoformat(),
                "investments_analyzed": len(measured_investments),
                "category_performance": category_performance,
                "performance_summary": performance_summary,
                "visualization_paths": visualization_paths
            }
            
        except Exception as e:
            logger.error(f"Error al analizar rendimiento de inversiones: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar rendimiento de inversiones: {str(e)}"
            }
    
    def _generate_performance_charts(self, category_performance: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Genera gráficos para visualizar el rendimiento de inversiones.
        
        Args:
            category_performance: Datos de rendimiento por categoría
            
        Returns:
            Rutas de los archivos de visualización
        """
        # Crear directorio para visualizaciones
        viz_dir = os.path.join(self.data_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Timestamp para nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Diccionario para almacenar rutas
        visualization_paths = {}
        
        if not category_performance:
            return visualization_paths
        
        # Extraer datos para gráficos
        categories = list(category_performance.keys())
        growth_rois = [category_performance[cat]["growth_roi"] for cat in categories]
        revenue_rois = [category_performance[cat]["revenue_roi"] for cat in categories]
        total_invested = [category_performance[cat]["total_invested"] for cat in categories]
        
        # Gráfico de barras para ROI de crecimiento
        plt.figure(figsize=(12, 8))
        bars = plt.bar(categories, growth_rois, color='blue', alpha=0.7)
        
        plt.title("Growth ROI by Investment Category", fontsize=16)
        plt.ylabel("Growth ROI", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Guardar gráfico
        growth_roi_path = os.path.join(viz_dir, f"growth_roi_by_category_{timestamp}.png")
        plt.savefig(growth_roi_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["growth_roi_chart"] = growth_roi_path
        
        # Gráfico de barras para ROI de ingresos
        plt.figure(figsize=(12, 8))
        bars = plt.bar(categories, revenue_rois, color='green', alpha=0.7)
        
        plt.title("Revenue ROI by Investment Category", fontsize=16)
        plt.ylabel("Revenue ROI", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Guardar gráfico
        revenue_roi_path = os.path.join(viz_dir, f"revenue_roi_by_category_{timestamp}.png")
        plt.savefig(revenue_roi_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["revenue_roi_chart"] = revenue_roi_path
        
        # Gráfico de dispersión para comparar ROI de crecimiento vs. ingresos
        plt.figure(figsize=(10, 8))
        
        # Tamaño de los puntos proporcional a la inversión total
        sizes = [inv/max(total_invested)*500 if max(total_invested) > 0 else 100 for inv in total_invested]
        
        plt.scatter(growth_rois, revenue_rois, s=sizes, alpha=0.6, c=range(len(categories)), cmap='viridis')
        
        # Añadir etiquetas a los puntos
        for i, cat in enumerate(categories):
            plt.annotate(cat, (growth_rois[i], revenue_rois[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title("Growth ROI vs Revenue ROI by Category", fontsize=16)
        plt.xlabel("Growth ROI", fontsize=12)
        plt.ylabel("Revenue ROI", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir líneas de referencia
        if growth_rois and revenue_rois:
            avg_growth_roi = sum(growth_rois) / len(growth_rois)
            avg_revenue_roi = sum(revenue_rois) / len(revenue_rois)
            
            plt.axvline(x=avg_growth_roi, color='blue', linestyle='--', alpha=0.5)
            plt.axhline(y=avg_revenue_roi, color='green', linestyle='--', alpha=0.5)
            
            # Añadir anotaciones para los cuadrantes
            max_growth = max(growth_rois)
            max_revenue = max(revenue_rois)
            
            plt.text(max_growth*0.75, max_revenue*0.75, "High Growth & Revenue", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(avg_growth_roi*0.5, max_revenue*0.75, "High Revenue, Low Growth", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(max_growth*0.75, avg_revenue_roi*0.5, "High Growth, Low Revenue", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(avg_growth_roi*0.5, avg_revenue_roi*0.5, "Low Growth & Revenue", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        
        # Guardar gráfico
        scatter_path = os.path.join(viz_dir, f"roi_comparison_scatter_{timestamp}.png")
        plt.savefig(scatter_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["roi_comparison_chart"] = scatter_path
        
        return visualization_paths
    
    def generate_reinvestment_recommendation(self, platform: str, channel_id: str, 
                                           available_amount: float,
                                           time_horizon: int = 90) -> Dict[str, Any]:
        """
        Genera una recomendación de reinversión basada en el análisis de rendimiento histórico.
        
        Args:
            platform: Plataforma
            channel_id: ID del canal
            available_amount: Cantidad disponible para reinversión
            time_horizon: Horizonte temporal en días
            
        Returns:
            Recomendación de reinversión
        """
        if available_amount <= 0:
            return {
                "status": "error",
                "message": "La cantidad disponible debe ser mayor que cero"
            }
        
        try:
            # Analizar rendimiento histórico
            performance_analysis = self.analyze_investment_performance(platform, channel_id)
            
            if performance_analysis["status"] not in ["success", "warning"]:
                # Si no hay datos históricos, usar optimización estándar
                return self.optimize_reinvestment(
                    platform=platform,
                    channel_id=channel_id,
                    available_amount=available_amount,
                    optimization_goal="balanced",
                    time_horizon=time_horizon
                )
            
            # Crear restricciones basadas en rendimiento histórico
            constraints = {}
            
            if "category_performance" in performance_analysis:
                category_performance = performance_analysis.get("category_performance", {})
                
                for category in self.investment_categories:
                    # Valores predeterminados
                    min_allocation = 0.0
                    max_allocation = available_amount
                    
                    # Ajustar basado en rendimiento histórico
                    if category in category_performance:
                        perf = category_performance[category]
                        combined_roi = (perf["growth_roi"] + perf["revenue_roi"]) / 2
                        
                        # Categorías de alto rendimiento reciben más presupuesto
                        if combined_roi > 0.1:  # Umbral de buen rendimiento
                            min_allocation = available_amount * 0.1  # Al menos 10%
                        
                        # Categorías de bajo rendimiento reciben menos presupuesto
                        if combined_roi < 0.05:  # Umbral de bajo rendimiento
                            max_allocation = available_amount * 0.1  # Máximo 10%
                    
                    constraints[category] = {
                        "min": min_allocation,
                        "max": max_allocation
                    }
            
            # Determinar objetivo de optimización basado en rendimiento histórico
            optimization_goal = "balanced"  # Valor predeterminado
            
            if "performance_summary" in performance_analysis:
                # Contar categorías de mejor rendimiento
                best_performers = performance_analysis["performance_summary"].get("best_performers", {})
                
                # Si la misma categoría es la mejor para crecimiento e ingresos
                if best_performers.get("growth") == best_performers.get("revenue"):
                    # Enfocarse en esa categoría
                    best_category = best_performers.get("growth")
                    if best_category:
                        constraints[best_category] = {
                            "min": available_amount * 0.3,  # Al menos 30%
                            "max": available_amount
                        }
            
            # Ejecutar optimización con restricciones personalizadas
            optimization_result = self.optimize_reinvestment(
                platform=platform,
                channel_id=channel_id,
                available_amount=available_amount,
                optimization_goal=optimization_goal,
                time_horizon=time_horizon,
                constraints=constraints
            )
            
            # Añadir información de recomendación
            if optimization_result["status"] == "success":
                optimization_result["recommendation_basis"] = "historical_performance"
                optimization_result["performance_analysis"] = performance_analysis
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error al generar recomendación de reinversión: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al generar recomendación de reinversión: {str(e)}"
            }
    
    def export_analysis_report(self, platform: str, channel_id: str, 
                              start_date: str, end_date: str,
                              output_format: str = "json") -> Dict[str, Any]:
        """
        Exporta un informe completo de análisis para un canal y período específico.
        
        Args:
            platform: Plataforma
            channel_id: ID del canal
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            output_format: Formato de salida ('json', 'csv', 'excel')
            
        Returns:
            Resultado de la exportación
        """
        try:
            # Validar fechas
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            if end_dt < start_dt:
                return {
                    "status": "error",
                    "message": "La fecha de fin debe ser posterior a la fecha de inicio"
                }
            
            # Crear ID para el informe
            report_id = f"report_{platform}_{channel_id}_{start_date}_{end_date}"
            
            # Recopilar datos para el informe
            report_data = {
                "report_id": report_id,
                "platform": platform,
                "channel_id": channel_id,
                "start_date": start_date,
                "end_date": end_date,
                "generated_at": datetime.now().isoformat(),
                "revenue_data": {},
                "cost_data": {},
                "profit_analysis": {},
                "investment_history": {},
                "performance_analysis": {}
            }
            
            # Obtener datos de ingresos
            period_id = f"{platform}_{channel_id}_{start_date}_{end_date}"
            if period_id in self.revenue_data:
                report_data["revenue_data"] = self.revenue_data[period_id]
            
            # Obtener datos de costos
            if period_id in self.cost_data:
                report_data["cost_data"] = self.cost_data[period_id]
            
            # Calcular análisis de beneficio
            profit_result = self.calculate_profit_margin(platform, channel_id, start_date, end_date)
            if profit_result["status"] == "success":
                report_data["profit_analysis"] = profit_result["analysis"]
            
            # Obtener historial de inversiones
            history_result = self.get_investment_history(platform, channel_id, start_date, end_date)
            if history_result["status"] == "success":
                report_data["investment_history"] = history_result["investments"]
            
            # Obtener análisis de rendimiento
            performance_result = self.analyze_investment_performance(platform, channel_id)
            if performance_result["status"] == "success":
                report_data["performance_analysis"] = {
                    "category_performance": performance_result["category_performance"],
                    "performance_summary": performance_result["performance_summary"]
                }
            
            # Crear directorio para informes
            reports_dir = os.path.join(self.data_path, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Exportar según formato solicitado
            if output_format == "json":
                # Exportar a JSON
                json_path = os.path.join(reports_dir, f"{report_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=4)
                
                return {
                    "status": "success",
                    "report_id": report_id,
                    "format": "json",
                    "file_path": json_path,
                    "message": "Informe exportado correctamente en formato JSON"
                }
                
            elif output_format == "csv":
                # Exportar a CSV (múltiples archivos)
                csv_dir = os.path.join(reports_dir, report_id)
                os.makedirs(csv_dir, exist_ok=True)
                
                # Exportar datos principales
                main_data = {
                    "report_id": report_id,
                    "platform": platform,
                    "channel_id": channel_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "generated_at": report_data["generated_at"]
                }
                
                main_df = pd.DataFrame([main_data])
                main_csv_path = os.path.join(csv_dir, "report_info.csv")
                main_df.to_csv(main_csv_path, index=False)
                
                # Exportar datos de ingresos
                if report_data["revenue_data"]:
                    revenue_sources = report_data["revenue_data"].get("revenue_sources", {})
                    revenue_df = pd.DataFrame([{
                        "source": source,
                        "amount": amount
                    } for source, amount in revenue_sources.items()])
                    
                    revenue_csv_path = os.path.join(csv_dir, "revenue_data.csv")
                    revenue_df.to_csv(revenue_csv_path, index=False)
                
                # Exportar datos de costos
                if report_data["cost_data"]:
                    cost_categories = report_data["cost_data"].get("cost_categories", {})
                    cost_df = pd.DataFrame([{
                        "category": category,
                        "amount": amount
                    } for category, amount in cost_categories.items()])
                    
                    cost_csv_path = os.path.join(csv_dir, "cost_data.csv")
                    cost_df.to_csv(cost_csv_path, index=False)
                
                # Exportar historial de inversiones
                if report_data["investment_history"]:
                    investments = []
                    for inv_id, inv_data in report_data["investment_history"].items():
                        inv_row = {
                            "investment_id": inv_id,
                            "investment_date": inv_data.get("investment_date", ""),
                            "total_invested": inv_data.get("total_invested", 0)
                        }
                        
                        # Añadir asignaciones por categoría
                        for category, amount in inv_data.get("allocation", {}).items():
                            inv_row[f"allocation_{category}"] = amount
                        
                        # Añadir impactos medidos
                        if inv_data.get("measured_impacts"):
                            for impact_type, value in inv_data["measured_impacts"].items():
                                inv_row[f"impact_{impact_type}"] = value
                        
                        investments.append(inv_row)
                    
                    if investments:
                        inv_df = pd.DataFrame(investments)
                        inv_csv_path = os.path.join(csv_dir, "investment_history.csv")
                        inv_df.to_csv(inv_csv_path, index=False)
                
                return {
                    "status": "success",
                    "report_id": report_id,
                    "format": "csv",
                    "directory_path": csv_dir,
                    "message": "Informe exportado correctamente en formato CSV"
                }
                
            elif output_format == "excel":
                # Exportar a Excel (un solo archivo con múltiples hojas)
                excel_path = os.path.join(reports_dir, f"{report_id}.xlsx")
                
                with pd.ExcelWriter(excel_path) as writer:
                    # Hoja de información general
                    main_data = {
                        "report_id": report_id,
                        "platform": platform,
                        "channel_id": channel_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "generated_at": report_data["generated_at"]
                    }
                    
                    # Añadir datos de análisis de beneficio
                    if report_data["profit_analysis"]:
                        for key, value in report_data["profit_analysis"].items():
                            if key not in ["platform", "channel_id", "start_date", "end_date"]:
                                main_data[key] = value
                    
                    main_df = pd.DataFrame([main_data])
                    main_df.to_excel(writer, sheet_name="Report Info", index=False)
                    
                    # Hoja de ingresos
                    if report_data["revenue_data"]:
                        revenue_sources = report_data["revenue_data"].get("revenue_sources", {})
                        revenue_df = pd.DataFrame([{
                            "source": source,
                            "amount": amount
                        } for source, amount in revenue_sources.items()])
                        
                        revenue_df.to_excel(writer, sheet_name="Revenue Data", index=False)
                    
                    # Hoja de costos
                    if report_data["cost_data"]:
                        cost_categories = report_data["cost_data"].get("cost_categories", {})
                        cost_df = pd.DataFrame([{
                            "category": category,
                            "amount": amount
                        } for category, amount in cost_categories.items()])
                        
                        cost_df.to_excel(writer, sheet_name="Cost Data", index=False)
                    
                    # Hoja de inversiones
                    if report_data["investment_history"]:
                        investments = []
                        for inv_id, inv_data in report_data["investment_history"].items():
                            inv_row = {
                                "investment_id": inv_id,
                                "investment_date": inv_data.get("investment_date", ""),
                                "total_invested": inv_data.get("total_invested", 0)
                            }
                            
                            # Añadir asignaciones por categoría
                            for category, amount in inv_data.get("allocation", {}).items():
                                inv_row[f"allocation_{category}"] = amount
                            
                            # Añadir impactos medidos
                            if inv_data.get("measured_impacts"):
                                for impact_type, value in inv_data["measured_impacts"].items():
                                    inv_row[f"impact_{impact_type}"] = value
                            
                            investments.append(inv_row)
                        
                        if investments:
                            inv_df = pd.DataFrame(investments)
                            inv_df.to_excel(writer, sheet_name="Investment History", index=False)
                    
                    # Hoja de rendimiento por categoría
                    if report_data["performance_analysis"].get("category_performance"):
                        perf_data = []
                        for category, metrics in report_data["performance_analysis"]["category_performance"].items():
                            perf_row = {"category": category}
                            perf_row.update(metrics)
                            perf_data.append(perf_row)
                        
                        if perf_data:
                            perf_df = pd.DataFrame(perf_data)
                            perf_df.to_excel(writer, sheet_name="Category Performance", index=False)
                
                return {
                    "status": "success",
                    "report_id": report_id,
                    "format": "excel",
                    "file_path": excel_path,
                    "message": "Informe exportado correctamente en formato Excel"
                }
                
            else:
                return {
                    "status": "error",
                    "message": f"Formato de salida no soportado: {output_format}"
                }
            
        except Exception as e:
            logger.error(f"Error al exportar informe de análisis: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al exportar informe de análisis: {str(e)}"
            }
"""
Módulo de pronóstico de tendencias.

Este módulo analiza las tendencias históricas y actuales para predecir
su evolución futura y determinar el mejor momento para crear contenido.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importar módulos relacionados
from trends.trend_radar import TrendRadar
from trends.opportunity_scorer import OpportunityScorer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'trends.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('forecasting_engine')

class ForecastingEngine:
    """
    Motor de pronóstico para predecir la evolución de tendencias.
    """
    
    def __init__(self, history_path: str = 'data/trend_history.json'):
        """
        Inicializa el motor de pronóstico.
        
        Args:
            history_path: Ruta al archivo de historial de tendencias
        """
        self.history_path = history_path
        self.trend_radar = TrendRadar()
        self.opportunity_scorer = OpportunityScorer()
        self.load_history()
        
        # Parámetros de pronóstico
        self.forecast_days = 7  # Pronóstico a 7 días
        self.min_data_points = 3  # Mínimo de puntos para hacer pronóstico
        self.polynomial_degree = 2  # Grado del polinomio para regresión
        
        # Caché de pronósticos
        self.forecasts_cache = {}
        self.last_forecast = datetime.now() - timedelta(days=1)  # Forzar primera ejecución
    
    def load_history(self) -> None:
        """Carga el historial de tendencias desde el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Historial de tendencias cargado: {len(self.history)} registros")
            else:
                self.history = {"trends": {}, "last_update": datetime.now().isoformat()}
                self.save_history()
                logger.info("Creado nuevo archivo de historial de tendencias")
        except Exception as e:
            logger.error(f"Error al cargar el historial de tendencias: {e}")
            self.history = {"trends": {}, "last_update": datetime.now().isoformat()}
    
    def save_history(self) -> None:
        """Guarda el historial de tendencias en el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
            logger.info("Historial de tendencias guardado correctamente")
        except Exception as e:
            logger.error(f"Error al guardar el historial de tendencias: {e}")
    
    def update_history(self) -> None:
        """Actualiza el historial con las tendencias actuales."""
        logger.info("Actualizando historial de tendencias")
        
        # Obtener tendencias actuales
        opportunities = self.opportunity_scorer.evaluate_opportunities(force_refresh=True)
        
        # Fecha actual para el registro
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Actualizar historial para cada tendencia
        for opp in opportunities[:50]:  # Guardar solo las 50 mejores
            keyword = opp['keyword']
            
            if keyword not in self.history['trends']:
                self.history['trends'][keyword] = {
                    "data_points": {},
                    "first_seen": current_date,
                    "platforms": opp['platforms'],
                    "niches": opp['niches']
                }
            
            # Actualizar plataformas y nichos
            self.history['trends'][keyword]['platforms'] = list(set(
                self.history['trends'][keyword]['platforms'] + opp['platforms']
            ))
            self.history['trends'][keyword]['niches'] = list(set(
                self.history['trends'][keyword]['niches'] + opp['niches']
            ))
            
            # Añadir punto de datos para hoy
            self.history['trends'][keyword]['data_points'][current_date] = {
                "score": opp['total_score'],
                "metrics": opp['scores'],
                "platform_count": opp['platform_count']
            }
        
        # Actualizar timestamp
        self.history['last_update'] = datetime.now().isoformat()
        
        # Guardar historial actualizado
        self.save_history()
    
    def forecast_trends(self, force_update: bool = False) -> List[Dict[str, Any]]:
        """
        Genera pronósticos para las tendencias basados en datos históricos.
        
        Args:
            force_update: Si es True, fuerza una actualización del historial
            
        Returns:
            Lista de tendencias con pronósticos, ordenadas por potencial futuro
        """
        # Verificar si necesitamos actualizar
        time_since_last = datetime.now() - self.last_forecast
        if not force_update and time_since_last < timedelta(hours=12) and self.forecasts_cache:
            logger.info("Usando caché de pronósticos (última actualización hace menos de 12 horas)")
            return self.forecasts_cache
        
        # Actualizar historial si es necesario
        if force_update or time_since_last > timedelta(hours=24):
            self.update_history()
        
        logger.info("Generando pronósticos de tendencias")
        
        forecasts = []
        current_date = datetime.now().date()
        
        # Generar pronósticos para cada tendencia en el historial
        for keyword, trend_data in self.history['trends'].items():
            data_points = trend_data['data_points']
            
            # Verificar si hay suficientes puntos para hacer pronóstico
            if len(data_points) < self.min_data_points:
                continue
            
            # Convertir datos a formato para regresión
            dates = []
            scores = []
            
            for date_str, point in data_points.items():
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                days_diff = (date_obj - current_date).days
                dates.append(days_diff)
                scores.append(point['score'])
            
            # Crear DataFrame para análisis
            df = pd.DataFrame({
                'days': dates,
                'score': scores
            })
            
            # Ordenar por fecha
            df = df.sort_values('days')
            
            # Calcular tendencia (pendiente de la regresión lineal)
            X = df[['days']]
            y = df['score']
            
            # Usar regresión polinómica para capturar tendencias no lineales
            poly = PolynomialFeatures(degree=self.polynomial_degree)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Generar días para pronóstico (desde hoy hasta N días en el futuro)
            future_days = list(range(0, self.forecast_days + 1))
            future_X = poly.transform(np.array(future_days).reshape(-1, 1))
            
            # Predecir puntuaciones futuras
            future_scores = model.predict(future_X)
            
            # Calcular métricas de pronóstico
            current_score = float(future_scores[0])
            peak_score = float(max(future_scores))
            peak_day = future_days[np.argmax(future_scores)]
            
            growth_potential = (peak_score / current_score) - 1 if current_score > 0 else 0
            
            # Determinar estado de la tendencia
            if growth_potential > 0.2:
                trend_state = "crecimiento"
            elif growth_potential < -0.1:
                trend_state = "declive"
            else:
                trend_state = "estable"
            
            # Calcular confianza del pronóstico basada en cantidad y consistencia de datos
            confidence = min(1.0, len(data_points) / 10) * (1 - np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0.5)
            confidence = max(0.3, min(0.9, confidence))  # Limitar entre 0.3 y 0.9
            
            # Crear objeto de pronóstico
            forecast = {
                'keyword': keyword,
                'current_score': round(current_score, 2),
                'peak_score': round(peak_score, 2),
                'peak_day': peak_day,
                'growth_potential': round(growth_potential * 100, 1),  # Convertir a porcentaje
                'trend_state': trend_state,
                'confidence': round(confidence, 2),
                'forecast_values': [round(float(score), 2) for score in future_scores],
                'forecast_days': future_days,
                'data_points_count': len(data_points),
                'first_seen': trend_data['first_seen'],
                'platforms': trend_data['platforms'],
                'niches': trend_data['niches'],
                'timestamp': datetime.now().isoformat()
            }
            
            forecasts.append(forecast)
        
        # Ordenar pronósticos por potencial de crecimiento y puntuación actual
        forecasts.sort(key=lambda x: (x['growth_potential'], x['current_score']), reverse=True)
        
        # Actualizar caché y timestamp
        self.forecasts_cache = forecasts
        self.last_forecast = datetime.now()
        
        return forecasts
    
    def get_rising_trends(self, limit: int = 10, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Obtiene las tendencias en crecimiento con mayor potencial.
        
        Args:
            limit: Número máximo de tendencias a devolver
            min_confidence: Confianza mínima requerida
            
        Returns:
            Lista de tendencias en crecimiento
        """
        forecasts = self.forecast_trends()
        
        # Filtrar tendencias en crecimiento con confianza suficiente
        rising = [f for f in forecasts if f['trend_state'] == "crecimiento" and f['confidence'] >= min_confidence]
        
        return rising[:limit]
    
    def get_trend_forecast(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el pronóstico para una tendencia específica.
        
        Args:
            keyword: Palabra clave de la tendencia
            
        Returns:
            Pronóstico de la tendencia o None si no existe
        """
        forecasts = self.forecast_trends()
        
        for forecast in forecasts:
            if forecast['keyword'].lower() == keyword.lower():
                return forecast
        
        return None
    
    def get_best_timing(self, keyword: str) -> Dict[str, Any]:
        """
        Determina el mejor momento para crear contenido sobre una tendencia.
        
        Args:
            keyword: Palabra clave de la tendencia
            
        Returns:
            Información sobre el mejor momento para publicar
        """
        forecast = self.get_trend_forecast(keyword)
        
        if not forecast:
            return {
                "keyword": keyword,
                "recommendation": "no_data",
                "message": "No hay suficientes datos para hacer una recomendación",
                "confidence": 0.0
            }
        
        current_date = datetime.now().date()
        
        # Analizar el estado de la tendencia
        if forecast['trend_state'] == "crecimiento":
            # Si está en crecimiento, verificar cuándo alcanzará su pico
            if forecast['peak_day'] <= 1:
                # El pico es hoy o mañana
                recommendation = "immediate"
                message = "Publicar inmediatamente. La tendencia está en su punto máximo o lo alcanzará muy pronto."
            elif forecast['peak_day'] <= 3:
                # El pico es en 2-3 días
                recommendation = "very_soon"
                message = f"Preparar contenido para publicar en {forecast['peak_day']} días. La tendencia alcanzará su pico pronto."
            else:
                # El pico es en 4+ días
                recommendation = "prepare"
                message = f"Comenzar a preparar contenido. La tendencia alcanzará su pico en {forecast['peak_day']} días."
        elif forecast['trend_state'] == "estable":
            # Si es estable, depende de la puntuación
            if forecast['current_score'] > 0.7:
                recommendation = "good_anytime"
                message = "La tendencia es estable y tiene buena puntuación. Buen momento para publicar en cualquier momento de esta semana."
            else:
                recommendation = "monitor"
                message = "La tendencia es estable pero con puntuación moderada. Monitorear por unos días más."
        else:  # declive
            # Si está en declive, probablemente no sea buen momento
            if forecast['current_score'] > 0.8:
                recommendation = "last_chance"
                message = "La tendencia está en declive pero aún tiene buena puntuación. Última oportunidad para aprovecharla."
            else:
                recommendation = "avoid"
                message = "La tendencia está en declive. Mejor buscar otras oportunidades."
        
        # Crear objeto de recomendación
        timing = {
            "keyword": keyword,
            "recommendation": recommendation,
            "message": message,
            "best_day": forecast['peak_day'],
            "current_score": forecast['current_score'],
            "peak_score": forecast['peak_score'],
            "growth_potential": forecast['growth_potential'],
            "trend_state": forecast['trend_state'],
            "confidence": forecast['confidence'],
            "best_date": (current_date + timedelta(days=forecast['peak_day'])).strftime("%Y-%m-%d")
        }
        
        return timing
    
    def get_content_calendar(self, days: int = 7, limit_per_day: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Genera un calendario de contenido basado en pronósticos de tendencias.
        
        Args:
            days: Número de días para el calendario
            limit_per_day: Máximo de tendencias por día
            
        Returns:
            Calendario con las mejores tendencias para cada día
        """
        forecasts = self.forecast_trends()
        current_date = datetime.now().date()
        
        # Filtrar pronósticos con confianza suficiente
        valid_forecasts = [f for f in forecasts if f['confidence'] >= 0.5]
        
        # Inicializar calendario
        calendar = {}
        for i in range(days):
            date_str = (current_date + timedelta(days=i)).strftime("%Y-%m-%d")
            calendar[date_str] = []
        
        # Asignar tendencias a días basados en su pico proyectado
        for forecast in valid_forecasts:
            peak_day = min(forecast['peak_day'], days - 1)  # Limitar al número de días del calendario
            peak_date = (current_date + timedelta(days=peak_day)).strftime("%Y-%m-%d")
            
            # Solo añadir si no hemos alcanzado el límite para ese día
            if len(calendar[peak_date]) < limit_per_day:
                calendar[peak_date].append({
                    "keyword": forecast['keyword'],
                    "score": forecast['peak_score'],
                    "growth": forecast['growth_potential'],
                    "confidence": forecast['confidence'],
                    "niches": forecast['niches'],
                    "platforms": forecast['platforms']
                })
        
        # Ordenar tendencias de cada día por puntuación
        for date, trends in calendar.items():
            calendar[date] = sorted(trends, key=lambda x: x['score'], reverse=True)
        
        return calendar
    
    def get_niche_forecast(self, niche: str, days: int = 7) -> Dict[str, Any]:
        """
        Genera un pronóstico para un nicho específico.
        
        Args:
            niche: Nicho para analizar
            days: Número de días para el pronóstico
            
        Returns:
            Pronóstico del nicho con tendencias y métricas
        """
        forecasts = self.forecast_trends()
        
        # Filtrar tendencias del nicho
        niche_trends = [f for f in forecasts if niche in f['niches']]
        
        if not niche_trends:
            return {
                "niche": niche,
                "trend_count": 0,
                "average_growth": 0,
                "niche_momentum": 0,
                "top_trends": [],
                "recommendation": "No hay suficientes datos para este nicho"
            }
        
        # Calcular métricas del nicho
        avg_growth = sum(t['growth_potential'] for t in niche_trends) / len(niche_trends)
        avg_score = sum(t['current_score'] for t in niche_trends) / len(niche_trends)
        
        # Calcular "momentum" del nicho (combinación de crecimiento y puntuación)
        niche_momentum = avg_growth * avg_score
        
        # Ordenar tendencias por potencial de crecimiento
        top_trends = sorted(niche_trends, key=lambda x: x['growth_potential'], reverse=True)[:5]
        
        # Determinar recomendación para el nicho
        if niche_momentum > 30:
            recommendation = f"El nicho {niche} tiene un excelente momentum. Priorizar la creación de contenido en este nicho."
        elif niche_momentum > 15:
            recommendation = f"El nicho {niche} tiene buen momentum. Buena oportunidad para crear contenido."
        elif niche_momentum > 5:
            recommendation = f"El nicho {niche} tiene momentum moderado. Considerar para diversificar contenido."
        else:
            recommendation = f"El nicho {niche} tiene bajo momentum actualmente. Monitorear para cambios."
        
        # Crear objeto de pronóstico de nicho
        niche_forecast = {
            "niche": niche,
            "trend_count": len(niche_trends),
            "average_growth": round(avg_growth, 1),
            "average_score": round(avg_score, 2),
            "niche_momentum": round(niche_momentum, 1),
            "top_trends": [{"keyword": t['keyword'], "growth": t['growth_potential']} for t in top_trends],
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
        return niche_forecast


if __name__ == "__main__":
    # Ejemplo de uso
    engine = ForecastingEngine()
    
    # Actualizar historial (simulado para desarrollo)
    print("Actualizando historial de tendencias...")
    engine.update_history()
    
    # Obtener tendencias en crecimiento
    rising_trends = engine.get_rising_trends(limit=5)
    print(f"\nTop 5 tendencias en crecimiento:")
    for i, trend in enumerate(rising_trends, 1):
        print(f"{i}. {trend['keyword']} (Crecimiento: {trend['growth_potential']}%, Confianza: {trend['confidence']})")
        print(f"   Puntuación actual: {trend['current_score']}, Pico esperado: {trend['peak_score']} en {trend['peak_day']} días")
        print(f"   Nichos: {', '.join(trend['niches']) if trend['niches'] else 'Ninguno'}")
        print()
    
    # Obtener recomendación de timing para una tendencia específica
    if rising_trends:
        keyword = rising_trends[0]['keyword']
        timing = engine.get_best_timing(keyword)
        print(f"\nMejor momento para publicar sobre '{keyword}':")
        print(f"Recomendación: {timing['message']}")
        print(f"Mejor fecha: {timing['best_date']}")
        print(f"Confianza: {timing['confidence']}")
    
    # Generar calendario de contenido
    calendar = engine.get_content_calendar(days=5, limit_per_day=2)
    print("\nCalendario de contenido para los próximos 5 días:")
    for date, trends in calendar.items():
        print(f"\n{date}:")
        if trends:
            for trend in trends:
                print(f"- {trend['keyword']} (Score: {trend['score']}, Crecimiento: {trend['growth']}%)")
        else:
            print("- No hay tendencias recomendadas para este día")
    
    # Obtener pronóstico para un nicho específico
    tech_forecast = engine.get_niche_forecast("tecnología")
    print(f"\nPronóstico para el nicho de tecnología:")
    print(f"Momentum: {tech_forecast['niche_momentum']}")
    print(f"Tendencias: {tech_forecast['trend_count']}")
    print(f"Crecimiento promedio: {tech_forecast['average_growth']}%")
    print(f"Recomendación: {tech_forecast['recommendation']}")
    print("\nTendencias principales:")
    for trend in tech_forecast['top_trends']:
        print(f"- {trend['keyword']} (Crecimiento: {trend['growth']}%)")
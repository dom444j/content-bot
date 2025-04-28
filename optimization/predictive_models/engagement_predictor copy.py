"""
Engagement Predictor

Modelo predictivo para estimar el engagement futuro de contenido
basado en características históricas y actuales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

class EngagementPredictor(BaseModel):
    """
    Modelo para predecir métricas de engagement como:
    - Tasa de retención
    - CTR (Click-Through Rate)
    - Tasa de comentarios
    - Tasa de likes
    - Tiempo de visualización
    """
    
    def __init__(self, target_metric: str = 'retention_rate', 
                 algorithm: str = 'random_forest',
                 model_params: Dict[str, Any] = None):
        """
        Inicializa el predictor de engagement
        
        Args:
            target_metric: Métrica objetivo ('retention_rate', 'ctr', 'comment_rate', etc.)
            algorithm: Algoritmo a utilizar ('random_forest', 'gradient_boosting', 'elastic_net')
            model_params: Parámetros específicos del algoritmo
        """
        # Definir características relevantes para engagement
        features = [
            # Características del contenido
            'content_duration',
            'cta_position',
            'cta_type',
            'has_custom_thumbnail',
            'title_length',
            'description_length',
            'hashtag_count',
            'emoji_count',
            'question_in_title',
            'content_category',
            'content_topic',
            'content_quality_score',
            
            # Características temporales
            'day_of_week',
            'hour_of_day',
            'is_weekend',
            'days_since_last_post',
            'post_frequency_last_week',
            
            # Características de audiencia
            'audience_size',
            'audience_growth_rate',
            'audience_engagement_rate',
            'audience_retention_history',
            'audience_demographic_match',
            
            # Características de plataforma
            'platform_name',
            'platform_algorithm_change',
            'platform_trending_score',
            
            # Características históricas
            'avg_retention_last_5',
            'avg_ctr_last_5',
            'avg_comment_rate_last_5',
            'avg_like_rate_last_5',
            'content_performance_trend'
        ]
        
        # Configurar parámetros por defecto según el algoritmo
        default_params = self._get_default_params(algorithm)
        if model_params:
            default_params.update(model_params)
        
        # Inicializar modelo base
        super().__init__(
            model_name=f"engagement_{target_metric}_{algorithm}",
            model_type='regression',
            features=features,
            target=target_metric,
            model_params=default_params
        )
        
        self.algorithm = algorithm
        self.target_metric = target_metric
    
    def _get_default_params(self, algorithm: str) -> Dict[str, Any]:
        """
        Obtiene parámetros por defecto para el algoritmo especificado
        
        Args:
            algorithm: Nombre del algoritmo
            
        Returns:
            Diccionario con parámetros por defecto
        """
        if algorithm == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        elif algorithm == 'gradient_boosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        elif algorithm == 'elastic_net':
            return {
                'alpha': 0.1,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'random_state': 42
            }
        else:
            self.logger.warning(f"Algoritmo desconocido: {algorithm}. Usando Random Forest.")
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
    
    def build_model(self) -> Pipeline:
        """
        Construye el modelo con el algoritmo y parámetros especificados
        
        Returns:
            Pipeline con el modelo
        """
        # Crear el estimador según el algoritmo
        if self.algorithm == 'random_forest':
            estimator = RandomForestRegressor(**self.model_params)
        elif self.algorithm == 'gradient_boosting':
            estimator = GradientBoostingRegressor(**self.model_params)
        elif self.algorithm == 'elastic_net':
            estimator = ElasticNet(**self.model_params)
        else:
            self.logger.warning(f"Algoritmo desconocido: {self.algorithm}. Usando Random Forest.")
            estimator = RandomForestRegressor(**self.model_params)
        
        # Crear pipeline con escalado y modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', estimator)
        ])
        
        return pipeline
    
    def predict_engagement(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice métricas de engagement para nuevo contenido
        
        Args:
            content_data: DataFrame con características del contenido
            
        Returns:
            Diccionario con predicciones y metadatos
        """
        # Realizar predicción
        predictions = self.predict(content_data)
        
        # Preparar resultados
        results = {
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'target_metric': self.target_metric,
            'content_ids': content_data.index.tolist() if hasattr(content_data, 'index') else list(range(len(predictions))),
            'model_info': {
                'model_name': self.model_name,
                'algorithm': self.algorithm,
                'last_trained': self.last_trained,
                'metrics': self.metrics
            }
        }
        
        # Añadir top features si están disponibles
        if self.feature_importance:
            top_features = dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            results['top_features'] = top_features
        
        return results
    
    def analyze_cta_impact(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza el impacto de diferentes posiciones y tipos de CTA en el engagement
        
        Args:
            data: DataFrame con datos históricos
            
        Returns:
            Análisis del impacto de CTAs
        """
        if 'cta_position' not in data.columns or 'cta_type' not in data.columns:
            self.logger.error("Se requieren las columnas 'cta_position' y 'cta_type' para el análisis")
            return {'error': "Datos insuficientes para análisis de CTA"}
        
        # Agrupar por posición de CTA
        cta_position_analysis = data.groupby('cta_position')[self.target_metric].agg(['mean', 'std', 'count']).reset_index()
        cta_position_analysis = cta_position_analysis.sort_values('mean', ascending=False)
        
        # Agrupar por tipo de CTA
        cta_type_analysis = data.groupby('cta_type')[self.target_metric].agg(['mean', 'std', 'count']).reset_index()
        cta_type_analysis = cta_type_analysis.sort_values('mean', ascending=False)
        
        # Análisis combinado
        if len(data) >= 100:  # Solo si hay suficientes datos
            cta_combined = data.groupby(['cta_position', 'cta_type'])[self.target_metric].agg(['mean', 'count']).reset_index()
            cta_combined = cta_combined[cta_combined['count'] >= 5]  # Filtrar combinaciones con pocos datos
            cta_combined = cta_combined.sort_values('mean', ascending=False)
            top_combinations = cta_combined.head(5).to_dict('records')
        else:
            top_combinations = []
        
        return {
            'metric': self.target_metric,
            'best_cta_positions': cta_position_analysis.head(3).to_dict('records'),
            'best_cta_types': cta_type_analysis.head(3).to_dict('records'),
            'top_combinations': top_combinations,
            'sample_size': len(data),
            'recommendation': self._generate_cta_recommendation(cta_position_analysis, cta_type_analysis)
        }
    
    def _generate_cta_recommendation(self, position_analysis, type_analysis) -> str:
        """
        Genera una recomendación basada en el análisis de CTAs
        
        Args:
            position_analysis: Análisis por posición
            type_analysis: Análisis por tipo
            
        Returns:
            Recomendación textual
        """
        try:
            best_position = position_analysis.iloc[0]['cta_position']
            best_type = type_analysis.iloc[0]['cta_type']
            
            return (f"Para maximizar {self.target_metric}, se recomienda utilizar CTAs de tipo "
                   f"'{best_type}' en la posición {best_position} segundos del contenido.")
        except:
            return "Datos insuficientes para generar una recomendación precisa."
    
    def get_optimal_posting_time(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Determina el momento óptimo para publicar contenido
        
        Args:
            data: DataFrame con datos históricos
            
        Returns:
            Análisis de tiempos óptimos
        """
        if 'day_of_week' not in data.columns or 'hour_of_day' not in data.columns:
            self.logger.error("Se requieren las columnas 'day_of_week' y 'hour_of_day' para el análisis")
            return {'error': "Datos insuficientes para análisis de tiempo"}
        
        # Agrupar por día de la semana
        day_analysis = data.groupby('day_of_week')[self.target_metric].agg(['mean', 'std', 'count']).reset_index()
        day_analysis = day_analysis.sort_values('mean', ascending=False)
        
        # Agrupar por hora del día
        hour_analysis = data.groupby('hour_of_day')[self.target_metric].agg(['mean', 'std', 'count']).reset_index()
        hour_analysis = hour_analysis.sort_values('mean', ascending=False)
        
        # Análisis combinado
        if len(data) >= 100:  # Solo si hay suficientes datos
            time_combined = data.groupby(['day_of_week', 'hour_of_day'])[self.target_metric].agg(['mean', 'count']).reset_index()
            time_combined = time_combined[time_combined['count'] >= 3]  # Filtrar combinaciones con pocos datos
            time_combined = time_combined.sort_values('mean', ascending=False)
            top_times = time_combined.head(5).to_dict('records')
        else:
            top_times = []
        
        # Mapear días numéricos a nombres
        day_names = {
            0: 'Lunes',
            1: 'Martes',
            2: 'Miércoles',
            3: 'Jueves',
            4: 'Viernes',
            5: 'Sábado',
            6: 'Domingo'
        }
        
        # Generar recomendación
        try:
            best_day = day_analysis.iloc[0]['day_of_week']
            best_hour = hour_analysis.iloc[0]['hour_of_day']
            
            recommendation = (f"Para maximizar {self.target_metric}, se recomienda publicar contenido "
                             f"los {day_names.get(best_day, best_day)} a las {best_hour}:00 horas.")
            
            if top_times:
                best_combined_day = top_times[0]['day_of_week']
                best_combined_hour = top_times[0]['hour_of_day']
                recommendation += (f" La combinación óptima es {day_names.get(best_combined_day, best_combined_day)} "
                                  f"a las {best_combined_hour}:00 horas.")
        except:
            recommendation = "Datos insuficientes para generar una recomendación precisa."
        
        return {
            'metric': self.target_metric,
            'best_days': day_analysis.head(3).to_dict('records'),
            'best_hours': hour_analysis.head(3).to_dict('records'),
            'top_combinations': top_times,
            'sample_size': len(data),
            'recommendation': recommendation
                }
    
    def analyze_content_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza qué características del contenido tienen mayor impacto en el engagement
        
        Args:
            data: DataFrame con datos históricos
            
        Returns:
            Análisis de características del contenido
        """
        content_features = [
            'content_duration', 'cta_position', 'cta_type', 'has_custom_thumbnail',
            'title_length', 'description_length', 'hashtag_count', 'emoji_count',
            'question_in_title', 'content_category', 'content_topic', 'content_quality_score'
        ]
        
        # Verificar que las características necesarias estén presentes
        missing_features = [f for f in content_features if f not in data.columns]
        if missing_features:
            self.logger.warning(f"Faltan características en los datos: {missing_features}")
            content_features = [f for f in content_features if f in data.columns]
            
        if not content_features:
            return {'error': "No hay características de contenido disponibles para análisis"}
        
        # Calcular correlaciones con la métrica objetivo
        if self.target_metric in data.columns:
            correlations = {}
            for feature in content_features:
                if pd.api.types.is_numeric_dtype(data[feature]):
                    correlations[feature] = data[feature].corr(data[self.target_metric])
                else:
                    # Para características categóricas, usar eta_squared (implementación simplificada)
                    try:
                        categories = data[feature].unique()
                        group_means = data.groupby(feature)[self.target_metric].mean()
                        overall_mean = data[self.target_metric].mean()
                        
                        ss_between = sum([(group_means[cat] - overall_mean)**2 * (data[feature] == cat).sum() 
                                        for cat in categories])
                        ss_total = sum((data[self.target_metric] - overall_mean)**2)
                        
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        correlations[feature] = eta_squared
                    except:
                        correlations[feature] = 0
            
            # Ordenar por importancia
            sorted_correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
            
            # Análisis de características categóricas
            categorical_insights = {}
            for feature in content_features:
                if not pd.api.types.is_numeric_dtype(data[feature]) and len(data[feature].unique()) < 20:
                    try:
                        feature_analysis = data.groupby(feature)[self.target_metric].agg(['mean', 'count']).reset_index()
                        feature_analysis = feature_analysis[feature_analysis['count'] >= 5]
                        feature_analysis = feature_analysis.sort_values('mean', ascending=False)
                        
                        if not feature_analysis.empty:
                            categorical_insights[feature] = {
                                'best_values': feature_analysis.head(3).to_dict('records'),
                                'worst_values': feature_analysis.tail(3).to_dict('records')
                            }
                    except:
                        pass
            
            return {
                'metric': self.target_metric,
                'feature_importance': sorted_correlations,
                'top_features': list(sorted_correlations.keys())[:5],
                'categorical_insights': categorical_insights,
                'recommendation': self._generate_content_recommendation(sorted_correlations, categorical_insights)
            }
        else:
            return {'error': f"La métrica objetivo {self.target_metric} no está disponible en los datos"}
    
    def _generate_content_recommendation(self, correlations: Dict[str, float], 
                                        categorical_insights: Dict[str, Any]) -> str:
        """
        Genera recomendaciones basadas en el análisis de características
        
        Args:
            correlations: Correlaciones entre características y métrica objetivo
            categorical_insights: Análisis de características categóricas
            
        Returns:
            Recomendación textual
        """
        try:
            # Obtener las 3 características más importantes
            top_features = list(correlations.keys())[:3]
            
            recommendation = f"Para maximizar {self.target_metric}, se recomienda enfocarse en: "
            
            for feature in top_features:
                if feature in categorical_insights and categorical_insights[feature]['best_values']:
                    best_value = categorical_insights[feature]['best_values'][0][feature]
                    recommendation += f"\n- Utilizar '{best_value}' para {feature}"
                elif correlations[feature] > 0:
                    recommendation += f"\n- Aumentar {feature}"
                else:
                    recommendation += f"\n- Reducir {feature}"
            
            # Añadir recomendación específica para duración si está disponible
            if 'content_duration' in correlations:
                if 'content_duration' in categorical_insights:
                    best_duration = categorical_insights['content_duration']['best_values'][0]['content_duration']
                    recommendation += f"\n\nLa duración óptima del contenido es de aproximadamente {best_duration} segundos."
                elif correlations['content_duration'] > 0:
                    recommendation += "\n\nLos contenidos más largos tienden a generar mejor engagement."
                else:
                    recommendation += "\n\nLos contenidos más cortos tienden a generar mejor engagement."
            
            # Añadir recomendación para CTAs si están disponibles
            if 'cta_position' in correlations or 'cta_type' in correlations:
                recommendation += "\n\nPara los CTAs:"
                
                if 'cta_position' in categorical_insights:
                    best_position = categorical_insights['cta_position']['best_values'][0]['cta_position']
                    recommendation += f"\n- Colocar CTAs alrededor del segundo {best_position}"
                
                if 'cta_type' in categorical_insights:
                    best_type = categorical_insights['cta_type']['best_values'][0]['cta_type']
                    recommendation += f"\n- Utilizar CTAs de tipo '{best_type}'"
            
            return recommendation
        except:
            return "No hay suficientes datos para generar recomendaciones específicas."
    
    def predict_viral_potential(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice el potencial viral de nuevo contenido
        
        Args:
            content_data: DataFrame con características del contenido
            
        Returns:
            Predicciones de potencial viral
        """
        if self.model is None:
            return {'error': "El modelo no está entrenado"}
        
        # Realizar predicción
        predictions = self.predict(content_data)
        
        # Calcular percentiles si hay suficientes datos históricos
        if hasattr(self, 'historical_percentiles') and self.historical_percentiles is not None:
            viral_scores = []
            for pred in predictions:
                # Convertir predicción a percentil
                percentile = 0
                for p, threshold in self.historical_percentiles.items():
                    if pred >= threshold:
                        percentile = p
                        break
                viral_scores.append({
                    'prediction': pred,
                    'percentile': percentile,
                    'viral_potential': 'Alto' if percentile >= 90 else 
                                      'Medio' if percentile >= 70 else 'Bajo'
                })
        else:
            # Sin datos históricos, usar heurística simple
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions) if len(predictions) > 1 else mean_pred * 0.1
            
            viral_scores = []
            for pred in predictions:
                z_score = (pred - mean_pred) / std_pred if std_pred > 0 else 0
                percentile = min(100, max(0, int(norm.cdf(z_score) * 100)))
                
                viral_scores.append({
                    'prediction': pred,
                    'percentile': percentile,
                    'viral_potential': 'Alto' if percentile >= 90 else 
                                      'Medio' if percentile >= 70 else 'Bajo'
                })
        
        return {
            'metric': self.target_metric,
            'predictions': viral_scores,
            'content_ids': content_data.index.tolist() if hasattr(content_data, 'index') else list(range(len(predictions)))
        }
    
    def calculate_historical_percentiles(self, historical_data: pd.DataFrame) -> None:
        """
        Calcula percentiles históricos para la métrica objetivo
        
        Args:
            historical_data: DataFrame con datos históricos
        """
        if self.target_metric not in historical_data.columns:
            self.logger.error(f"La métrica {self.target_metric} no está en los datos históricos")
            return
        
        # Calcular percentiles
        percentiles = [50, 70, 80, 90, 95, 99]
        self.historical_percentiles = {}
        
        for p in percentiles:
            self.historical_percentiles[p] = historical_data[self.target_metric].quantile(p/100)
        
        self.logger.info(f"Percentiles históricos calculados: {self.historical_percentiles}")
    
    def get_engagement_trends(self, time_series_data: pd.DataFrame, 
                             time_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Analiza tendencias de engagement a lo largo del tiempo
        
        Args:
            time_series_data: DataFrame con datos de series temporales
            time_column: Nombre de la columna de tiempo
            
        Returns:
            Análisis de tendencias
        """
        if time_column not in time_series_data.columns or self.target_metric not in time_series_data.columns:
            return {'error': f"Se requieren las columnas {time_column} y {self.target_metric}"}
        
        # Asegurar que la columna de tiempo es datetime
        try:
            time_series_data[time_column] = pd.to_datetime(time_series_data[time_column])
        except:
            return {'error': f"No se puede convertir {time_column} a formato datetime"}
        
        # Ordenar por tiempo
        time_series_data = time_series_data.sort_values(time_column)
        
        # Calcular tendencia general (regresión lineal simple)
        X = np.array(range(len(time_series_data))).reshape(-1, 1)
        y = time_series_data[self.target_metric].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        trend_direction = "ascendente" if trend_slope > 0 else "descendente" if trend_slope < 0 else "estable"
        
        # Calcular tendencias por período
        # Diaria
        if len(time_series_data) >= 7:  # Al menos una semana de datos
            daily_data = time_series_data.copy()
            daily_data['day'] = daily_data[time_column].dt.date
            daily_avg = daily_data.groupby('day')[self.target_metric].mean().reset_index()
            
            # Calcular cambio porcentual diario
            daily_avg['pct_change'] = daily_avg[self.target_metric].pct_change() * 100
            
            # Últimos 7 días
            last_7_days = daily_avg.tail(7)
            avg_daily_change = last_7_days['pct_change'].mean()
            
            # Detectar días de la semana con mejor rendimiento
            daily_data['weekday'] = daily_data[time_column].dt.weekday
            weekday_performance = daily_data.groupby('weekday')[self.target_metric].mean().reset_index()
            best_weekday = weekday_performance.loc[weekday_performance[self.target_metric].idxmax(), 'weekday']
            
            day_names = {
                0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves',
                4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
            }
            
            best_day_name = day_names.get(best_weekday, str(best_weekday))
        else:
            avg_daily_change = None
            best_day_name = None
        
        # Mensual (si hay suficientes datos)
        if (time_series_data[time_column].max() - time_series_data[time_column].min()).days >= 60:
            monthly_data = time_series_data.copy()
            monthly_data['month'] = monthly_data[time_column].dt.to_period('M')
            monthly_avg = monthly_data.groupby('month')[self.target_metric].mean().reset_index()
            
            # Calcular cambio porcentual mensual
            monthly_avg['pct_change'] = monthly_avg[self.target_metric].pct_change() * 100
            avg_monthly_change = monthly_avg['pct_change'].tail(3).mean()
        else:
            avg_monthly_change = None
        
        # Detectar estacionalidad (si hay suficientes datos)
        has_seasonality = False
        seasonality_period = None
        
        if len(time_series_data) >= 90:  # Al menos 3 meses de datos
            try:
                # Convertir a serie temporal
                ts = time_series_data.set_index(time_column)[self.target_metric]
                
                # Resamplear a frecuencia diaria
                ts_daily = ts.resample('D').mean().interpolate()
                
                # Descomponer la serie
                decomposition = seasonal_decompose(ts_daily, model='additive', period=7)
                
                # Verificar si hay estacionalidad significativa
                seasonality_strength = np.std(decomposition.seasonal) / np.std(decomposition.trend)
                has_seasonality = seasonality_strength > 0.3
                
                if has_seasonality:
                    # Intentar detectar el período
                    from statsmodels.tsa.stattools import acf
                    
                    acf_values = acf(ts_daily.dropna(), nlags=90)
                    # Buscar picos en la ACF después del lag 1
                    peaks, _ = find_peaks(acf_values[1:], height=0.2)
                    
                    if len(peaks) > 0:
                        seasonality_period = peaks[0] + 1  # +1 porque empezamos desde lag 1
            except:
                pass
        
        return {
            'metric': self.target_metric,
            'trend_direction': trend_direction,
            'trend_slope': float(trend_slope),
            'daily_change_pct': float(avg_daily_change) if avg_daily_change is not None else None,
            'monthly_change_pct': float(avg_monthly_change) if avg_monthly_change is not None else None,
            'best_day': best_day_name,
            'has_seasonality': has_seasonality,
            'seasonality_period': seasonality_period,
            'recommendation': self._generate_trend_recommendation(
                trend_direction, avg_daily_change, best_day_name, has_seasonality
            )
        }
    
    def _generate_trend_recommendation(self, trend_direction: str, daily_change: float,
                                      best_day: str, has_seasonality: bool) -> str:
        """
        Genera recomendaciones basadas en el análisis de tendencias
        
        Args:
            trend_direction: Dirección de la tendencia general
            daily_change: Cambio porcentual diario promedio
            best_day: Mejor día de la semana
            has_seasonality: Si hay estacionalidad
            
        Returns:
            Recomendación textual
        """
        recommendation = f"La tendencia de {self.target_metric} es {trend_direction}."
        
        if daily_change is not None:
            if daily_change > 0:
                recommendation += f" El engagement está creciendo un {abs(daily_change):.2f}% diario en promedio."
            elif daily_change < 0:
                recommendation += f" El engagement está disminuyendo un {abs(daily_change):.2f}% diario en promedio."
        
        if best_day:
            recommendation += f"\n\nEl mejor día para publicar es {best_day}."
        
        if has_seasonality:
            recommendation += "\n\nSe detecta un patrón estacional en el engagement."
        
        # Recomendaciones específicas según la tendencia
        if trend_direction == "ascendente":
            recommendation += "\n\nRecomendaciones:"
            recommendation += "\n- Mantener la estrategia actual de contenido"
            recommendation += "\n- Considerar aumentar la frecuencia de publicación"
            recommendation += "\n- Aprovechar el momentum para experimentar con nuevos formatos"
        elif trend_direction == "descendente":
            recommendation += "\n\nRecomendaciones:"
            recommendation += "\n- Revisar y ajustar la estrategia de contenido"
            recommendation += "\n- Analizar qué ha cambiado desde que comenzó la tendencia negativa"
            recommendation += "\n- Considerar realizar pruebas A/B para identificar mejoras"
        else:
            recommendation += "\n\nRecomendaciones:"
            recommendation += "\n- Mantener la consistencia en la publicación"
            recommendation += "\n- Buscar oportunidades para innovar y generar picos de engagement"
        
        return recommendation
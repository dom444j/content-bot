"""
Revenue Predictor

Modelo predictivo para estimar ingresos futuros basado en
características del contenido, audiencia y monetización.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from .base_model import BaseModel

class RevenuePredictor(BaseModel):
    """
    Modelo para predecir métricas de ingresos como:
    - RPM (Revenue Per Mille)
    - Ingresos totales
    - Conversión de afiliados
    - Valor de cliente (LTV)
    - ROI de campañas
    """
    
    def __init__(self, target_metric: str = 'rpm', 
                 algorithm: str = 'gradient_boosting',
                 model_params: Dict[str, Any] = None):
        """
        Inicializa el predictor de ingresos
        
        Args:
            target_metric: Métrica objetivo ('rpm', 'total_revenue', 'affiliate_conversion', etc.)
            algorithm: Algoritmo a utilizar ('random_forest', 'gradient_boosting', 'elastic_net')
            model_params: Parámetros específicos del algoritmo
        """
        # Definir características relevantes para ingresos
        features = [
            # Características del contenido
            'content_duration',
            'content_category',
            'content_topic',
            'content_quality_score',
            'has_sponsorship',
            'has_affiliate_links',
            'affiliate_link_count',
            'cta_position',
            'cta_type',
            'product_price',
            'product_category',
            
            # Características de audiencia
            'audience_size',
            'audience_growth_rate',
            'audience_engagement_rate',
            'audience_retention_rate',
            'audience_demographic',
            'audience_geo_distribution',
            'audience_device_type',
            
            # Características de plataforma
            'platform_name',
            'platform_rpm_average',
            'platform_ad_fill_rate',
            'platform_monetization_options',
            
            # Características temporales
            'day_of_week',
            'month',
            'is_holiday_season',
            'days_since_last_post',
            
            # Métricas de engagement
            'view_count',
            'like_rate',
            'comment_rate',
            'share_rate',
            'click_through_rate',
            'watch_time',
            
            # Características históricas
            'avg_rpm_last_10',
            'avg_revenue_last_10',
            'avg_conversion_last_10',
            'revenue_trend'
        ]
        
        # Configurar parámetros por defecto según el algoritmo
        default_params = self._get_default_params(algorithm)
        if model_params:
            default_params.update(model_params)
        
        # Inicializar modelo base
        super().__init__(
            model_name=f"revenue_{target_metric}_{algorithm}",
            model_type='regression',
            features=features,
            target=target_metric,
            model_params=default_params
        )
        
        self.algorithm = algorithm
        self.target_metric = target_metric
        self.revenue_thresholds = None
    
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
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        elif algorithm == 'gradient_boosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 8,
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
            self.logger.warning(f"Algoritmo desconocido: {algorithm}. Usando Gradient Boosting.")
            return {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 8,
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
            self.logger.warning(f"Algoritmo desconocido: {self.algorithm}. Usando Gradient Boosting.")
            estimator = GradientBoostingRegressor(**self.model_params)
        
        # Crear pipeline con escalado y modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', estimator)
        ])
        
        return pipeline
    
    def predict_revenue(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice ingresos para nuevo contenido
        
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
        
        # Añadir categorización si hay umbrales definidos
        if self.revenue_thresholds and isinstance(predictions, np.ndarray):
            categories = []
            for pred in predictions:
                if pred >= self.revenue_thresholds.get('high', float('inf')):
                    categories.append('alto')
                elif pred >= self.revenue_thresholds.get('medium', 0):
                    categories.append('medio')
                else:
                    categories.append('bajo')
            results['categories'] = categories
        
        return results
    
    def set_revenue_thresholds(self, historical_data: pd.DataFrame = None, 
                              manual_thresholds: Dict[str, float] = None) -> None:
        """
        Establece umbrales para categorizar predicciones de ingresos
        
        Args:
            historical_data: DataFrame con datos históricos para calcular umbrales
            manual_thresholds: Diccionario con umbrales manuales
        """
        if manual_thresholds:
            self.revenue_thresholds = manual_thresholds
            self.logger.info(f"Umbrales de ingresos establecidos manualmente: {manual_thresholds}")
        elif historical_data is not None and self.target_metric in historical_data.columns:
            # Calcular percentiles
            low_threshold = historical_data[self.target_metric].quantile(0.33)
            high_threshold = historical_data[self.target_metric].quantile(0.66)
            
            self.revenue_thresholds = {
                'low': 0,
                'medium': low_threshold,
                'high': high_threshold
            }
            
            self.logger.info(f"Umbrales de ingresos calculados: {self.revenue_thresholds}")
        else:
            self.logger.warning("No se pudieron establecer umbrales de ingresos")
    
    def analyze_monetization_factors(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza factores que influyen en la monetización
        
        Args:
            data: DataFrame con datos históricos
            
        Returns:
            Análisis de factores de monetización
        """
        if self.target_metric not in data.columns:
            return {'error': f"La métrica {self.target_metric} no está en los datos"}
        
        # Factores de monetización a analizar
        monetization_factors = [
            'content_category', 'content_topic', 'has_sponsorship', 
            'has_affiliate_links', 'platform_name', 'audience_demographic',
            'audience_geo_distribution', 'cta_type', 'product_category'
        ]
        
        # Filtrar factores disponibles
        available_factors = [f for f in monetization_factors if f in data.columns]
        
        if not available_factors:
            return {'error': "No hay factores de monetización disponibles para análisis"}
        
        # Analizar cada factor
        factor_analysis = {}
        
        for factor in available_factors:
            # Agrupar por factor
            factor_data = data.groupby(factor)[self.target_metric].agg(['mean', 'std', 'count']).reset_index()
            factor_data = factor_data[factor_data['count'] >= 5]  # Filtrar grupos con pocos datos
            factor_data = factor_data.sort_values('mean', ascending=False)
            
            if not factor_data.empty:
                factor_analysis[factor] = {
                    'top_values': factor_data.head(3).to_dict('records'),
                    'bottom_values': factor_data.tail(3).to_dict('records')
                }
        
        # Análisis de combinaciones (si hay suficientes datos)
        combination_analysis = {}
        
        if len(data) >= 100 and len(available_factors) >= 2:
            # Analizar combinaciones de dos factores
            for i, factor1 in enumerate(available_factors):
                for factor2 in available_factors[i+1:]:
                    try:
                        # Agrupar por combinación de factores
                        combo_data = data.groupby([factor1, factor2])[self.target_metric].agg(['mean', 'count']).reset_index()
                        combo_data = combo_data[combo_data['count'] >= 5]  # Filtrar combinaciones con pocos datos
                        combo_data = combo_data.sort_values('mean', ascending=False)
                        
                        if not combo_data.empty:
                            combination_analysis[f"{factor1}_{factor2}"] = combo_data.head(3).to_dict('records')
                    except:
                        pass
        
        # Generar recomendaciones
        recommendations = self._generate_monetization_recommendations(factor_analysis, combination_analysis)
        
        return {
            'metric': self.target_metric,
            'factor_analysis': factor_analysis,
            'combination_analysis': combination_analysis,
            'recommendations': recommendations
        }
    
    def _generate_monetization_recommendations(self, factor_analysis: Dict[str, Any],
                                              combination_analysis: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones basadas en el análisis de factores de monetización
        
        Args:
            factor_analysis: Análisis de factores individuales
            combination_analysis: Análisis de combinaciones de factores
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendaciones por factor individual
        for factor, analysis in factor_analysis.items():
            if analysis['top_values']:
                top_value = analysis['top_values'][0][factor]
                top_mean = analysis['top_values'][0]['mean']
                
                if factor == 'content_category':
                    recommendations.append(
                                                f"La categoría de contenido '{top_value}' genera el mayor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
                elif factor == 'content_topic':
                    recommendations.append(
                        f"El tema '{top_value}' es el más rentable con un {self.target_metric} promedio de {top_mean:.2f}"
                    )
                elif factor == 'platform_name':
                    recommendations.append(
                        f"La plataforma '{top_value}' ofrece el mejor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
                elif factor == 'cta_type':
                    recommendations.append(
                        f"Los CTAs de tipo '{top_value}' generan el mayor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
                elif factor == 'audience_demographic':
                    recommendations.append(
                        f"El segmento demográfico '{top_value}' produce el mejor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
                elif factor == 'audience_geo_distribution':
                    recommendations.append(
                        f"La distribución geográfica '{top_value}' ofrece el mejor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
                elif factor == 'product_category':
                    recommendations.append(
                        f"La categoría de producto '{top_value}' genera el mayor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
                else:
                    recommendations.append(
                        f"El factor '{factor}' con valor '{top_value}' produce el mejor {self.target_metric} con un promedio de {top_mean:.2f}"
                    )
        
        # Recomendaciones para factores binarios
        binary_factors = ['has_sponsorship', 'has_affiliate_links']
        for factor in binary_factors:
            if factor in factor_analysis:
                analysis = factor_analysis[factor]
                if len(analysis['top_values']) > 0:
                    top_value = analysis['top_values'][0][factor]
                    top_mean = analysis['top_values'][0]['mean']
                    
                    if top_value:  # Si el valor es True
                        if factor == 'has_sponsorship':
                            recommendations.append(
                                f"Los contenidos con patrocinio generan un {self.target_metric} promedio de {top_mean:.2f}"
                            )
                        elif factor == 'has_affiliate_links':
                            recommendations.append(
                                f"Los contenidos con enlaces de afiliados generan un {self.target_metric} promedio de {top_mean:.2f}"
                            )
        
        # Recomendaciones basadas en combinaciones
        if combination_analysis:
            top_combinations = sorted(
                [(k, v[0]['mean']) for k, v in combination_analysis.items() if v],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for combo_name, combo_mean in top_combinations:
                factors = combo_name.split('_')
                if len(factors) == 2:
                    factor1, factor2 = factors
                    combo_data = combination_analysis[combo_name][0]
                    value1 = combo_data[factor1]
                    value2 = combo_data[factor2]
                    
                    recommendations.append(
                        f"La combinación de '{factor1}={value1}' con '{factor2}={value2}' "
                        f"produce el mejor {self.target_metric} con un promedio de {combo_mean:.2f}"
                    )
        
        # Recomendaciones generales basadas en el análisis
        if factor_analysis:
            recommendations.append("\nRecomendaciones generales:")
            
            # Recomendar las mejores plataformas
            if 'platform_name' in factor_analysis:
                top_platforms = [item['platform_name'] for item in factor_analysis['platform_name']['top_values'][:2]]
                recommendations.append(
                    f"- Priorizar contenido en las plataformas: {', '.join(top_platforms)}"
                )
            
            # Recomendar las mejores categorías
            if 'content_category' in factor_analysis:
                top_categories = [item['content_category'] for item in factor_analysis['content_category']['top_values'][:2]]
                recommendations.append(
                    f"- Enfocarse en las categorías de contenido: {', '.join(top_categories)}"
                )
            
            # Recomendar los mejores temas
            if 'content_topic' in factor_analysis:
                top_topics = [item['content_topic'] for item in factor_analysis['content_topic']['top_values'][:2]]
                recommendations.append(
                    f"- Desarrollar contenido sobre los temas: {', '.join(top_topics)}"
                )
            
            # Recomendar los mejores CTAs
            if 'cta_type' in factor_analysis:
                top_ctas = [item['cta_type'] for item in factor_analysis['cta_type']['top_values'][:2]]
                recommendations.append(
                    f"- Utilizar CTAs de tipo: {', '.join(top_ctas)}"
                )
            
            # Recomendar sobre patrocinios y afiliados
            if 'has_sponsorship' in factor_analysis and 'has_affiliate_links' in factor_analysis:
                sponsor_value = factor_analysis['has_sponsorship']['top_values'][0]['has_sponsorship']
                affiliate_value = factor_analysis['has_affiliate_links']['top_values'][0]['has_affiliate_links']
                
                if sponsor_value and affiliate_value:
                    recommendations.append(
                        "- Incluir tanto patrocinios como enlaces de afiliados en el contenido"
                    )
                elif sponsor_value:
                    recommendations.append(
                        "- Priorizar contenido patrocinado sobre enlaces de afiliados"
                    )
                elif affiliate_value:
                    recommendations.append(
                        "- Priorizar enlaces de afiliados sobre contenido patrocinado"
                    )
        
        return recommendations
    
    def forecast_revenue(self, historical_data: pd.DataFrame, 
                        forecast_periods: int = 12,
                        frequency: str = 'M') -> Dict[str, Any]:
        """
        Realiza pronóstico de ingresos futuros basado en datos históricos
        
        Args:
            historical_data: DataFrame con datos históricos (debe incluir columna de fecha)
            forecast_periods: Número de períodos a pronosticar
            frequency: Frecuencia de los datos ('D'=diario, 'W'=semanal, 'M'=mensual)
            
        Returns:
            Pronóstico de ingresos
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            import matplotlib.pyplot as plt
            from datetime import datetime
        except ImportError:
            return {'error': "Se requieren los paquetes statsmodels y matplotlib para pronósticos"}
        
        # Verificar que los datos tengan una columna de fecha
        date_columns = [col for col in historical_data.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_columns:
            return {'error': "Se requiere una columna de fecha para realizar pronósticos"}
        
        date_column = date_columns[0]
        
        # Verificar que la métrica objetivo esté presente
        if self.target_metric not in historical_data.columns:
            return {'error': f"La métrica {self.target_metric} no está en los datos históricos"}
        
        # Preparar datos para pronóstico
        try:
            # Convertir a datetime si es necesario
            if not pd.api.types.is_datetime64_any_dtype(historical_data[date_column]):
                historical_data[date_column] = pd.to_datetime(historical_data[date_column])
            
            # Ordenar por fecha
            historical_data = historical_data.sort_values(date_column)
            
            # Crear serie temporal
            ts_data = historical_data.set_index(date_column)[self.target_metric]
            
            # Resamplear a la frecuencia deseada
            ts_data = ts_data.resample(frequency).mean().fillna(method='ffill')
            
            # Dividir en entrenamiento y prueba
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data[:train_size]
            test_data = ts_data[train_size:]
            
            # Entrenar modelos
            models = {}
            forecasts = {}
            
            # Modelo SARIMA
            try:
                sarima_model = SARIMAX(
                    train_data,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12) if frequency == 'M' else (0, 0, 0, 0),
                    enforce_stationarity=False
                )
                sarima_results = sarima_model.fit(disp=False)
                
                # Pronóstico en datos de prueba
                sarima_test_pred = sarima_results.get_forecast(steps=len(test_data))
                sarima_test_mean = sarima_test_pred.predicted_mean
                
                # Calcular error
                sarima_mape = mean_absolute_percentage_error(test_data, sarima_test_mean) * 100
                
                # Pronóstico futuro
                sarima_forecast = sarima_results.get_forecast(steps=forecast_periods)
                sarima_mean = sarima_forecast.predicted_mean
                sarima_conf_int = sarima_forecast.conf_int()
                
                models['sarima'] = {
                    'mape': sarima_mape,
                    'model': sarima_results
                }
                
                forecasts['sarima'] = {
                    'mean': sarima_mean.tolist(),
                    'lower': sarima_conf_int.iloc[:, 0].tolist(),
                    'upper': sarima_conf_int.iloc[:, 1].tolist(),
                    'dates': sarima_mean.index.strftime('%Y-%m-%d').tolist()
                }
            except:
                self.logger.warning("No se pudo entrenar el modelo SARIMA")
            
            # Modelo Holt-Winters
            try:
                hw_model = ExponentialSmoothing(
                    train_data,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=12 if frequency == 'M' else 7 if frequency == 'W' else 1
                )
                hw_results = hw_model.fit()
                
                # Pronóstico en datos de prueba
                hw_test_pred = hw_results.forecast(steps=len(test_data))
                
                # Calcular error
                hw_mape = mean_absolute_percentage_error(test_data, hw_test_pred) * 100
                
                # Pronóstico futuro
                hw_forecast = hw_results.forecast(steps=forecast_periods)
                
                models['holt_winters'] = {
                    'mape': hw_mape,
                    'model': hw_results
                }
                
                forecasts['holt_winters'] = {
                    'mean': hw_forecast.tolist(),
                    'dates': hw_forecast.index.strftime('%Y-%m-%d').tolist()
                }
            except:
                self.logger.warning("No se pudo entrenar el modelo Holt-Winters")
            
            # Seleccionar el mejor modelo
            if models:
                best_model = min(models.items(), key=lambda x: x[1]['mape'])[0]
                best_mape = models[best_model]['mape']
                
                # Generar gráfico
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Datos históricos
                    plt.plot(ts_data.index, ts_data.values, label='Histórico', color='blue')
                    
                    # Pronóstico del mejor modelo
                    forecast_dates = pd.date_range(
                        start=ts_data.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq=frequency
                    )
                    
                    best_forecast = forecasts[best_model]['mean']
                    plt.plot(forecast_dates, best_forecast, label=f'Pronóstico ({best_model})', color='red')
                    
                    # Intervalo de confianza si está disponible
                    if 'lower' in forecasts[best_model] and 'upper' in forecasts[best_model]:
                        plt.fill_between(
                            forecast_dates,
                            forecasts[best_model]['lower'],
                            forecasts[best_model]['upper'],
                            color='red',
                            alpha=0.2,
                            label='Intervalo de confianza 95%'
                        )
                    
                    plt.title(f'Pronóstico de {self.target_metric} - MAPE: {best_mape:.2f}%')
                    plt.xlabel('Fecha')
                    plt.ylabel(self.target_metric)
                    plt.legend()
                    plt.grid(True)
                    
                    # Guardar gráfico
                    plot_path = f"revenue_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(plot_path)
                    plt.close()
                except:
                    plot_path = None
                    self.logger.warning("No se pudo generar el gráfico de pronóstico")
                
                # Calcular métricas de tendencia
                last_value = ts_data.iloc[-1]
                forecast_values = forecasts[best_model]['mean']
                
                growth_rate = ((forecast_values[-1] / last_value) - 1) * 100
                trend_direction = "ascendente" if growth_rate > 0 else "descendente" if growth_rate < 0 else "estable"
                
                # Preparar resultado
                result = {
                    'metric': self.target_metric,
                    'best_model': best_model,
                    'mape': best_mape,
                    'forecast': {
                        'dates': forecasts[best_model]['dates'],
                        'values': forecasts[best_model]['mean']
                    },
                    'trend': {
                        'direction': trend_direction,
                        'growth_rate': growth_rate
                    },
                    'current_value': float(last_value),
                    'forecast_end_value': float(forecast_values[-1]),
                    'plot_path': plot_path
                }
                
                # Añadir recomendaciones
                result['recommendations'] = self._generate_forecast_recommendations(
                    trend_direction, growth_rate, best_mape
                )
                
                return result
            else:
                return {'error': "No se pudo entrenar ningún modelo de pronóstico"}
            
        except Exception as e:
            self.logger.error(f"Error en pronóstico: {str(e)}")
            return {'error': f"Error en pronóstico: {str(e)}"}
    
    def _generate_forecast_recommendations(self, trend_direction: str, 
                                          growth_rate: float, 
                                          mape: float) -> List[str]:
        """
        Genera recomendaciones basadas en el pronóstico de ingresos
        
        Args:
            trend_direction: Dirección de la tendencia
            growth_rate: Tasa de crecimiento
            mape: Error porcentual absoluto medio
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Evaluar la confiabilidad del pronóstico
        if mape < 10:
            reliability = "alta"
        elif mape < 20:
            reliability = "moderada"
        else:
            reliability = "baja"
        
        recommendations.append(
            f"La confiabilidad del pronóstico es {reliability} (MAPE: {mape:.2f}%)"
        )
        
        # Recomendaciones según la tendencia
        if trend_direction == "ascendente":
            recommendations.append(
                f"Se proyecta un crecimiento del {abs(growth_rate):.2f}% en {self.target_metric}"
            )
            
            if growth_rate > 50:
                recommendations.append(
                    "Recomendaciones para tendencia fuertemente ascendente:"
                    "\n- Aumentar la inversión en contenido"
                    "\n- Expandir a nuevos nichos relacionados"
                    "\n- Considerar estrategias de monetización premium"
                    "\n- Preparar infraestructura para mayor escala"
                )
            elif growth_rate > 20:
                recommendations.append(
                    "Recomendaciones para tendencia moderadamente ascendente:"
                    "\n- Mantener la estrategia actual"
                    "\n- Optimizar los canales más rentables"
                    "\n- Experimentar con nuevos formatos de contenido"
                    "\n- Reinvertir ganancias en mejoras de calidad"
                )
            else:
                recommendations.append(
                    "Recomendaciones para tendencia ligeramente ascendente:"
                    "\n- Mantener la estrategia actual"
                    "\n- Realizar pruebas A/B para mejorar conversión"
                    "\n- Optimizar CTAs para aumentar ingresos"
                )
        elif trend_direction == "descendente":
            recommendations.append(
                f"Se proyecta una disminución del {abs(growth_rate):.2f}% en {self.target_metric}"
            )
            
            if growth_rate < -50:
                recommendations.append(
                    "Recomendaciones para tendencia fuertemente descendente:"
                    "\n- Revisar urgentemente la estrategia de contenido"
                    "\n- Diversificar fuentes de ingresos"
                    "\n- Considerar pivotar a nichos más rentables"
                    "\n- Reducir costos operativos"
                    "\n- Analizar competencia y cambios en el mercado"
                )
            elif growth_rate < -20:
                recommendations.append(
                    "Recomendaciones para tendencia moderadamente descendente:"
                    "\n- Revisar y ajustar la estrategia de contenido"
                    "\n- Experimentar con nuevos formatos y CTAs"
                    "\n- Analizar qué canales están perdiendo rentabilidad"
                    "\n- Considerar nuevas fuentes de monetización"
                )
            else:
                recommendations.append(
                    "Recomendaciones para tendencia ligeramente descendente:"
                    "\n- Optimizar CTAs y estrategias de monetización"
                    "\n- Realizar pruebas A/B para identificar mejoras"
                    "\n- Revisar la calidad del contenido"
                )
        else:
            recommendations.append(
                f"Se proyecta que {self.target_metric} se mantendrá estable"
            )
            
            recommendations.append(
                "Recomendaciones para tendencia estable:"
                "\n- Mantener la estrategia actual"
                "\n- Buscar oportunidades para innovar y generar crecimiento"
                "\n- Optimizar costos para mejorar márgenes"
            )
        
        # Recomendaciones según la confiabilidad
        if reliability == "baja":
            recommendations.append(
                "Debido a la baja confiabilidad del pronóstico:"
                "\n- Tomar decisiones con cautela"
                "\n- Recopilar más datos históricos"
                "\n- Considerar factores externos no capturados por el modelo"
            )
        
        return recommendations
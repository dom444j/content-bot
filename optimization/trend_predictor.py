"""
Predictor de tendencias que analiza datos históricos y actuales para anticipar
tendencias futuras y oportunidades de contenido.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

class TrendPredictor:
    """
    Predictor de tendencias que analiza datos históricos y actuales para anticipar
    tendencias futuras y oportunidades de contenido.
    
    Capacidades:
    - Análisis de tendencias históricas
    - Predicción de tendencias emergentes
    - Identificación de oportunidades de contenido
    - Visualización de tendencias
    """
    
    def __init__(self, model_path: Optional[str] = None, data_path: Optional[str] = None):
        """
        Inicializa el predictor de tendencias.
        
        Args:
            model_path: Ruta opcional al modelo pre-entrenado
            data_path: Ruta opcional al archivo de datos de tendencias
        """
        self.logger = logging.getLogger(__name__)
        
        # Configurar rutas
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models',
            'trend_predictor_model.joblib'
        )
        
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'trend_data.json'
        )
        
        # Inicializar modelos
        self.models = {}
        self.scalers = {}
        
        # Cargar datos existentes o crear nuevos
        self.trend_data = self._load_data()
        
        # Cargar modelos si existen
        self._load_models()
        
        # Configurar categorías de tendencias
        self.trend_categories = [
            'finance', 'crypto', 'health', 'fitness', 'technology', 
            'ai', 'gaming', 'entertainment', 'education', 'lifestyle'
        ]
        
        # Configurar fuentes de datos
        self.data_sources = {
            'google_trends': {
                'enabled': True,
                'api_key': None,  # No requiere API key para pytrends
                'url': None
            },
            'twitter': {
                'enabled': True,
                'api_key': os.environ.get('TWITTER_API_KEY', ''),
                'url': 'https://api.twitter.com/2/trends/place'
            },
            'youtube': {
                'enabled': True,
                'api_key': os.environ.get('YOUTUBE_API_KEY', ''),
                'url': 'https://www.googleapis.com/youtube/v3/videos'
            }
        }
    
    def _load_data(self) -> Dict[str, Any]:
        """
        Carga datos de tendencias existentes o crea una estructura nueva.
        
        Returns:
            Datos de tendencias
        """
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Crear estructura de datos inicial
                return {
                    'historical_trends': {},
                    'predicted_trends': {},
                    'trend_performance': {},
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Error al cargar datos de tendencias: {str(e)}")
            # Devolver estructura vacía en caso de error
            return {
                'historical_trends': {},
                'predicted_trends': {},
                'trend_performance': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def _save_data(self) -> bool:
        """
        Guarda los datos de tendencias en disco.
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            # Actualizar timestamp
            self.trend_data['last_updated'] = datetime.now().isoformat()
            
            # Guardar datos
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.trend_data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar datos de tendencias: {str(e)}")
            return False
    
    def _load_models(self) -> bool:
        """
        Carga modelos pre-entrenados si existen.
        
        Returns:
            True si se cargaron correctamente, False en caso contrario
        """
        try:
            model_dir = os.path.dirname(self.model_path)
            
            # Verificar si existe el directorio
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                return False
            
            # Cargar modelos por categoría
            for category in self.trend_categories:
                model_file = os.path.join(model_dir, f"trend_model_{category}.joblib")
                scaler_file = os.path.join(model_dir, f"trend_scaler_{category}.joblib")
                
                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    self.models[category] = joblib.load(model_file)
                    self.scalers[category] = joblib.load(scaler_file)
            
            if self.models:
                self.logger.info(f"Modelos cargados: {list(self.models.keys())}")
                return True
            else:
                self.logger.info("No se encontraron modelos pre-entrenados")
                return False
                
        except Exception as e:
            self.logger.error(f"Error al cargar modelos: {str(e)}")
            return False
    
    def _save_models(self) -> bool:
        """
        Guarda los modelos entrenados en disco.
        
        Returns:
            True si se guardaron correctamente, False en caso contrario
        """
        try:
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            
            # Guardar modelos por categoría
            for category, model in self.models.items():
                model_file = os.path.join(model_dir, f"trend_model_{category}.joblib")
                joblib.dump(model, model_file)
                
                if category in self.scalers:
                    scaler_file = os.path.join(model_dir, f"trend_scaler_{category}.joblib")
                    joblib.dump(self.scalers[category], scaler_file)
            
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar modelos: {str(e)}")
            return False
    
    def fetch_current_trends(self, categories: List[str] = None) -> Dict[str, Any]:
        """
        Obtiene tendencias actuales de diversas fuentes.
        
        Args:
            categories: Lista opcional de categorías a consultar
            
        Returns:
            Tendencias actuales por categoría y fuente
        """
        if categories is None:
            categories = self.trend_categories
        
        current_trends = {
            'timestamp': datetime.now().isoformat(),
            'trends': {}
        }
        
        try:
            # Implementar conexión con Google Trends (pytrends)
            if self.data_sources['google_trends']['enabled']:
                try:
                    from pytrends.request import TrendReq
                    
                    pytrends = TrendReq(hl='es-ES', tz=360)
                    
                    for category in categories:
                        # Consultar tendencias por categoría
                        kw_list = [category]
                        pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')
                        
                        # Obtener tendencias relacionadas
                        related_queries = pytrends.related_queries()
                        
                        if category in related_queries and related_queries[category]:
                            top_queries = related_queries[category]['top']
                            rising_queries = related_queries[category]['rising']
                            
                            if top_queries is not None and not top_queries.empty:
                                if category not in current_trends['trends']:
                                    current_trends['trends'][category] = {}
                                
                                current_trends['trends'][category]['google_trends'] = {
                                                                        'top': top_queries.head(10).to_dict('records'),
                                    'rising': rising_queries.head(10).to_dict('records') if rising_queries is not None and not rising_queries.empty else []
                                }
                
                # Obtener datos de interés a lo largo del tiempo
                interest_over_time = pytrends.interest_over_time()
                if not interest_over_time.empty:
                    if category not in current_trends['trends']:
                        current_trends['trends'][category] = {}
                    
                    if 'google_trends' not in current_trends['trends'][category]:
                        current_trends['trends'][category]['google_trends'] = {}
                    
                    # Convertir a formato serializable
                    interest_data = interest_over_time.reset_index()
                    interest_data['date'] = interest_data['date'].dt.strftime('%Y-%m-%d')
                    
                    current_trends['trends'][category]['google_trends']['interest_over_time'] = interest_data.to_dict('records')
                
            except ImportError:
                self.logger.warning("Biblioteca pytrends no instalada. No se pueden obtener datos de Google Trends.")
                
            except Exception as e:
                self.logger.error(f"Error al obtener datos de Google Trends: {str(e)}")
            
            # Implementar conexión con Twitter/X API
            if self.data_sources['twitter']['enabled'] and self.data_sources['twitter']['api_key']:
                try:
                    # Configurar headers para la API
                    headers = {
                        'Authorization': f"Bearer {self.data_sources['twitter']['api_key']}",
                        'Content-Type': 'application/json'
                    }
                    
                    # Obtener tendencias globales (ID 1 para tendencias mundiales)
                    response = requests.get(
                        f"{self.data_sources['twitter']['url']}/1",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        twitter_trends = response.json()
                        
                        # Procesar tendencias por categoría
                        for category in categories:
                            # Filtrar tendencias relevantes para la categoría
                            # Esto es simplificado, idealmente se usaría NLP para categorizar
                            relevant_trends = [
                                trend for trend in twitter_trends[0]['trends']
                                if category.lower() in trend['name'].lower() or
                                category.lower() in trend.get('query', '').lower()
                            ]
                            
                            if relevant_trends:
                                if category not in current_trends['trends']:
                                    current_trends['trends'][category] = {}
                                
                                current_trends['trends'][category]['twitter'] = relevant_trends[:10]
                    else:
                        self.logger.warning(f"Error al obtener tendencias de Twitter: {response.status_code}")
                
                except Exception as e:
                    self.logger.error(f"Error al conectar con API de Twitter: {str(e)}")
            
            # Implementar conexión con YouTube API
            if self.data_sources['youtube']['enabled'] and self.data_sources['youtube']['api_key']:
                try:
                    for category in categories:
                        # Parámetros para la búsqueda
                        params = {
                            'part': 'snippet,statistics',
                            'chart': 'mostPopular',
                            'regionCode': 'ES',  # Código de país (España)
                            'maxResults': 10,
                            'videoCategoryId': self._get_youtube_category_id(category),
                            'key': self.data_sources['youtube']['api_key']
                        }
                        
                        response = requests.get(
                            self.data_sources['youtube']['url'],
                            params=params
                        )
                        
                        if response.status_code == 200:
                            youtube_trends = response.json()
                            
                            if 'items' in youtube_trends and youtube_trends['items']:
                                if category not in current_trends['trends']:
                                    current_trends['trends'][category] = {}
                                
                                # Simplificar datos para almacenamiento
                                simplified_trends = []
                                for item in youtube_trends['items']:
                                    simplified_trends.append({
                                        'id': item['id'],
                                        'title': item['snippet']['title'],
                                        'channelTitle': item['snippet']['channelTitle'],
                                        'publishedAt': item['snippet']['publishedAt'],
                                        'viewCount': item['statistics'].get('viewCount', 0),
                                        'likeCount': item['statistics'].get('likeCount', 0),
                                        'commentCount': item['statistics'].get('commentCount', 0)
                                    })
                                
                                current_trends['trends'][category]['youtube'] = simplified_trends
                        else:
                            self.logger.warning(f"Error al obtener tendencias de YouTube: {response.status_code}")
                
                except Exception as e:
                    self.logger.error(f"Error al conectar con API de YouTube: {str(e)}")
            
            # Guardar tendencias en datos históricos
            date_key = datetime.now().strftime('%Y-%m-%d')
            if date_key not in self.trend_data['historical_trends']:
                self.trend_data['historical_trends'][date_key] = {}
            
            self.trend_data['historical_trends'][date_key] = current_trends
            self._save_data()
            
            return current_trends
            
        except Exception as e:
            self.logger.error(f"Error al obtener tendencias actuales: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'trends': {}
            }
    
    def _get_youtube_category_id(self, category: str) -> str:
        """
        Convierte una categoría general a ID de categoría de YouTube.
        
        Args:
            category: Categoría general
            
        Returns:
            ID de categoría de YouTube
        """
        # Mapeo de categorías a IDs de YouTube
        # Referencia: https://developers.google.com/youtube/v3/docs/videoCategories
        category_mapping = {
            'finance': '20',  # Blogs y vlogs
            'crypto': '20',   # Blogs y vlogs
            'health': '26',   # Deportes y salud
            'fitness': '26',  # Deportes y salud
            'technology': '28', # Ciencia y tecnología
            'ai': '28',       # Ciencia y tecnología
            'gaming': '20',   # Videojuegos
            'entertainment': '24', # Entretenimiento
            'education': '27', # Educación
            'lifestyle': '22'  # Gente y blogs
        }
        
        return category_mapping.get(category.lower(), '0')  # 0 es categoría de película
    
    def train_models(self, training_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Entrena modelos predictivos para cada categoría de tendencias.
        
        Args:
            training_data: DataFrame opcional con datos de entrenamiento
            
        Returns:
            Resultados del entrenamiento
        """
        results = {}
        
        try:
            # Si no se proporcionan datos, usar datos históricos
            if training_data is None:
                training_data = self._prepare_historical_data_for_training()
            
            if training_data.empty:
                return {'error': 'No hay suficientes datos para entrenar los modelos'}
            
            # Entrenar modelo para cada categoría
            for category in self.trend_categories:
                # Filtrar datos para la categoría
                category_data = training_data[training_data['category'] == category]
                
                if len(category_data) < 10:
                    self.logger.warning(f"Insuficientes datos para entrenar modelo de {category}")
                    results[category] = {'status': 'error', 'message': 'Insuficientes datos'}
                    continue
                
                # Preparar características y objetivo
                X = category_data.drop(['category', 'trend_score', 'date'], axis=1)
                y = category_data['trend_score']
                
                # Escalar características
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Dividir en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Entrenar modelo
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluar modelo
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Guardar modelo y scaler
                self.models[category] = model
                self.scalers[category] = scaler
                
                # Guardar resultados
                results[category] = {
                    'status': 'success',
                    'train_score': train_score,
                    'test_score': test_score,
                    'samples': len(category_data)
                }
                
                self.logger.info(f"Modelo para {category} entrenado. R² test: {test_score:.4f}")
            
            # Guardar modelos
            self._save_models()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error al entrenar modelos: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_historical_data_for_training(self) -> pd.DataFrame:
        """
        Prepara datos históricos para entrenamiento.
        
        Returns:
            DataFrame con datos preparados
        """
        try:
            data_rows = []
            
            # Procesar datos históricos
            for date, trends in self.trend_data['historical_trends'].items():
                if 'trends' not in trends:
                    continue
                
                for category, sources in trends['trends'].items():
                    # Extraer métricas de Google Trends
                    if 'google_trends' in sources:
                        google_data = sources['google_trends']
                        
                        # Calcular puntuación de tendencia basada en datos disponibles
                        trend_score = 0
                        
                        # Usar datos de top queries si están disponibles
                        if 'top' in google_data and google_data['top']:
                            # Calcular promedio de búsquedas
                            if isinstance(google_data['top'], list) and len(google_data['top']) > 0:
                                if 'value' in google_data['top'][0]:
                                    values = [item['value'] for item in google_data['top']]
                                    trend_score += sum(values) / len(values)
                        
                        # Usar datos de interest_over_time si están disponibles
                        if 'interest_over_time' in google_data and google_data['interest_over_time']:
                            if isinstance(google_data['interest_over_time'], list) and len(google_data['interest_over_time']) > 0:
                                if category in google_data['interest_over_time'][0]:
                                    values = [item[category] for item in google_data['interest_over_time']]
                                    trend_score += sum(values) / len(values)
                        
                        # Crear fila de datos
                        row = {
                            'date': date,
                            'category': category,
                            'trend_score': trend_score,
                            'has_google_data': 1,
                            'has_twitter_data': 1 if 'twitter' in sources else 0,
                            'has_youtube_data': 1 if 'youtube' in sources else 0
                        }
                        
                        # Agregar métricas adicionales si están disponibles
                        if 'twitter' in sources and isinstance(sources['twitter'], list) and len(sources['twitter']) > 0:
                            if 'tweet_volume' in sources['twitter'][0]:
                                tweet_volumes = [item.get('tweet_volume', 0) for item in sources['twitter'] if item.get('tweet_volume') is not None]
                                if tweet_volumes:
                                    row['avg_tweet_volume'] = sum(tweet_volumes) / len(tweet_volumes)
                                else:
                                    row['avg_tweet_volume'] = 0
                            else:
                                row['avg_tweet_volume'] = 0
                        else:
                            row['avg_tweet_volume'] = 0
                        
                        if 'youtube' in sources and isinstance(sources['youtube'], list) and len(sources['youtube']) > 0:
                            view_counts = [int(item.get('viewCount', 0)) for item in sources['youtube']]
                            like_counts = [int(item.get('likeCount', 0)) for item in sources['youtube']]
                            
                            if view_counts:
                                row['avg_view_count'] = sum(view_counts) / len(view_counts)
                            else:
                                row['avg_view_count'] = 0
                                
                            if like_counts:
                                row['avg_like_count'] = sum(like_counts) / len(like_counts)
                            else:
                                row['avg_like_count'] = 0
                        else:
                            row['avg_view_count'] = 0
                            row['avg_like_count'] = 0
                        
                        data_rows.append(row)
            
            # Crear DataFrame
            if data_rows:
                df = pd.DataFrame(data_rows)
                
                # Convertir fecha a datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Ordenar por fecha
                df = df.sort_values('date')
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error al preparar datos históricos: {str(e)}")
            return pd.DataFrame()
    
    def predict_trends(self, horizon_days: int = 7, categories: List[str] = None) -> Dict[str, Any]:
        """
        Predice tendencias futuras para un horizonte específico.
        
        Args:
            horizon_days: Número de días en el futuro para predecir
            categories: Lista opcional de categorías a predecir
            
        Returns:
            Predicciones de tendencias
        """
        if categories is None:
            categories = self.trend_categories
        
        predictions = {
            'timestamp': datetime.now().isoformat(),
            'horizon_days': horizon_days,
            'predictions': {}
        }
        
        try:
            # Verificar si hay modelos entrenados
            if not self.models:
                self.logger.warning("No hay modelos entrenados disponibles para predicción")
                return {'error': 'No hay modelos entrenados'}
            
            # Obtener datos actuales para predicción
            current_data = self.fetch_current_trends(categories)
            
            # Preparar datos para predicción
            prediction_data = self._prepare_data_for_prediction(current_data)
            
            # Predecir para cada categoría
            for category in categories:
                if category not in self.models or category not in self.scalers:
                    self.logger.warning(f"No hay modelo entrenado para la categoría {category}")
                    continue
                
                # Filtrar datos para la categoría
                category_data = prediction_data[prediction_data['category'] == category]
                
                if category_data.empty:
                    continue
                
                # Preparar características
                X = category_data.drop(['category', 'date'], axis=1)
                
                # Escalar características
                X_scaled = self.scalers[category].transform(X)
                
                # Predecir tendencia actual
                current_trend = self.models[category].predict(X_scaled)[0]
                
                # Usar modelo ARIMA para proyectar tendencia futura
                try:
                    # Obtener datos históricos de tendencia
                    historical_trends = self._get_historical_trend_scores(category)
                    
                    if len(historical_trends) >= 5:  # Mínimo de datos para ARIMA
                        # Ajustar modelo ARIMA
                        model = ARIMA(historical_trends, order=(1, 1, 1))
                        model_fit = model.fit()
                        
                        # Predecir para horizonte
                        future_trends = model_fit.forecast(steps=horizon_days)
                        
                        # Crear predicciones diarias
                        daily_predictions = []
                        start_date = datetime.now()
                        
                        for i, trend in enumerate(future_trends):
                            prediction_date = start_date + timedelta(days=i+1)
                            daily_predictions.append({
                                'date': prediction_date.strftime('%Y-%m-%d'),
                                'trend_score': float(trend),
                                'confidence': 0.8 - (i * 0.05)  # Confianza disminuye con el horizonte
                            })
                        
                        predictions['predictions'][category] = {
                            'current_trend': float(current_trend),
                            'daily_forecast': daily_predictions,
                            'trending_direction': 'up' if future_trends[-1] > current_trend else 'down',
                            'peak_day': max(range(len(future_trends)), key=lambda i: future_trends[i])
                        }
                    else:
                        # Si no hay suficientes datos históricos, usar tendencia constante
                        daily_predictions = []
                        start_date = datetime.now()
                        
                        for i in range(horizon_days):
                            prediction_date = start_date + timedelta(days=i+1)
                            daily_predictions.append({
                                'date': prediction_date.strftime('%Y-%m-%d'),
                                'trend_score': float(current_trend),
                                'confidence': 0.6 - (i * 0.05)
                            })
                        
                        predictions['predictions'][category] = {
                            'current_trend': float(current_trend),
                            'daily_forecast': daily_predictions,
                            'trending_direction': 'stable',
                            'peak_day': 0
                        }
                
                except Exception as e:
                    self.logger.error(f"Error en predicción ARIMA para {category}: {str(e)}")
                    
                    # Usar enfoque simple si ARIMA falla
                    daily_predictions = []
                    start_date = datetime.now()
                    
                    for i in range(horizon_days):
                        prediction_date = start_date + timedelta(days=i+1)
                        # Tendencia simple con decaimiento
                        trend = current_trend * (1 - (i * 0.1))
                        daily_predictions.append({
                            'date': prediction_date.strftime('%Y-%m-%d'),
                            'trend_score': float(trend),
                            'confidence': 0.5 - (i * 0.05)
                        })
                    
                    predictions['predictions'][category] = {
                        'current_trend': float(current_trend),
                        'daily_forecast': daily_predictions,
                        'trending_direction': 'down',
                        'peak_day': 0
                    }
            
            # Guardar predicciones
            date_key = datetime.now().strftime('%Y-%m-%d')
            self.trend_data['predicted_trends'][date_key] = predictions
            self._save_data()
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error al predecir tendencias: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'predictions': {}
            }
    
    def _prepare_data_for_prediction(self, current_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepara datos actuales para predicción.
        
        Args:
            current_data: Datos actuales de tendencias
            
        Returns:
            DataFrame preparado para predicción
        """
        try:
            data_rows = []
            
            if 'trends' not in current_data:
                return pd.DataFrame()
            
            # Procesar datos actuales
            for category, sources in current_data['trends'].items():
                # Extraer métricas de Google Trends
                if 'google_trends' in sources:
                    google_data = sources['google_trends']
                    
                    # Crear fila de datos
                    row = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'category': category,
                        'has_google_data': 1,
                        'has_twitter_data': 1 if 'twitter' in sources else 0,
                        'has_youtube_data': 1 if 'youtube' in sources else 0
                    }
                    
                    # Agregar métricas adicionales si están disponibles
                    if 'twitter' in sources and isinstance(sources['twitter'], list) and len(sources['twitter']) > 0:
                        if 'tweet_volume' in sources['twitter'][0]:
                            tweet_volumes = [item.get('tweet_volume', 0) for item in sources['twitter'] if item.get('tweet_volume') is not None]
                            if tweet_volumes:
                                row['avg_tweet_volume'] = sum(tweet_volumes) / len(tweet_volumes)
                            else:
                                row['avg_tweet_volume'] = 0
                        else:
                            row['avg_tweet_volume'] = 0
                    else:
                        row['avg_tweet_volume'] = 0
                    
                    if 'youtube' in sources and isinstance(sources['youtube'], list) and len(sources['youtube']) > 0:
                        view_counts = [int(item.get('viewCount', 0)) for item in sources['youtube']]
                        like_counts = [int(item.get('likeCount', 0)) for item in sources['youtube']]
                        
                        if view_counts:
                            row['avg_view_count'] = sum(view_counts) / len(view_counts)
                        else:
                            row['avg_view_count'] = 0
                            
                        if like_counts:
                            row['avg_like_count'] = sum(like_counts) / len(like_counts)
                        else:
                            row['avg_like_count'] = 0
                    else:
                        row['avg_view_count'] = 0
                        row['avg_like_count'] = 0
                    
                    data_rows.append(row)
            
            # Crear DataFrame
            if data_rows:
                df = pd.DataFrame(data_rows)
                
                # Convertir fecha a datetime
                df['date'] = pd.to_datetime(df['date'])
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error al preparar datos para predicción: {str(e)}")
            return pd.DataFrame()
    
    def _get_historical_trend_scores(self, category: str) -> List[float]:
        """
        Obtiene puntuaciones históricas de tendencia para una categoría.
        
        Args:
            category: Categoría de tendencia
            
        Returns:
            Lista de puntuaciones históricas
        """
        trend_scores = []
        
        try:
            # Ordenar fechas cronológicamente
            sorted_dates = sorted(self.trend_data['historical_trends'].keys())
            
            for date in sorted_dates:
                trends = self.trend_data['historical_trends'][date]
                
                if 'trends' in trends and category in trends['trends']:
                    sources = trends['trends'][category]
                    
                    # Extraer métricas de Google Trends
                    if 'google_trends' in sources:
                        google_data = sources['google_trends']
                        
                        # Calcular puntuación de tendencia
                        trend_score = 0
                        
                        # Usar datos de top queries si están disponibles
                        if 'top' in google_data and google_data['top']:
                            # Calcular promedio de búsquedas
                            if isinstance(google_data['top'], list) and len(google_data['top']) > 0:
                                if 'value' in google_data['top'][0]:
                                    values = [item['value'] for item in google_data['top']]
                                    trend_score += sum(values) / len(values)
                        
                        # Usar datos de interest_over_time si están disponibles
                        if 'interest_over_time' in google_data and google_data['interest_over_time']:
                            if isinstance(google_data['interest_over_time'], list) and len(google_data['interest_over_time']) > 0:
                                if category in google_data['interest_over_time'][0]:
                                    values = [item[category] for item in google_data['interest_over_time']]
                                    trend_score += sum(values) / len(values)
                        
                        trend_scores.append(trend_score)
            
            return trend_scores
            
        except Exception as e:
            self.logger.error(f"Error al obtener puntuaciones históricas: {str(e)}")
            return []
    
    def visualize_trends(self, category: str, days: int = 30) -> Dict[str, Any]:
        """
        Genera visualizaciones de tendencias para una categoría.
        
        Args:
            category: Categoría de tendencia
            days: Número de días a visualizar
            
        Returns:
            Rutas a las visualizaciones generadas
        """
        try:
            # Crear directorio para visualizaciones
            vis_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'visualizations',
                'trends'
            )
            os.makedirs(vis_dir, exist_ok=True)
            
            # Obtener datos históricos
            historical_data = self._get_trend_data_for_visualization(category, days)
            
            if historical_data.empty:
                return {'error': f'No hay datos suficientes para visualizar tendencias de {category}'}
            
            # Generar visualizaciones
            visualizations = {}
            
            # 1. Tendencia histórica
            plt.figure(figsize=(12, 6))
            plt.plot(historical_data['date'], historical_data['trend_score'], marker='o')
            plt.title(f'Tendencia histórica: {category}')
            plt.xlabel('Fecha')
            plt.ylabel('Puntuación de tendencia')
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Guardar gráfico
            trend_path = os.path.join(vis_dir, f'{category}_trend.png')
            plt.tight_layout()
            plt.savefig(trend_path)
            plt.close()
            
            visualizations['trend_chart'] = trend_path
            
            # 2. Heatmap de correlación si hay suficientes características
            if len(historical_data.columns) > 3:
                plt.figure(figsize=(10, 8))
                correlation_columns = [col for col in historical_data.columns if col not in ['date', 'category']]
                correlation = historical_data[correlation_columns].corr()
                
                sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Correlación de métricas: {category}')
                
                # Guardar gráfico
                corr_path = os.path.join(vis_dir, f'{category}_correlation.png')
                plt.tight_layout()
                plt.savefig(corr_path)
                plt.close()
                
                visualizations['correlation_heatmap'] = corr_path
            
            # 3. Predicción futura si está disponible
            predictions = self.predict_trends(horizon_days=7, categories=[category])
            
            if 'predictions' in predictions and category in predictions['predictions']:
                pred_data = predictions['predictions'][category]
                
                if 'daily_forecast' in pred_data:
                    # Crear DataFrame con predicciones
                    pred_df = pd.DataFrame(pred_data['daily_forecast'])
                    pred_df['date'] = pd.to_datetime(pred_df['date'])
                    
                    # Combinar datos históricos y predicciones
                    plt.figure(figsize=(12, 6))
                    
                    # Datos históricos
                    plt.plot(historical_data['date'], historical_data['trend_score'], 
                             marker='o', label='Histórico')
                    
                    # Predicciones
                    plt.plot(pred_df['date'], pred_df['trend_score'], 
                             marker='x', linestyle='--', color='red', label='Predicción')
                    
                    # Área de confianza
                    if 'confidence' in pred_df.columns:
                        upper_bound = pred_df['trend_score'] + pred_df['trend_score'] * pred_df['confidence']
                        lower_bound = pred_df['trend_score'] - pred_df['trend_score'] * pred_df['confidence']
                        
                        plt.fill_between(pred_df['date'], lower_bound, upper_bound, 
                                         color='red', alpha=0.2, label='Intervalo de confianza')
                    
                    plt.title(f'Predicción de tendencia: {category}')
                    plt.xlabel('Fecha')
                    plt.ylabel('Puntuación de tendencia')
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    plt.legend()
                    
                    # Guardar gráfico
                    forecast_path = os.path.join(vis_dir, f'{category}_forecast.png')
                    plt.tight_layout()
                    plt.savefig(forecast_path)
                    plt.close()
                    
                    visualizations['forecast_chart'] = forecast_path
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error al generar visualizaciones: {str(e)}")
            return {'error': str(e)}
    
    def _get_trend_data_for_visualization(self, category: str, days: int) -> pd.DataFrame:
        """
        Obtiene datos de tendencia para visualización.
        
        Args:
            category: Categoría de tendencia
            days: Número de días a incluir
            
        Returns:
            DataFrame con datos para visualización
        """
        try:
            data_rows = []
            
            # Calcular fecha límite
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
                        # Filtrar fechas relevantes
            relevant_dates = [
                date for date in self.trend_data['historical_trends'].keys()
                if date >= cutoff_date
            ]
            
            # Ordenar fechas cronológicamente
            relevant_dates.sort()
            
            for date in relevant_dates:
                trends = self.trend_data['historical_trends'][date]
                
                if 'trends' not in trends or category not in trends['trends']:
                    continue
                
                sources = trends['trends'][category]
                
                # Extraer métricas de Google Trends
                if 'google_trends' in sources:
                    google_data = sources['google_trends']
                    
                    # Calcular puntuación de tendencia
                    trend_score = 0
                    
                    # Usar datos de top queries si están disponibles
                    if 'top' in google_data and google_data['top']:
                        # Calcular promedio de búsquedas
                        if isinstance(google_data['top'], list) and len(google_data['top']) > 0:
                            if 'value' in google_data['top'][0]:
                                values = [item['value'] for item in google_data['top']]
                                trend_score += sum(values) / len(values)
                    
                    # Usar datos de interest_over_time si están disponibles
                    if 'interest_over_time' in google_data and google_data['interest_over_time']:
                        if isinstance(google_data['interest_over_time'], list) and len(google_data['interest_over_time']) > 0:
                            if category in google_data['interest_over_time'][0]:
                                values = [item[category] for item in google_data['interest_over_time']]
                                trend_score += sum(values) / len(values)
                    
                    # Crear fila de datos
                    row = {
                        'date': date,
                        'category': category,
                        'trend_score': trend_score,
                        'has_google_data': 1,
                        'has_twitter_data': 1 if 'twitter' in sources else 0,
                        'has_youtube_data': 1 if 'youtube' in sources else 0
                    }
                    
                    # Agregar métricas adicionales si están disponibles
                    if 'twitter' in sources and isinstance(sources['twitter'], list) and len(sources['twitter']) > 0:
                        if 'tweet_volume' in sources['twitter'][0]:
                            tweet_volumes = [item.get('tweet_volume', 0) for item in sources['twitter'] if item.get('tweet_volume') is not None]
                            if tweet_volumes:
                                row['avg_tweet_volume'] = sum(tweet_volumes) / len(tweet_volumes)
                            else:
                                row['avg_tweet_volume'] = 0
                        else:
                            row['avg_tweet_volume'] = 0
                    else:
                        row['avg_tweet_volume'] = 0
                    
                    if 'youtube' in sources and isinstance(sources['youtube'], list) and len(sources['youtube']) > 0:
                        view_counts = [int(item.get('viewCount', 0)) for item in sources['youtube']]
                        like_counts = [int(item.get('likeCount', 0)) for item in sources['youtube']]
                        
                        if view_counts:
                            row['avg_view_count'] = sum(view_counts) / len(view_counts)
                        else:
                            row['avg_view_count'] = 0
                            
                        if like_counts:
                            row['avg_like_count'] = sum(like_counts) / len(like_counts)
                        else:
                            row['avg_like_count'] = 0
                    else:
                        row['avg_view_count'] = 0
                        row['avg_like_count'] = 0
                    
                    data_rows.append(row)
            
            # Crear DataFrame
            if data_rows:
                df = pd.DataFrame(data_rows)
                
                # Convertir fecha a datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Ordenar por fecha
                df = df.sort_values('date')
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error al obtener datos para visualización: {str(e)}")
            return pd.DataFrame()
    
    def get_trend_recommendations(self, category: str = None, top_n: int = 5) -> Dict[str, Any]:
        """
        Obtiene recomendaciones de tendencias para creación de contenido.
        
        Args:
            category: Categoría opcional para filtrar recomendaciones
            top_n: Número de recomendaciones a devolver
            
        Returns:
            Recomendaciones de tendencias
        """
        try:
            # Obtener predicciones recientes
            date_key = datetime.now().strftime('%Y-%m-%d')
            
            # Si no hay predicciones para hoy, generarlas
            if date_key not in self.trend_data['predicted_trends']:
                self.predict_trends()
            
            # Si sigue sin haber predicciones, usar la más reciente
            if date_key not in self.trend_data['predicted_trends']:
                # Ordenar fechas cronológicamente
                sorted_dates = sorted(self.trend_data['predicted_trends'].keys(), reverse=True)
                
                if not sorted_dates:
                    return {'error': 'No hay predicciones disponibles'}
                
                date_key = sorted_dates[0]
            
            predictions = self.trend_data['predicted_trends'][date_key]
            
            # Filtrar por categoría si se especifica
            if category:
                if 'predictions' in predictions and category in predictions['predictions']:
                    category_predictions = {category: predictions['predictions'][category]}
                else:
                    return {'error': f'No hay predicciones para la categoría {category}'}
            else:
                category_predictions = predictions.get('predictions', {})
            
            # Ordenar categorías por tendencia actual
            sorted_categories = sorted(
                category_predictions.keys(),
                key=lambda cat: category_predictions[cat].get('current_trend', 0),
                reverse=True
            )
            
            # Limitar al número solicitado
            top_categories = sorted_categories[:top_n]
            
            # Obtener tendencias actuales para recomendaciones
            current_trends = self.fetch_current_trends(top_categories)
            
            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'recommendations': []
            }
            
            for cat in top_categories:
                cat_pred = category_predictions[cat]
                
                # Obtener términos de búsqueda populares
                popular_terms = []
                
                if 'trends' in current_trends and cat in current_trends['trends']:
                    sources = current_trends['trends'][cat]
                    
                    # Términos de Google Trends
                    if 'google_trends' in sources:
                        google_data = sources['google_trends']
                        
                        if 'top' in google_data and google_data['top']:
                            if isinstance(google_data['top'], list):
                                for item in google_data['top'][:3]:
                                    if 'query' in item:
                                        popular_terms.append(item['query'])
                    
                    # Términos de Twitter
                    if 'twitter' in sources and isinstance(sources['twitter'], list):
                        for item in sources['twitter'][:3]:
                            if 'name' in item:
                                popular_terms.append(item['name'])
                
                # Crear recomendación
                recommendation = {
                    'category': cat,
                    'trend_score': cat_pred.get('current_trend', 0),
                    'trending_direction': cat_pred.get('trending_direction', 'stable'),
                    'popular_terms': popular_terms,
                    'content_ideas': self._generate_content_ideas(cat, popular_terms)
                }
                
                recommendations['recommendations'].append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al obtener recomendaciones: {str(e)}")
            return {'error': str(e)}
    
    def _generate_content_ideas(self, category: str, terms: List[str]) -> List[str]:
        """
        Genera ideas de contenido basadas en categoría y términos populares.
        
        Args:
            category: Categoría de tendencia
            terms: Términos populares relacionados
            
        Returns:
            Lista de ideas de contenido
        """
        ideas = []
        
        # Plantillas de ideas por categoría
        templates = {
            'finance': [
                "Guía completa sobre {term}",
                "Cómo invertir en {term} en 2023",
                "5 estrategias para optimizar {term}",
                "Análisis de tendencias: ¿Es momento de invertir en {term}?",
                "Comparativa: {term} vs alternativas tradicionales"
            ],
            'crypto': [
                "Guía para principiantes: Todo sobre {term}",
                "Análisis técnico de {term}: ¿Subida o caída?",
                "Cómo diversificar tu portafolio con {term}",
                "El futuro de {term} en el ecosistema cripto",
                "Riesgos y oportunidades de invertir en {term}"
            ],
            'health': [
                "Beneficios de {term} para tu salud",
                "Guía científica sobre {term}",
                "Cómo incorporar {term} en tu rutina diaria",
                "Mitos y verdades sobre {term}",
                "Estudios recientes sobre los efectos de {term}"
            ],
            'fitness': [
                "Rutina de 30 días con {term}",
                "Cómo maximizar resultados con {term}",
                "Guía completa de {term} para principiantes",
                "Combina {term} con otros ejercicios para mejores resultados",
                "Errores comunes al practicar {term}"
            ],
            'technology': [
                "Análisis completo: {term}",
                "Cómo {term} está transformando la industria",
                "Guía de compra: Lo mejor de {term} en 2023",
                "El futuro de {term}: Tendencias y predicciones",
                "Comparativa: {term} vs competidores"
            ],
            'ai': [
                "Cómo implementar {term} en tu negocio",
                "Guía para principiantes sobre {term}",
                "Casos de uso prácticos de {term}",
                "El impacto de {term} en la industria",
                "Tutorial paso a paso: Crea tu propio {term}"
            ],
            'gaming': [
                "Guía completa de {term}: Trucos y consejos",
                "Análisis de {term}: ¿Vale la pena?",
                "Cómo dominar {term} en tiempo récord",
                "Los mejores complementos para disfrutar {term}",
                "Comparativa: {term} vs alternativas populares"
            ],
            'entertainment': [
                "Todo lo que debes saber sobre {term}",
                "Análisis sin spoilers: {term}",
                "Las mejores escenas de {term}",
                "El fenómeno {term}: Por qué todos hablan de ello",
                "Teorías y predicciones sobre {term}"
            ],
            'education': [
                "Guía completa para aprender {term}",
                "Recursos gratuitos para dominar {term}",
                "Cómo enseñar {term} de forma efectiva",
                "Beneficios de aprender {term} en 2023",
                "Plan de estudio de 30 días para {term}"
            ],
            'lifestyle': [
                "Cómo incorporar {term} en tu rutina diaria",
                "Guía para principiantes sobre {term}",
                "Los beneficios de {term} para tu bienestar",
                "Tendencias de {term} para 2023",
                "Transforma tu vida con {term}: Guía paso a paso"
            ]
        }
        
        # Usar plantillas de la categoría o plantillas genéricas
        category_templates = templates.get(category, [
            "Guía completa sobre {term}",
            "Tendencias 2023: Todo sobre {term}",
            "Cómo aprovechar {term} al máximo",
            "Análisis detallado de {term}",
            "Lo que debes saber sobre {term}"
        ])
        
        # Generar ideas con términos populares
        for term in terms:
            # Seleccionar 2 plantillas aleatorias para cada término
            selected_templates = random.sample(category_templates, min(2, len(category_templates)))
            
            for template in selected_templates:
                idea = template.format(term=term)
                ideas.append(idea)
        
        # Agregar ideas genéricas de la categoría
        generic_ideas = [
            f"Las 10 tendencias de {category} para 2023",
            f"Guía completa sobre {category} para principiantes",
            f"Cómo mantenerse actualizado en {category}",
            f"Lo que los expertos no te cuentan sobre {category}"
        ]
        
        ideas.extend(generic_ideas)
        
        # Eliminar duplicados y limitar a 10 ideas
        unique_ideas = list(dict.fromkeys(ideas))
        
        return unique_ideas[:10]
    
    def evaluate_trend_performance(self) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de las predicciones de tendencias.
        
        Returns:
            Métricas de rendimiento
        """
        try:
            performance = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'categories': {}
            }
            
            # Obtener fechas con predicciones y datos reales
            prediction_dates = set(self.trend_data['predicted_trends'].keys())
            actual_dates = set(self.trend_data['historical_trends'].keys())
            
            # Encontrar fechas que tienen tanto predicciones como datos reales
            common_dates = prediction_dates.intersection(actual_dates)
            
            if not common_dates:
                return {'error': 'No hay suficientes datos para evaluar rendimiento'}
            
            # Calcular métricas por categoría
            all_errors = []
            
            for date in common_dates:
                predictions = self.trend_data['predicted_trends'][date]
                actuals = self.trend_data['historical_trends'][date]
                
                if 'predictions' not in predictions or 'trends' not in actuals:
                    continue
                
                for category in self.trend_categories:
                    if (category in predictions['predictions'] and 
                        category in actuals['trends'] and 
                        'google_trends' in actuals['trends'][category]):
                        
                        # Obtener valor predicho
                        predicted = predictions['predictions'][category].get('current_trend', 0)
                        
                        # Calcular valor real
                        actual = 0
                        google_data = actuals['trends'][category]['google_trends']
                        
                        # Usar datos de top queries si están disponibles
                        if 'top' in google_data and google_data['top']:
                            if isinstance(google_data['top'], list) and len(google_data['top']) > 0:
                                if 'value' in google_data['top'][0]:
                                    values = [item['value'] for item in google_data['top']]
                                    actual += sum(values) / len(values)
                        
                        # Usar datos de interest_over_time si están disponibles
                        if 'interest_over_time' in google_data and google_data['interest_over_time']:
                            if isinstance(google_data['interest_over_time'], list) and len(google_data['interest_over_time']) > 0:
                                if category in google_data['interest_over_time'][0]:
                                    values = [item[category] for item in google_data['interest_over_time']]
                                    actual += sum(values) / len(values)
                        
                        # Calcular error
                        if actual > 0:
                            error = abs(predicted - actual) / actual
                            all_errors.append(error)
                            
                            # Agregar a métricas por categoría
                            if category not in performance['categories']:
                                performance['categories'][category] = {
                                    'errors': [],
                                    'predictions': 0,
                                    'mape': 0,
                                    'accuracy': 0
                                }
                            
                            performance['categories'][category]['errors'].append(error)
                            performance['categories'][category]['predictions'] += 1
            
            # Calcular métricas globales
            if all_errors:
                performance['metrics']['mape'] = sum(all_errors) / len(all_errors)
                performance['metrics']['accuracy'] = 1 - performance['metrics']['mape']
                performance['metrics']['total_predictions'] = len(all_errors)
                
                # Calcular métricas por categoría
                for category, metrics in performance['categories'].items():
                    if metrics['errors']:
                        metrics['mape'] = sum(metrics['errors']) / len(metrics['errors'])
                        metrics['accuracy'] = 1 - metrics['mape']
                        # Eliminar lista de errores para reducir tamaño
                        del metrics['errors']
            
            # Guardar métricas
            date_key = datetime.now().strftime('%Y-%m-%d')
            self.trend_data['trend_performance'][date_key] = performance
            self._save_data()
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error al evaluar rendimiento: {str(e)}")
            return {'error': str(e)}
"""
Dashboard Manager - Sistema de visualización y monitoreo

Este módulo implementa un dashboard interactivo para monitorear el rendimiento,
costos, ingresos y métricas clave del sistema de creación y monetización de contenido.
Proporciona visualizaciones en tiempo real, alertas, y reportes personalizables.

Características principales:
- Visualización de KPIs y métricas de rendimiento
- Seguimiento de costos e ingresos por canal y plataforma
- Monitoreo de tendencias y predicciones
- Alertas configurables para eventos importantes
- Reportes automáticos y exportación de datos
- Integración con todos los subsistemas
"""

import os
import json
import time
import logging
import threading
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import requests
from io import BytesIO
import base64
import schedule
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard_manager")

# Constantes y configuración
DEFAULT_CONFIG_PATH = "config/dashboard_config.json"
DEFAULT_DB_PATH = "data/dashboard.db"
DEFAULT_REPORT_PATH = "reports"
REFRESH_INTERVAL = 60  # segundos
DEFAULT_PORT = 8050
DEFAULT_HOST = "0.0.0.0"
DEFAULT_DEBUG = False

# Colores para gráficos
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#bcbd22",
    "info": "#17becf",
    "light": "#7f7f7f",
    "dark": "#8c564b",
    "youtube": "#FF0000",
    "tiktok": "#000000",
    "instagram": "#C13584",
    "threads": "#000000",
    "bluesky": "#0085FF",
    "x": "#1DA1F2",
    "background": "#F8F9FA",
    "text": "#212529"
}

class DashboardManager:
    """
    Gestor principal del dashboard de monitoreo para el sistema de creación y monetización.
    
    Proporciona visualizaciones interactivas, alertas y reportes sobre el rendimiento
    del sistema, costos, ingresos y métricas clave.
    """
    
    def __init__(self, 
                config_path: str = DEFAULT_CONFIG_PATH,
                db_path: str = DEFAULT_DB_PATH,
                report_path: str = DEFAULT_REPORT_PATH,
                auto_start: bool = False,
                port: int = DEFAULT_PORT,
                host: str = DEFAULT_HOST,
                debug: bool = DEFAULT_DEBUG):
        """
        Inicializa el gestor del dashboard.
        
        Args:
            config_path: Ruta al archivo de configuración
            db_path: Ruta a la base de datos
            report_path: Ruta para guardar reportes
            auto_start: Si es True, inicia automáticamente el dashboard
            port: Puerto para el servidor web
            host: Host para el servidor web
            debug: Modo debug para el servidor
        """
        self.config_path = config_path
        self.db_path = db_path
        self.report_path = report_path
        self.port = port
        self.host = host
        self.debug = debug
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar base de datos
        self._init_database()
        
        # Crear directorios necesarios
        os.makedirs(self.report_path, exist_ok=True)
        
        # Inicializar aplicación Dash
        self.server = Flask(__name__)
        self.app = dash.Dash(
            __name__,
            server=self.server,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Configurar layout
        self._setup_layout()
        
        # Configurar callbacks
        self._setup_callbacks()
        
        # Inicializar hilos y flags
        self.running = False
        self.data_thread = None
        self.report_thread = None
        self.alert_thread = None
        
        # Iniciar si auto_start es True
        if auto_start:
            self.start()
            
        logger.info(f"DashboardManager inicializado. Puerto: {self.port}, Host: {self.host}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración del dashboard."""
        default_config = {
            "refresh_interval": REFRESH_INTERVAL,
            "email_notifications": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "alerts": {
                "shadowban_detected": True,
                "niche_saturation": True,
                "low_engagement": True,
                "high_costs": True,
                "revenue_milestone": True
            },
            "report_schedule": {
                "daily": True,
                "weekly": True,
                "monthly": True,
                "time": "23:00"
            },
            "data_sources": {
                "analytics_engine": "http://localhost:8000/api/analytics",
                "monetization_tracker": "http://localhost:8000/api/monetization",
                "batch_processor": "http://localhost:8000/api/batch",
                "trend_radar": "http://localhost:8000/api/trends"
            },
            "kpis": {
                "engagement": [
                    "subscriptions_post_cta",
                    "scroll_post_cta",
                    "engagement_deferred",
                    "comment_sentiment",
                    "retention_30d"
                ],
                "monetization": [
                    "rpm",
                    "affiliate_conversion",
                    "b2b_revenue",
                    "roi_per_channel"
                ],
                "operational": [
                    "time_to_cta",
                    "cta_reputation",
                    "cost_per_video",
                    "cache_savings",
                    "alerts_sent"
                ]
            },
            "channels": [
                {
                    "name": "Finanzas",
                    "platforms": ["youtube", "tiktok"],
                    "description": "Cripto y ahorro",
                    "color": "#1f77b4"
                },
                {
                    "name": "Salud",
                    "platforms": ["tiktok", "instagram"],
                    "description": "Fitness",
                    "color": "#2ca02c"
                },
                {
                    "name": "Gaming",
                    "platforms": ["youtube", "tiktok"],
                    "description": "Estrategias",
                    "color": "#ff7f0e"
                },
                {
                    "name": "Tecnología",
                    "platforms": ["youtube", "instagram"],
                    "description": "Gadgets",
                    "color": "#d62728"
                },
                {
                    "name": "Humor",
                    "platforms": ["tiktok", "instagram"],
                    "description": "Memes",
                    "color": "#9467bd"
                }
            ],
            "platforms": [
                {
                    "name": "youtube",
                    "display_name": "YouTube",
                    "color": "#FF0000"
                },
                {
                    "name": "tiktok",
                    "display_name": "TikTok",
                    "color": "#000000"
                },
                {
                    "name": "instagram",
                    "display_name": "Instagram",
                    "color": "#C13584"
                },
                {
                    "name": "threads",
                    "display_name": "Threads",
                    "color": "#000000"
                },
                {
                    "name": "bluesky",
                    "display_name": "Bluesky",
                    "color": "#0085FF"
                },
                {
                    "name": "x",
                    "display_name": "X",
                    "color": "#1DA1F2"
                }
            ]
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Combinar con configuración predeterminada
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            else:
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                # Guardar configuración predeterminada
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return default_config
    
    def _init_database(self) -> None:
        """Inicializa la base de datos para el dashboard."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Conectar a la base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crear tablas necesarias
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS engagement_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel TEXT,
                platform TEXT,
                video_id TEXT,
                subscriptions_post_cta REAL,
                scroll_post_cta REAL,
                engagement_deferred REAL,
                comment_sentiment REAL,
                retention_30d REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS monetization_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel TEXT,
                platform TEXT,
                video_id TEXT,
                rpm REAL,
                affiliate_conversion REAL,
                b2b_revenue REAL,
                roi REAL,
                total_revenue REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS operational_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel TEXT,
                platform TEXT,
                video_id TEXT,
                time_to_cta REAL,
                cta_reputation REAL,
                cost_per_video REAL,
                cache_savings REAL,
                processing_time REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT,
                channel TEXT,
                platform TEXT,
                video_id TEXT,
                message TEXT,
                severity TEXT,
                acknowledged BOOLEAN DEFAULT 0
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                subcategory TEXT,
                amount REAL,
                description TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel TEXT,
                platform TEXT,
                source TEXT,
                amount REAL,
                description TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trend TEXT,
                category TEXT,
                score REAL,
                source TEXT,
                used BOOLEAN DEFAULT 0
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel TEXT,
                platform TEXT,
                video_id TEXT,
                title TEXT,
                views INTEGER,
                likes INTEGER,
                comments INTEGER,
                shares INTEGER,
                ctr REAL,
                watch_time REAL
            )
            ''')
            
            # Guardar cambios
            conn.commit()
            conn.close()
            
            logger.info("Base de datos inicializada correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {str(e)}")
    
    def _setup_layout(self) -> None:
        """Configura el layout del dashboard."""
        self.app.layout = html.Div([
            # Barra de navegación
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("Content Bot Dashboard", className="ms-2"), width="auto"),
                    ], align="center", className="g-0"),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Resumen", href="#", id="nav-overview")),
                            dbc.NavItem(dbc.NavLink("Engagement", href="#", id="nav-engagement")),
                            dbc.NavItem(dbc.NavLink("Monetización", href="#", id="nav-monetization")),
                            dbc.NavItem(dbc.NavLink("Operaciones", href="#", id="nav-operations")),
                            dbc.NavItem(dbc.NavLink("Tendencias", href="#", id="nav-trends")),
                            dbc.NavItem(dbc.NavLink("Alertas", href="#", id="nav-alerts")),
                            dbc.NavItem(dbc.NavLink("Reportes", href="#", id="nav-reports")),
                            dbc.NavItem(dbc.NavLink("Configuración", href="#", id="nav-settings")),
                        ], className="ms-auto", navbar=True),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ]),
                color="primary",
                dark=True,
                className="mb-4",
            ),
            
            # Contenedor principal
            dbc.Container([
                # Filtros globales
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Período"),
                                        dcc.Dropdown(
                                            id="filter-period",
                                            options=[
                                                {"label": "Hoy", "value": "today"},
                                                {"label": "Ayer", "value": "yesterday"},
                                                {"label": "Últimos 7 días", "value": "7days"},
                                                {"label": "Últimos 30 días", "value": "30days"},
                                                {"label": "Este mes", "value": "this_month"},
                                                {"label": "Mes pasado", "value": "last_month"},
                                                {"label": "Personalizado", "value": "custom"}
                                            ],
                                            value="7days",
                                            clearable=False
                                        ),
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Canal"),
                                        dcc.Dropdown(
                                            id="filter-channel",
                                            options=[
                                                {"label": "Todos", "value": "all"}
                                            ] + [
                                                {"label": channel["name"], "value": channel["name"]}
                                                for channel in self.config["channels"]
                                            ],
                                            value="all",
                                            clearable=False
                                        ),
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Plataforma"),
                                        dcc.Dropdown(
                                            id="filter-platform",
                                            options=[
                                                {"label": "Todas", "value": "all"}
                                            ] + [
                                                {"label": platform["display_name"], "value": platform["name"]}
                                                for platform in self.config["platforms"]
                                            ],
                                            value="all",
                                            clearable=False
                                        ),
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Actualizar"),
                                        html.Div([
                                            dbc.Button("Actualizar", id="btn-refresh", color="primary", className="w-100"),
                                        ]),
                                    ], width=3),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id="date-range-container", style={"display": "none"}, children=[
                                            dcc.DatePickerRange(
                                                id="filter-date-range",
                                                start_date=datetime.datetime.now() - datetime.timedelta(days=7),
                                                end_date=datetime.datetime.now(),
                                                display_format="DD/MM/YYYY"
                                            )
                                        ])
                                    ], width=12)
                                ], className="mt-2")
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),
                
                # Contenido principal (cambia según la navegación)
                html.Div(id="page-content", children=[
                    # Página de resumen (default)
                    self._create_overview_page()
                ])
            ], fluid=True)
        ])
    
    def _create_overview_page(self) -> html.Div:
        """Crea la página de resumen del dashboard."""
        return html.Div([
            # Tarjetas de KPIs principales
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Ingresos Totales", className="card-title"),
                            html.H3(id="kpi-total-revenue", children="$0.00", className="card-text text-primary"),
                            html.P(id="kpi-revenue-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Costos Totales", className="card-title"),
                            html.H3(id="kpi-total-costs", children="$0.00", className="card-text text-danger"),
                            html.P(id="kpi-costs-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("ROI Promedio", className="card-title"),
                            html.H3(id="kpi-avg-roi", children="0%", className="card-text text-success"),
                            html.P(id="kpi-roi-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Videos Publicados", className="card-title"),
                            html.H3(id="kpi-videos-published", children="0", className="card-text text-info"),
                            html.P(id="kpi-videos-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            # Gráficos principales
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ingresos vs Costos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-costs")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rendimiento por Canal"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-channel-performance")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Métricas de Engagement"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-engagement-metrics")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Fuentes de Ingresos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-sources")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            # Alertas recientes y tendencias
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Alertas Recientes", className="me-auto"),
                            dbc.Button("Ver todas", id="btn-view-all-alerts", color="link", size="sm")
                        ], className="d-flex align-items-center"),
                        dbc.CardBody([
                            html.Div(id="recent-alerts", children=[
                                html.P("No hay alertas recientes", className="text-muted")
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Tendencias Actuales", className="me-auto"),
                            dbc.Button("Ver todas", id="btn-view-all-trends", color="link", size="sm")
                        ], className="d-flex align-items-center"),
                        dbc.CardBody([
                            html.Div(id="current-trends", children=[
                                html.P("No hay tendencias disponibles", className="text-muted")
                            ])
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            # Videos recientes
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Videos Recientes", className="me-auto"),
                            dbc.Button("Ver todos", id="btn-view-all-videos", color="link", size="sm")
                        ], className="d-flex align-items-center"),
                        dbc.CardBody([
                            html.Div(id="recent-videos", children=[
                                html.P("No hay videos recientes", className="text-muted")
                            ])
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
        ])
    
    def _create_engagement_page(self) -> html.Div:
        """Crea la página de métricas de engagement."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Suscripciones post-CTA", className="card-title"),
                            html.H3(id="kpi-subscriptions-post-cta", children="0%", className="card-text text-primary"),
                            html.P(id="kpi-subscriptions-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Abandono post-CTA", className="card-title"),
                            html.H3(id="kpi-scroll-post-cta", children="0%", className="card-text text-danger"),
                            html.P(id="kpi-scroll-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Engagement diferido", className="card-title"),
                            html.H3(id="kpi-engagement-deferred", children="0%", className="card-text text-success"),
                            html.P(id="kpi-engagement-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Retención a 30 días", className="card-title"),
                            html.H3(id="kpi-retention-30d", children="0%", className="card-text text-info"),
                            html.P(id="kpi-retention-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Evolución de Engagement"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-engagement-evolution")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Engagement por Canal"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-engagement-by-channel")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Engagement por Plataforma"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-engagement-by-platform")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sentimiento de Comentarios"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-comment-sentiment")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Análisis de Cohortes"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-cohort-analysis")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Videos con Mayor Engagement"),
                        dbc.CardBody([
                            html.Div(id="top-engagement-videos")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
        ])
    
    def _create_monetization_page(self) -> html.Div:
        """Crea la página de métricas de monetización."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("RPM Promedio", className="card-title"),
                            html.H3(id="kpi-avg-rpm", children="$0.00", className="card-text text-primary"),
                            html.P(id="kpi-rpm-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Conversión Afiliados", className="card-title"),
                            html.H3(id="kpi-affiliate-conversion", children="0%", className="card-text text-success"),
                            html.P(id="kpi-affiliate-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Ingresos B2B", className="card-title"),
                            html.H3(id="kpi-b2b-revenue", children="$0.00", className="card-text text-info"),
                            html.P(id="kpi-b2b-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("ROI Promedio", className="card-title"),
                            html.H3(id="kpi-monetization-roi", children="0%", className="card-text text-warning"),
                            html.P(id="kpi-monetization-roi-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Evolución de Ingresos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-evolution")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
                        dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ingresos por Canal"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-by-channel")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ingresos por Plataforma"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-by-platform")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Distribución de Fuentes de Ingresos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-distribution")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Tendencia de ROI"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-roi-trend")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Videos con Mayor Monetización"),
                        dbc.CardBody([
                            html.Div(id="top-monetization-videos")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predicción de Ingresos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-revenue-prediction")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
        ])
    
    def _create_operations_page(self) -> html.Div:
        """Crea la página de métricas operacionales."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Tiempo a CTA", className="card-title"),
                            html.H3(id="kpi-time-to-cta", children="0s", className="card-text text-primary"),
                            html.P(id="kpi-time-to-cta-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Reputación CTA", className="card-title"),
                            html.H3(id="kpi-cta-reputation", children="0%", className="card-text text-success"),
                            html.P(id="kpi-cta-reputation-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Costo por Video", className="card-title"),
                            html.H3(id="kpi-cost-per-video", children="$0.00", className="card-text text-danger"),
                            html.P(id="kpi-cost-per-video-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Ahorro de Caché", className="card-title"),
                            html.H3(id="kpi-cache-savings", children="$0.00", className="card-text text-info"),
                            html.P(id="kpi-cache-savings-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Evolución de Costos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-costs-evolution")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Costos por Categoría"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-costs-by-category")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Tiempo de Procesamiento"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-processing-time")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Eficiencia por Canal"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-channel-efficiency")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Uso de Recursos"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-resource-usage")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historial de Costos"),
                        dbc.CardBody([
                            html.Div(id="costs-history")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
        ])
    
    def _create_trends_page(self) -> html.Div:
        """Crea la página de análisis de tendencias."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Tendencias Detectadas", className="card-title"),
                            html.H3(id="kpi-trends-detected", children="0", className="card-text text-primary"),
                            html.P(id="kpi-trends-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Tendencias Utilizadas", className="card-title"),
                            html.H3(id="kpi-trends-used", children="0", className="card-text text-success"),
                            html.P(id="kpi-trends-used-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Tasa de Conversión", className="card-title"),
                            html.H3(id="kpi-trend-conversion", children="0%", className="card-text text-info"),
                            html.P(id="kpi-trend-conversion-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Tiempo de Detección", className="card-title"),
                            html.H3(id="kpi-trend-detection-time", children="0h", className="card-text text-warning"),
                            html.P(id="kpi-trend-detection-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Tendencias Actuales"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-current-trends")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Tendencias por Categoría"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-trends-by-category")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rendimiento de Tendencias"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-trend-performance")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predicción de Tendencias"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-trend-prediction")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historial de Tendencias"),
                        dbc.CardBody([
                            html.Div(id="trends-history")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
        ])
    
    def _create_alerts_page(self) -> html.Div:
        """Crea la página de alertas del sistema."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Alertas Activas", className="card-title"),
                            html.H3(id="kpi-active-alerts", children="0", className="card-text text-danger"),
                            html.P(id="kpi-alerts-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Alertas Críticas", className="card-title"),
                            html.H3(id="kpi-critical-alerts", children="0", className="card-text text-warning"),
                            html.P(id="kpi-critical-alerts-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Tiempo de Resolución", className="card-title"),
                            html.H3(id="kpi-resolution-time", children="0h", className="card-text text-primary"),
                            html.P(id="kpi-resolution-time-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Alertas Resueltas", className="card-title"),
                            html.H3(id="kpi-resolved-alerts", children="0", className="card-text text-success"),
                            html.P(id="kpi-resolved-alerts-change", children="0% vs período anterior", className="card-text text-muted")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Filtros de Alertas", className="me-auto"),
                        ], className="d-flex align-items-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Severidad"),
                                    dcc.Dropdown(
                                        id="filter-alert-severity",
                                        options=[
                                            {"label": "Todas", "value": "all"},
                                            {"label": "Crítica", "value": "critical"},
                                            {"label": "Alta", "value": "high"},
                                            {"label": "Media", "value": "medium"},
                                            {"label": "Baja", "value": "low"},
                                            {"label": "Informativa", "value": "info"}
                                        ],
                                        value="all",
                                        clearable=False
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Tipo"),
                                    dcc.Dropdown(
                                        id="filter-alert-type",
                                        options=[
                                            {"label": "Todos", "value": "all"},
                                            {"label": "Shadowban", "value": "shadowban_detected"},
                                            {"label": "Saturación", "value": "niche_saturation"},
                                            {"label": "Engagement Bajo", "value": "low_engagement"},
                                            {"label": "Costos Altos", "value": "high_costs"},
                                            {"label": "Hito de Ingresos", "value": "revenue_milestone"}
                                        ],
                                        value="all",
                                        clearable=False
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Estado"),
                                    dcc.Dropdown(
                                        id="filter-alert-status",
                                        options=[
                                            {"label": "Todos", "value": "all"},
                                            {"label": "Activas", "value": "active"},
                                            {"label": "Reconocidas", "value": "acknowledged"},
                                            {"label": "Resueltas", "value": "resolved"}
                                        ],
                                        value="all",
                                        clearable=False
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Acciones"),
                                    html.Div([
                                        dbc.Button("Reconocer Seleccionadas", id="btn-acknowledge-alerts", color="primary", className="w-100"),
                                    ]),
                                ], width=3),
                            ]),
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Listado de Alertas"),
                        dbc.CardBody([
                            html.Div(id="alerts-table")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Distribución de Alertas"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-alerts-distribution")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Tendencia de Alertas"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-alerts-trend")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
        ])
    
    def _create_reports_page(self) -> html.Div:
        """Crea la página de reportes del sistema."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Generar Reporte"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Tipo de Reporte"),
                                    dcc.Dropdown(
                                        id="report-type",
                                        options=[
                                            {"label": "Reporte Completo", "value": "full"},
                                            {"label": "Reporte de Engagement", "value": "engagement"},
                                            {"label": "Reporte de Monetización", "value": "monetization"},
                                            {"label": "Reporte de Operaciones", "value": "operations"},
                                            {"label": "Reporte de Tendencias", "value": "trends"},
                                            {"label": "Reporte de Alertas", "value": "alerts"}
                                        ],
                                        value="full",
                                        clearable=False
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Formato"),
                                    dcc.Dropdown(
                                        id="report-format",
                                        options=[
                                            {"label": "PDF", "value": "pdf"},
                                            {"label": "Excel", "value": "excel"},
                                            {"label": "CSV", "value": "csv"},
                                            {"label": "JSON", "value": "json"}
                                        ],
                                        value="pdf",
                                        clearable=False
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Acción"),
                                    html.Div([
                                        dbc.Button("Generar Reporte", id="btn-generate-report", color="primary", className="w-100"),
                                    ]),
                                ], width=4),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="report-status", className="mt-3")
                                ], width=12)
                            ])
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Reportes Programados"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Checkbox(
                                            id="schedule-daily",
                                            className="form-check-input",
                                            checked=True
                                        ),
                                        dbc.Label(
                                            "Reporte Diario",
                                            html_for="schedule-daily",
                                            className="form-check-label"
                                        ),
                                    ], check=True),
                                ], width=4),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Checkbox(
                                            id="schedule-weekly",
                                            className="form-check-input",
                                            checked=True
                                        ),
                                        dbc.Label(
                                            "Reporte Semanal",
                                            html_for="schedule-weekly",
                                            className="form-check-label"
                                        ),
                                    ], check=True),
                                ], width=4),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Checkbox(
                                            id="schedule-monthly",
                                            className="form-check-input",
                                            checked=True
                                        ),
                                        dbc.Label(
                                            "Reporte Mensual",
                                            html_for="schedule-monthly",
                                            className="form-check-label"
                                        ),
                                    ], check=True),
                                ], width=4),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Hora de envío"),
                                    dbc.Input(
                                        id="schedule-time",
                                        type="time",
                                        value="23:00"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Acción"),
                                    html.Div([
                                        dbc.Button("Guardar Configuración", id="btn-save-schedule", color="primary", className="w-100"),
                                    ]),
                                ], width=6),
                            ], className="mt-3"),
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historial de Reportes"),
                        dbc.CardBody([
                            html.Div(id="reports-history")
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
        ])
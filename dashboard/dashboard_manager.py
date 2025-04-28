```python
"""
Dashboard Manager - Sistema de visualización y monitoreo

Este módulo implementa un dashboard interactivo para monitorear el rendimiento,
costos, ingresos y métricas clave del sistema de creación y monetización de contenido.
Proporciona visualizaciones en tiempo real, alertas, reportes personalizables y monitoreo de publicaciones.

Características principales:
- Visualización de KPIs y métricas de rendimiento
- Seguimiento de costos e ingresos por canal y plataforma
- Monitoreo de tendencias y predicciones
- Alertas configurables para eventos importantes
- Reportes automáticos y exportación de datos
- Integración con todos los subsistemas
- Monitoreo de publicaciones pendientes/pausadas con gráficos y reportes
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
MAX_RETRIES = 3
RETRY_DELAY = 5
TASKS_PER_PAGE = 10  # Para paginación

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
    
    Proporciona visualizaciones interactivas, alertas, reportes y monitoreo de publicaciones.
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
        self.components = {}
        self.task_cache = {"pending": {}, "paused": {}, "last_updated": None}
        
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
                "tasks": True,  # Nuevo: reportes de tareas
                "time": "23:00"
            },
            "data_sources": {
                "analytics_engine": "http://localhost:8000/api/analytics",
                "monetization_tracker": "http://localhost:8000/api/monetization",
                "batch_processor": "http://localhost:8000/api/batch",
                "trend_radar": "http://localhost:8000/api/trends",
                "orchestrator": "http://localhost:8000/api/orchestrator"
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
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            else:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return default_config
    
    def _init_database(self) -> None:
        """Inicializa la base de datos para el dashboard."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
            
            conn.commit()
            conn.close()
            logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {str(e)}")
    
    def _load_component(self, component_name: str) -> Any:
        """Carga dinámicamente un componente con reintentos."""
        if component_name in self.components:
            return self.components[component_name]
        
        for attempt in range(MAX_RETRIES):
            try:
                if component_name == 'orchestrator':
                    from orchestrator import Orchestrator
                    self.components[component_name] = Orchestrator()
                elif component_name == 'analytics_engine':
                    from analytics_engine import AnalyticsEngine
                    self.components[component_name] = AnalyticsEngine()
                elif component_name == 'trend_radar':
                    from trend_radar import TrendRadar
                    self.components[component_name] = TrendRadar()
                logger.info(f"Componente {component_name} cargado correctamente")
                return self.components[component_name]
            except ImportError as e:
                logger.error(f"Intento {attempt + 1}/{MAX_RETRIES} - Error al cargar {component_name}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"No se pudo cargar {component_name} tras {MAX_RETRIES} intentos")
                    return None
    
    def _setup_layout(self) -> None:
        """Configura el layout del dashboard."""
        # Tarjeta de publicaciones pendientes/pausadas
        pending_publications_card = dbc.Card(
            dbc.CardBody([
                html.H4("Publicaciones Pendientes y Pausadas", className="card-title"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Filtrar por:"),
                        dcc.Input(
                            id="pending-tasks-search",
                            type="text",
                            placeholder="Buscar por canal, tipo o estado...",
                            className="form-control mb-2"
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Canal:"),
                        dcc.Dropdown(
                            id="filter-task-channel",
                            options=[
                                {"label": "Todos", "value": "all"}
                            ] + [
                                {"label": channel["name"], "value": channel["name"]}
                                for channel in self.config["channels"]
                            ],
                            value="all",
                            clearable=False,
                            className="mb-2"
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Acciones:"),
                        html.Div([
                            dbc.Button(
                                "Exportar a CSV",
                                id="btn-export-tasks",
                                color="secondary",
                                className="w-100 mb-2"
                            ),
                        ]),
                    ], width=4),
                ]),
                html.Div(id="pending-publications-content", className="mt-3"),
                dcc.Interval(
                    id='pending-publications-interval',
                    interval=REFRESH_INTERVAL * 1000,
                    n_intervals=0
                ),
                # Paginación
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="tasks-page",
                            options=[{"label": f"Página {i+1}", "value": i} for i in range(10)],
                            value=0,
                            clearable=False,
                            style={"width": "150px"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.P(id="tasks-page-info", className="text-muted"),
                    ], width=9),
                ], className="mt-3"),
                # Modal para detalles de tarea
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Detalles de la Tarea")),
                    dbc.ModalBody(id="task-details-content"),
                    dbc.ModalFooter(
                        dbc.Button("Cerrar", id="close-task-details", className="ms-auto")
                    ),
                ], id="task-details-modal", size="lg"),
            ]),
            className="mb-4",
            style={"border-left": f"5px solid {COLORS['primary']}"}
        )
        
        # Definir tabs
        tabs = dbc.Tabs([
            dbc.Tab(
                self._create_overview_page(),
                label="Resumen",
                tab_id="overview",
                tab_style={"margin-left": "0"}
            ),
            dbc.Tab(
                self._create_engagement_page(),
                label="Engagement",
                tab_id="engagement"
            ),
            dbc.Tab(
                self._create_monetization_page(),
                label="Monetización",
                tab_id="monetization"
            ),
            dbc.Tab(
                self._create_operations_page(),
                label="Operaciones",
                tab_id="operations"
            ),
            dbc.Tab(
                self._create_trends_page(),
                label="Tendencias",
                tab_id="trends"
            ),
            dbc.Tab(
                self._create_alerts_page(),
                label="Alertas",
                tab_id="alerts"
            ),
            dbc.Tab(
                [
                    pending_publications_card,
                    # Gráfico de distribución de tareas
                    dbc.Card([
                        dbc.CardHeader("Distribución de Tareas"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-tasks-distribution")
                        ])
                    ], className="mb-4"),
                    # Gráfico de tendencia de procesamiento
                    dbc.Card([
                        dbc.CardHeader("Tiempo de Procesamiento por Canal"),
                        dbc.CardBody([
                            dcc.Graph(id="graph-tasks-processing-time")
                        ])
                    ], className="mb-4"),
                    # Alertas recientes
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
                    ], className="mb-4"),
                ],
                label="Monitoreo",
                tab_id="monitoring"
            ),
            dbc.Tab(
                self._create_reports_page(),
                label="Reportes",
                tab_id="reports"
            ),
        ], id="main-tabs", active_tab="overview")
        
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
                            dbc.NavItem(dbc.NavLink("Resumen", id="nav-overview")),
                            dbc.NavItem(dbc.NavLink("Engagement", id="nav-engagement")),
                            dbc.NavItem(dbc.NavLink("Monetización", id="nav-monetization")),
                            dbc.NavItem(dbc.NavLink("Operaciones", id="nav-operations")),
                            dbc.NavItem(dbc.NavLink("Tendencias", id="nav-trends")),
                            dbc.NavItem(dbc.NavLink("Alertas", id="nav-alerts")),
                            dbc.NavItem(dbc.NavLink("Monitoreo", id="nav-monitoring")),
                            dbc.NavItem(dbc.NavLink("Reportes", id="nav-reports")),
                            dbc.NavItem(dbc.NavLink("Configuración", id="nav-settings")),
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
                
                # Contenido principal
                tabs
            ], fluid=True)
        ])
    
    def _create_tasks_table(self, tasks: Dict, title: str, search_query: str = "", 
                          channel: str = "all", page: int = 0) -> html.Div:
        """Crea una tabla HTML para mostrar las tareas con paginación."""
        if not tasks:
            return html.Div([
                html.H5(f"Publicaciones {title}"),
                html.P("No hay publicaciones en esta categoría", className="text-muted")
            ])
        
        # Filtrar tareas
        filtered_tasks = tasks
        if search_query:
            search_query = search_query.lower()
            filtered_tasks = {
                task_id: task for task_id, task in filtered_tasks.items()
                if (search_query in task_id.lower() or
                    search_query in task.get('type', '').lower() or
                    search_query in task.get('status', '').lower() or
                    search_query in task.get('channel_id', '').lower() or
                    search_query in task.get('platform', '').lower())
            }
        
        if channel != "all":
            filtered_tasks = {
                task_id: task for task_id, task in filtered_tasks.items()
                if task.get('channel_id', '') == channel
            }
        
        if not filtered_tasks:
            return html.Div([
                html.H5(f"Publicaciones {title}"),
                html.P("No se encontraron tareas", className="text-muted")
            ])
        
        # Paginación
        total_tasks = len(filtered_tasks)
        total_pages = (total_tasks + TASKS_PER_PAGE - 1) // TASKS_PER_PAGE
        start_idx = page * TASKS_PER_PAGE
        end_idx = min(start_idx + TASKS_PER_PAGE, total_tasks)
        task_items = list(filtered_tasks.items())[start_idx:end_idx]
        
        # Crear filas
        rows = []
        analytics = self._load_component('analytics_engine')
        trend_radar = self._load_component('trend_radar')
        current_trends = trend_radar.current_trends if trend_radar else []
        
        for task_id, task in task_items:
            task_type = task.get('type', 'Desconocido')
            status = task.get('status', 'Desconocido')
            channel_id = task.get('channel_id', 'Desconocido')
            platform = task.get('platform', 'Todas')
            created_at = task.get('created_at', 'Desconocido')
            paused_reason = task.get('paused_reason', '')
            
            if isinstance(created_at, str):
                try:
                    created_at = datetime.datetime.fromisoformat(created_at)
                    created_at = created_at.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            warning_icon = ""
            if "shadowban" in paused_reason.lower():
                warning_icon = html.Span(
                    "⚠️",
                    title=paused_reason,
                    style={"color": COLORS['warning'], "cursor": "pointer"}
                )
            
            # Métricas proyectadas
            engagement = "N/A"
            if analytics and task.get('content_id'):
                metrics = analytics.get_content_metrics(task['content_id'])
                engagement = f"{metrics.get('projected_engagement', 0):.1f}%"
            
            # Tendencias relacionadas
            trend_icon = ""
            if any(trend in task.get('content_tags', []) for trend in current_trends):
                trend_icon = html.Span(
                    "🔥",
                    title="Aprovecha tendencia actual",
                    style={"color": COLORS['success'], "cursor": "pointer"}
                )
            
            # Botones de acción
            action_buttons = [
                dbc.Button(
                    "Detalles",
                    id={"type": "task-details", "index": task_id},
                    color="info",
                    size="sm",
                    className="me-1"
                )
            ]
            if title.lower() == "pausadas":
                action_buttons.append(
                    dbc.Button(
                        "Reanudar",
                        id={"type": "resume-task", "index": task_id},
                        color="success",
                        size="sm",
                        className="me-1"
                    )
                )
            action_buttons.append(
                dbc.Button(
                    "Cancelar",
                    id={"type": "cancel-task", "index": task_id},
                    color="danger",
                    size="sm"
                )
            )
            
            rows.append(html.Tr([
                html.Td(html.A(task_id[:8] + "...", href="#", id={"type": "task-details", "index": task_id})),
                html.Td(task_type),
                html.Td([status, " ", warning_icon]),
                html.Td(channel_id),
                html.Td(platform),
                html.Td(created_at),
                html.Td(engagement),
                html.Td(trend_icon),
                html.Td(action_buttons)
            ]))
        
        table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("ID"),
                        html.Th("Tipo"),
                        html.Th("Estado"),
                        html.Th("Canal"),
                        html.Th("Plataforma"),
                        html.Th("Creado"),
                        html.Th("Engagement Proyectado"),
                        html.Th("Tendencia"),
                        html.Th("Acciones")
                    ])
                ),
                html.Tbody(rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            size="sm",
            style={"font-size": "0.9rem"}
        )
        
        return html.Div([
            html.H5(f"Publicaciones {title} ({total_tasks} encontradas, mostrando {start_idx+1}-{end_idx})"),
            table
        ])
    
    def _export_tasks_to_csv(self, tasks: Dict, title: str) -> str:
        """Exporta tareas a un archivo CSV."""
        if not tasks:
            return ""
        
        df = pd.DataFrame([
            {
                "ID": task_id,
                "Tipo": task.get('type', 'Desconocido'),
                "Estado": task.get('status', 'Desconocido'),
                "Canal": task.get('channel_id', 'Desconocido'),
                "Plataforma": task.get('platform', 'Todas'),
                "Creado": task.get('created_at', 'Desconocido'),
                "Razón_Pausa": task.get('paused_reason', ''),
                "Content_ID": task.get('content_id', ''),
                "Tags": ", ".join(task.get('content_tags', []))
            }
            for task_id, task in tasks.items()
        ])
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.report_path}/tasks_{title.lower()}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename
    
    def _generate_task_report(self, format: str = "csv") -> str:
        """Genera un reporte de tareas pendientes y pausadas."""
        orchestrator = self._load_component('orchestrator')
        if not orchestrator:
            return ""
        
        pending_tasks = orchestrator.current_tasks
        paused_tasks = orchestrator.paused_tasks
        
        df = pd.DataFrame([
            {
                "ID": task_id,
                "Tipo": task.get('type', 'Desconocido'),
                "Estado": task.get('status', 'Desconocido'),
                "Canal": task.get('channel_id', 'Desconocido'),
                "Plataforma": task.get('platform', 'Todas'),
                "Creado": task.get('created_at', 'Desconocido'),
                "Razón_Pausa": task.get('paused_reason', ''),
                "Categoría": "Pendiente" if task_id in pending_tasks else "Pausada"
            }
            for task_id, task in {**pending_tasks, **paused_tasks}.items()
        ])
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.report_path}/task_report_{timestamp}.{format}"
        if format == "csv":
            df.to_csv(filename, index=False)
        elif format == "json":
            df.to_json(filename, orient="records", indent=4)
        return filename
    
    def _create_tasks_distribution_graph(self, pending_tasks: Dict, paused_tasks: Dict) -> go.Figure:
        """Crea un gráfico de distribución de tareas."""
        counts = {
            "Pendientes": len(pending_tasks),
            "Pausadas": len(paused_tasks),
            "Completadas": len(self._load_component('orchestrator').task_history)
        }
        fig = go.Figure(data=[
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                marker_colors=[COLORS['primary'], COLORS['warning'], COLORS['success']]
            )
        ])
        fig.update_layout(
            title="Distribución de Tareas",
            showlegend=True,
            paper_bgcolor=COLORS['background']
        )
        return fig
    
    def _create_tasks_processing_time_graph(self, tasks: Dict) -> go.Figure:
        """Crea un gráfico de tiempo de procesamiento por canal."""
        data = []
        for task in tasks.values():
            if task.get('processing_time') and task.get('channel_id'):
                data.append({
                    "Channel": task['channel_id'],
                    "ProcessingTime": task['processing_time']
                })
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        fig = px.box(
            df,
            x="Channel",
            y="ProcessingTime",
            color="Channel",
            color_discrete_map={ch['name']: ch['color'] for ch in self.config['channels']}
        )
        fig.update_layout(
            title="Tiempo de Procesamiento por Canal",
            yaxis_title="Tiempo (segundos)",
            showlegend=False,
            paper_bgcolor=COLORS['background']
        )
        return fig
    
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
                                            {"label": "Reporte de Alertas", "value": "alerts"},
                                            {"label": "Reporte de Tareas", "value": "tasks"}
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
                                        value="csv",
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
                                ], width=3),
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
                                ], width=3),
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
                                ], width=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Checkbox(
                                            id="schedule-tasks",
                                            className="form-check-input",
                                            checked=True
                                        ),
                                        dbc.Label(
                                            "Reporte de Tareas",
                                            html_for="schedule-tasks",
                                            className="form-check-label"
                                        ),
                                    ], check=True),
                                ], width=3),
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
    
    def _setup_callbacks(self) -> None:
        """Configura los callbacks del dashboard."""
        @self.app.callback(
            [
                Output("pending-publications-content", "children"),
                Output("tasks-page", "options"),
                Output("tasks-page-info", "children"),
                Output("graph-tasks-distribution", "figure"),
                Output("graph-tasks-processing-time", "figure")
            ],
            [
                Input("pending-publications-interval", "n_intervals"),
                Input("pending-tasks-search", "value"),
                Input("filter-task-channel", "value"),
                Input("tasks-page", "value")
            ]
        )
        def update_pending_publications(n, search_query, channel, page):
            """Actualiza la lista de publicaciones pendientes y pausadas."""
            try:
                orchestrator = self._load_component('orchestrator')
                if not orchestrator:
                    return [
                        html.Div("No se pudo cargar el orquestador", className="text-danger"),
                        [], "", go.Figure(), go.Figure()
                    ]
                
                # Usar cache si está reciente
                now = datetime.datetime.now()
                if (self.task_cache["last_updated"] and
                    (now - self.task_cache["last_updated"]).total_seconds() < REFRESH_INTERVAL):
                    pending_tasks = self.task_cache["pending"]
                    paused_tasks = self.task_cache["paused"]
                else:
                    pending_tasks = orchestrator.current_tasks
                    paused_tasks = orchestrator.paused_tasks
                    self.task_cache = {
                        "pending": pending_tasks,
                        "paused": paused_tasks,
                        "last_updated": now
                    }
                
                pending_table = self._create_tasks_table(
                    pending_tasks, "Pendientes", search_query or "", channel, page
                )
                paused_table = self._create_tasks_table(
                    paused_tasks, "Pausadas", search_query or "", channel, page
                )
                
                # Calcular paginación
                total_tasks = len({k: v for k, v in {**pending_tasks, **paused_tasks}.items()
                                 if (not search_query or search_query.lower() in str(v).lower()) and
                                 (channel == "all" or v.get('channel_id') == channel)})
                total_pages = (total_tasks + TASKS_PER_PAGE - 1) // TASKS_PER_PAGE
                page_options = [{"label": f"Página {i+1}", "value": i} for i in range(total_pages)]
                
                # Gráficos
                dist_fig = self._create_tasks_distribution_graph(pending_tasks, paused_tasks)
                proc_fig = self._create_tasks_processing_time_graph({**pending_tasks, **paused_tasks})
                
                return [
                    html.Div([pending_table, html.Hr(), paused_table]),
                    page_options,
                    f"Mostrando página {page + 1} de {total_pages}",
                    dist_fig,
                    proc_fig
                ]
            except Exception as e:
                logger.error(f"Error al actualizar publicaciones pendientes: {str(e)}")
                return [
                    html.Div(f"Error: {str(e)}", className="text-danger"),
                    [], "", go.Figure(), go.Figure()
                ]
        
        @self.app.callback(
            [
                Output("task-details-modal", "is_open"),
                Output("task-details-content", "children")
            ],
            [
                Input({"type": "task-details", "index": dash.dependencies.ALL}, "n_clicks"),
                Input("close-task-details", "n_clicks")
            ],
            [
                State({"type": "task-details", "index": dash.dependencies.ALL}, "id"),
                State("task-details-modal", "is_open")
            ]
        )
        def toggle_task_details(n_clicks, close_clicks, task_ids, is_open):
            """Muestra los detalles de una tarea en un modal."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return [is_open, []]
            
            triggered_id = ctx.triggered[0]["prop_id"]
            if "close-task-details" in triggered_id:
                return [False, []]
            
            import json
            triggered_id = json.loads(triggered_id.split(".")[0])
            task_id = triggered_id["index"]
            
            orchestrator = self._load_component('orchestrator')
            if not orchestrator:
                return [True, html.P("No se pudo cargar el orquestador", className="text-danger")]
            
            task = orchestrator.current_tasks.get(task_id) or orchestrator.paused_tasks.get(task_id)
            if not task:
                return [True, html.P("Tarea no encontrada", className="text-danger")]
            
            details = [
                html.P(f"ID: {task_id}"),
                html.P(f"Tipo: {task.get('type', 'Desconocido')}"),
                html.P(f"Estado: {task.get('status', 'Desconocido')}"),
                html.P(f"Canal: {task.get('channel_id', 'Desconocido')}"),
                html.P(f"Plataforma: {task.get('platform', 'Todas')}"),
                html.P(f"Creado: {task.get('created_at', 'Desconocido')}"),
                html.P(f"Razón de Pausa: {task.get('paused_reason', 'N/A')}"),
                html.P(f"Content ID: {task.get('content_id', 'N/A')}"),
                html.P(f"Tags: {', '.join(task.get('content_tags', []))}"),
            ]
            
            return [True, details]
        
        @self.app.callback(
            Output("dummy-output", "children"),
            [
                Input({"type": "resume-task", "index": dash.dependencies.ALL}, "n_clicks"),
                Input({"type": "cancel-task", "index": dash.dependencies.ALL}, "n_clicks")
            ],
            [
                State({"type": "resume-task", "index": dash.dependencies.ALL}, "id"),
                State({"type": "cancel-task", "index": dash.dependencies.ALL}, "id")
            ]
        )
        def handle_task_actions(resume_clicks, cancel_clicks, resume_ids, cancel_ids):
            """Maneja las acciones de reanudar o cancelar tareas."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return ""
            
            triggered_id = ctx.triggered[0]["prop_id"]
            triggered_value = ctx.triggered[0]["value"]
            
            if not triggered_value:
                return ""
            
            orchestrator = self._load_component('orchestrator')
            if not orchestrator:
                logger.error("No se pudo cargar el orquestador")
                return ""
            
            try:
                import json
                triggered_id = json.loads(triggered_id.split(".")[0])
                task_id = triggered_id["index"]
                action = triggered_id["type"]
                
                if action == "resume-task":
                    task = orchestrator.paused_tasks.get(task_id)
                    if task:
                        task['status'] = 'initiated'
                        task['priority'] = task.get('priority', 2)
                        orchestrator.task_queue.put((task['priority'], task))
                        orchestrator.current_tasks[task_id] = task
                        del orchestrator.paused_tasks[task_id]
                        orchestrator._persist_task(task)
                        self.task_cache["pending"][task_id] = task
                        del self.task_cache["paused"][task_id]
                        logger.info(f"Tarea {task_id} reanudada")
                elif action == "cancel-task":
                    if task_id in orchestrator.current_tasks:
                        task = orchestrator.current_tasks[task_id]
                        task['status'] = 'cancelled'
                        task['cancelled_at'] = datetime.datetime.now().isoformat()
                        orchestrator.task_history.append(task)
                        del orchestrator.current_tasks[task_id]
                        orchestrator._persist_task(task)
                        del self.task_cache["pending"][task_id]
                        logger.info(f"Tarea {task_id} cancelada (pendiente)")
                    elif task_id in orchestrator.paused_tasks:
                        task = orchestrator.paused_tasks[task_id]
                        task['status'] = 'cancelled'
                        task['cancelled_at'] = datetime.datetime.now().isoformat()
                        orchestrator.task_history.append(task)
                        del orchestrator.paused_tasks[task_id]
                        orchestrator._persist_task(task)
                        del self.task_cache["paused"][task_id]
                        logger.info(f"Tarea {task_id} cancelada (pausada)")
                
                return ""
            except Exception as e:
                logger.error(f"Error al manejar acción de tarea: {str(e)}")
                return ""
        
        @self.app.callback(
            Output("export-status", "children"),
            Input("btn-export-tasks", "n_clicks"),
            [
                State("pending-tasks-search", "value"),
                State("filter-task-channel", "value")
            ]
        )
        def export_tasks(n_clicks, search_query, channel):
            """Exporta las tareas a CSV."""
            if not n_clicks:
                return ""
            
            try:
                orchestrator = self._load_component('orchestrator')
                if not orchestrator:
                    return html.Div("No se pudo cargar el orquestador", className="text-danger")
                
                pending_tasks = orchestrator.current_tasks
                paused_tasks = orchestrator.paused_tasks
                
                if search_query:
                    search_query = search_query.lower()
                    pending_tasks = {
                        task_id: task for task_id, task in pending_tasks.items()
                        if (search_query in task_id.lower() or
                            search_query in task.get('type', '').lower() or
                            search_query in task.get('status', '').lower() or
                            search_query in task.get('channel_id', '').lower() or
                            search_query in task.get('platform', '').lower())
                    }
                    paused_tasks = {
                        task_id: task for task_id, task in paused_tasks.items()
                        if (search_query in task_id.lower() or
                            search_query in task.get('type', '').lower() or
                            search_query in task.get('status', '').lower() or
                            search_query in task.get('channel_id', '').lower() or
                            search_query in task.get('platform', '').lower())
                    }
                
                if channel != "all":
                    pending_tasks = {
                        task_id: task for task_id, task in pending_tasks.items()
                        if task.get('channel_id') == channel
                    }
                    paused_tasks = {
                        task_id: task for task_id, task in paused_tasks.items()
                        if task.get('channel_id') == channel
                    }
                
                pending_file = self._export_tasks_to_csv(pending_tasks, "Pendientes")
                paused_file = self._export_tasks_to_csv(paused_tasks, "Pausadas")
                
                message = []
                if pending_file:
                    message.append(f"Tareas pendientes exportadas a {pending_file}")
                if paused_file:
                    message.append(f"Tareas pausadas exportadas a {paused_file}")
                
                if not message:
                    message.append("No hay tareas para exportar")
                
                return html.Div(
                    [html.P(msg) for msg in message],
                    className="text-success" if message[0].startswith("Tareas") else "text-muted"
                )
            except Exception as e:
                logger.error(f"Error al exportar tareas: {str(e)}")
                return html.Div(f"Error: {str(e)}", className="text-danger")
        
        @self.app.callback(
            Output("report-status", "children"),
            Input("btn-generate-report", "n_clicks"),
            [
                State("report-type", "value"),
                State("report-format", "value")
            ]
        )
        def generate_report(n_clicks, report_type, report_format):
            """Genera un reporte según el tipo y formato seleccionados."""
            if not n_clicks:
                return ""
            
            try:
                if report_type == "tasks":
                    filename = self._generate_task_report(report_format)
                    if filename:
                        return html.Div(
                            f"Reporte de tareas generado: {filename}",
                                                        className="text-success"
                        )
                    else:
                        return html.Div(
                            "No se pudo generar el reporte de tareas",
                            className="text-danger"
                        )
                else:
                    # Otros tipos de reportes (placeholder para mantener compatibilidad)
                    filename = f"{self.report_path}/{report_type}_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format}"
                    # Simulación de generación de reporte (implementar según necesidades)
                    with open(filename, 'w') as f:
                        f.write(f"Reporte {report_type} en formato {report_format}")
                    return html.Div(
                        f"Reporte {report_type} generado: {filename}",
                        className="text-success"
                    )
            except Exception as e:
                logger.error(f"Error al generar reporte {report_type}: {str(e)}")
                return html.Div(
                    f"Error al generar reporte: {str(e)}",
                    className="text-danger"
                )

        @self.app.callback(
            Output("reports-history", "children"),
            Input("btn-generate-report", "n_clicks"),
            Input("main-tabs", "active_tab")
        )
        def update_reports_history(n_clicks, active_tab):
            """Actualiza el historial de reportes generados."""
            if active_tab != "reports" and not n_clicks:
                return html.P("No hay reportes recientes", className="text-muted")
            
            try:
                reports = []
                for file in os.listdir(self.report_path):
                    if file.endswith(('.csv', '.json', '.pdf', '.xlsx')):
                        file_path = os.path.join(self.report_path, file)
                        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                        reports.append({
                            "name": file,
                            "path": file_path,
                            "time": file_time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                
                if not reports:
                    return html.P("No hay reportes generados", className="text-muted")
                
                rows = [
                    html.Tr([
                        html.Td(report["name"]),
                        html.Td(report["time"]),
                        html.Td(dbc.Button("Descargar", href=f"/reports/{report['name']}", color="link", size="sm"))
                    ]) for report in sorted(reports, key=lambda x: x["time"], reverse=True)[:10]
                ]
                
                return dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Nombre"),
                                html.Th("Fecha"),
                                html.Th("Acción")
                            ])
                        ),
                        html.Tbody(rows)
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True
                )
            except Exception as e:
                logger.error(f"Error al actualizar historial de reportes: {str(e)}")
                return html.Div(f"Error: {str(e)}", className="text-danger")

        @self.app.callback(
            Output("schedule-status", "children"),
            Input("btn-save-schedule", "n_clicks"),
            [
                State("schedule-daily", "checked"),
                State("schedule-weekly", "checked"),
                State("schedule-monthly", "checked"),
                State("schedule-tasks", "checked"),
                State("schedule-time", "value")
            ]
        )
        def save_report_schedule(n_clicks, daily, weekly, monthly, tasks, time):
            """Guarda la configuración de reportes programados."""
            if not n_clicks:
                return ""
            
            try:
                self.config["report_schedule"] = {
                    "daily": daily,
                    "weekly": weekly,
                    "monthly": monthly,
                    "tasks": tasks,
                    "time": time
                }
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
                
                # Actualizar programación (simulación, implementar según necesidades)
                schedule.clear()
                if daily:
                    schedule.every().day.at(time).do(self._generate_task_report, format="csv")
                if weekly:
                    schedule.every().week.at(time).do(self._generate_task_report, format="csv")
                if monthly:
                    schedule.every(30).days.at(time).do(self._generate_task_report, format="csv")
                if tasks:
                    schedule.every().day.at(time).do(self._generate_task_report, format="csv")
                
                return html.Div(
                    "Configuración de reportes programados guardada",
                    className="text-success"
                )
            except Exception as e:
                logger.error(f"Error al guardar configuración de reportes: {str(e)}")
                return html.Div(f"Error: {str(e)}", className="text-danger")

        @self.app.callback(
            Output("navbar-collapse", "is_open"),
            Input("navbar-toggler", "n_clicks"),
            State("navbar-collapse", "is_open")
        )
        def toggle_navbar(n_clicks, is_open):
            """Alterna la barra de navegación en modo responsive."""
            if n_clicks:
                return not is_open
            return is_open

        @self.app.callback(
            [
                Output("main-tabs", "active_tab"),
                Output("nav-overview", "active"),
                Output("nav-engagement", "active"),
                Output("nav-monetization", "active"),
                Output("nav-operations", "active"),
                Output("nav-trends", "active"),
                Output("nav-alerts", "active"),
                Output("nav-monitoring", "active"),
                Output("nav-reports", "active"),
                Output("nav-settings", "active")
            ],
            [
                Input("nav-overview", "n_clicks"),
                Input("nav-engagement", "n_clicks"),
                Input("nav-monetization", "n_clicks"),
                Input("nav-operations", "n_clicks"),
                Input("nav-trends", "n_clicks"),
                Input("nav-alerts", "n_clicks"),
                Input("nav-monitoring", "n_clicks"),
                Input("nav-reports", "n_clicks"),
                Input("nav-settings", "n_clicks")
            ]
        )
        def navigate_tabs(*args):
            """Navega entre tabs al hacer clic en la barra de navegación."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return ["overview"] + [True] + [False] * 8
            
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
            tab_mapping = {
                "nav-overview": "overview",
                "nav-engagement": "engagement",
                "nav-monetization": "monetization",
                "nav-operations": "operations",
                "nav-trends": "trends",
                "nav-alerts": "alerts",
                "nav-monitoring": "monitoring",
                "nav-reports": "reports",
                "nav-settings": "settings"
            }
            active_tab = tab_mapping.get(triggered_id, "overview")
            
            return [
                active_tab,
                active_tab == "overview",
                active_tab == "engagement",
                active_tab == "monetization",
                active_tab == "operations",
                active_tab == "trends",
                active_tab == "alerts",
                active_tab == "monitoring",
                active_tab == "reports",
                active_tab == "settings"
            ]

        @self.app.callback(
            Output("date-range-container", "style"),
            Input("filter-period", "value")
        )
        def toggle_date_range(period):
            """Muestra el selector de fechas para período personalizado."""
            return {"display": "block"} if period == "custom" else {"display": "none"}

    def start(self) -> bool:
        """Inicia el servidor del dashboard."""
        if self.running:
            logger.warning("El dashboard ya está corriendo")
            return True
        
        try:
            # Iniciar hilo para reportes programados
            def run_schedule():
                while self.running:
                    schedule.run_pending()
                    time.sleep(60)
            
            self.running = True
            self.report_thread = threading.Thread(target=run_schedule, daemon=True)
            self.report_thread.start()
            
            logger.info("Iniciando servidor Dash...")
            self.app.run_server(
                port=self.port,
                host=self.host,
                debug=self.debug
            )
            return True
        except Exception as e:
            logger.error(f"Error al iniciar el servidor Dash: {str(e)}")
            self.running = False
            return False

    def stop(self) -> None:
        """Detiene el dashboard y limpia recursos."""
        self.running = False
        if self.report_thread:
            self.report_thread.join(timeout=5.0)
        logger.info("Dashboard detenido")
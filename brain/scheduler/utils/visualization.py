"""
Módulo de visualización para el sistema de planificación.

Este módulo proporciona funcionalidades para:
- Visualizar la programación de tareas en diferentes formatos
- Generar gráficos de dependencias entre tareas
- Crear dashboards interactivos para monitoreo
- Exportar datos de planificación para análisis
"""

import logging
import json
import os
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
from io import BytesIO
import base64
import pandas as pd
from enum import Enum

# Configurar logger
logger = logging.getLogger("scheduler.visualization")

class ChartType(Enum):
    """Tipos de gráficos disponibles."""
    GANTT = "gantt"
    TIMELINE = "timeline"
    DEPENDENCY = "dependency"
    HEATMAP = "heatmap"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE = "performance"

class OutputFormat(Enum):
    """Formatos de salida para visualizaciones."""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    JSON = "json"
    CSV = "csv"

class SchedulerVisualizer:
    """
    Visualizador para el sistema de planificación de tareas.
    
    Proporciona funcionalidades para:
    - Generar gráficos de Gantt para visualizar la programación
    - Crear gráficos de dependencias entre tareas
    - Visualizar el uso de recursos y rendimiento
    - Exportar datos para análisis externo
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el visualizador.
        
        Args:
            config: Configuración del visualizador
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'visualizations')
        
        # Crear directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Estilo de gráficos
        self.style = self.config.get('style', {
            'figsize': (12, 8),
            'dpi': 100,
            'colors': {
                'pending': '#3498db',
                'running': '#f39c12',
                'completed': '#2ecc71',
                'failed': '#e74c3c',
                'canceled': '#95a5a6',
                'background': '#f8f9fa',
                'grid': '#ecf0f1'
            },
            'font': {
                'family': 'sans-serif',
                'size': 10
            }
        })
        
        # Configurar estilo de matplotlib
        plt.rcParams['font.family'] = self.style['font']['family']
        plt.rcParams['font.size'] = self.style['font']['size']
    
    def create_gantt_chart(self, tasks: List[Dict[str, Any]], 
                          output_file: Optional[str] = None,
                          output_format: OutputFormat = OutputFormat.PNG,
                          show_dependencies: bool = False) -> str:
        """
        Crea un gráfico de Gantt para visualizar la programación de tareas.
        
        Args:
            tasks: Lista de tareas con información de programación
            output_file: Ruta del archivo de salida (opcional)
            output_format: Formato de salida
            show_dependencies: Si se deben mostrar las dependencias entre tareas
            
        Returns:
            Ruta del archivo generado o datos en base64 si no se especifica archivo
        """
        try:
            # Preparar datos
            task_data = []
            for task in tasks:
                # Extraer información relevante
                task_id = task.get('id', 'unknown')
                task_name = task.get('name', task_id)
                task_type = task.get('task_type', 'unknown')
                
                # Fechas de inicio y fin
                start_time = task.get('start_time')
                if isinstance(start_time, str):
                    start_time = datetime.datetime.fromisoformat(start_time)
                
                end_time = task.get('end_time')
                if isinstance(end_time, str):
                    end_time = datetime.datetime.fromisoformat(end_time)
                elif end_time is None and task.get('duration'):
                    # Calcular fin basado en duración
                    duration = task.get('duration')
                    if isinstance(duration, (int, float)):
                        end_time = start_time + datetime.timedelta(seconds=duration)
                
                # Si no hay tiempo de fin, usar tiempo actual para tareas en ejecución
                if end_time is None and task.get('status') == 'running':
                    end_time = datetime.datetime.now()
                
                # Si aún no hay tiempo de fin, usar tiempo de inicio + 1 minuto
                if end_time is None:
                    end_time = start_time + datetime.timedelta(minutes=1)
                
                # Estado de la tarea
                status = task.get('status', 'pending')
                
                # Añadir a los datos si tenemos fechas válidas
                if start_time and end_time:
                                        task_data.append({
                        'id': task_id,
                        'name': task_name,
                        'type': task_type,
                        'start': start_time,
                        'end': end_time,
                        'status': status,
                        'duration': (end_time - start_time).total_seconds(),
                        'dependencies': task.get('dependencies', [])
                    })
            
            # Ordenar tareas por tiempo de inicio
            task_data.sort(key=lambda x: x['start'])
            
            # Crear figura
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
            
            # Configurar fondo
            ax.set_facecolor(self.style['colors']['background'])
            
            # Configurar ejes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Calcular rango de tiempo para el eje X
            if task_data:
                min_time = min(t['start'] for t in task_data)
                max_time = max(t['end'] for t in task_data)
                
                # Añadir margen
                time_range = (max_time - min_time).total_seconds()
                margin = time_range * 0.1  # 10% de margen
                
                min_time = min_time - datetime.timedelta(seconds=margin)
                max_time = max_time + datetime.timedelta(seconds=margin)
                
                ax.set_xlim(min_time, max_time)
            
            # Dibujar barras para cada tarea
            y_positions = {}
            current_y = 0
            
            for i, task in enumerate(task_data):
                # Determinar color según estado
                color = self.style['colors'].get(
                    task['status'], 
                    self.style['colors']['pending']
                )
                
                # Posición Y para la tarea
                y_positions[task['id']] = current_y
                
                # Dibujar barra
                ax.barh(
                    current_y, 
                    task['duration'],
                    left=task['start'],
                    height=0.5,
                    color=color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Añadir etiqueta de tarea
                ax.text(
                    task['start'] + datetime.timedelta(seconds=task['duration']/2),
                    current_y,
                    task['name'],
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                    fontweight='bold'
                )
                
                current_y += 1
            
            # Dibujar dependencias si está habilitado
            if show_dependencies:
                for task in task_data:
                    for dep_id in task['dependencies']:
                        if dep_id in y_positions:
                            # Encontrar posiciones de las tareas
                            start_y = y_positions[dep_id]
                            end_y = y_positions[task['id']]
                            
                            # Encontrar tiempos
                            for t in task_data:
                                if t['id'] == dep_id:
                                    start_x = t['end']
                                    break
                            
                            end_x = task['start']
                            
                            # Dibujar flecha de dependencia
                            ax.annotate(
                                '',
                                xy=(end_x, end_y),
                                xytext=(start_x, start_y),
                                arrowprops=dict(
                                    arrowstyle='->',
                                    color='gray',
                                    alpha=0.6,
                                    connectionstyle='arc3,rad=0.2'
                                )
                            )
            
            # Configurar etiquetas de eje Y
            ax.set_yticks(range(len(task_data)))
            ax.set_yticklabels([t['name'] for t in task_data])
            
            # Añadir título y etiquetas
            ax.set_title('Programación de Tareas (Gráfico de Gantt)')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Tareas')
            
            # Añadir cuadrícula
            ax.grid(True, axis='x', linestyle='--', alpha=0.7, color=self.style['colors']['grid'])
            
            # Añadir leyenda
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, color=self.style['colors']['pending'], label='Pendiente'),
                plt.Rectangle((0, 0), 1, 1, color=self.style['colors']['running'], label='En ejecución'),
                plt.Rectangle((0, 0), 1, 1, color=self.style['colors']['completed'], label='Completada'),
                plt.Rectangle((0, 0), 1, 1, color=self.style['colors']['failed'], label='Fallida'),
                plt.Rectangle((0, 0), 1, 1, color=self.style['colors']['canceled'], label='Cancelada')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Guardar o devolver según configuración
            if output_file:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Guardar en el formato especificado
                if output_format == OutputFormat.PNG:
                    plt.savefig(output_file, format='png')
                elif output_format == OutputFormat.SVG:
                    plt.savefig(output_file, format='svg')
                elif output_format == OutputFormat.HTML:
                    # Convertir a HTML usando mpld3 o similar
                    try:
                        import mpld3
                        html_content = mpld3.fig_to_html(fig)
                        with open(output_file, 'w') as f:
                            f.write(html_content)
                    except ImportError:
                        logger.warning("mpld3 no está instalado. Guardando como PNG en su lugar.")
                        plt.savefig(output_file.replace('.html', '.png'), format='png')
                
                plt.close(fig)
                return output_file
            else:
                # Devolver como base64 para mostrar en web
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close(fig)
                
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            logger.error(f"Error al crear gráfico de Gantt: {str(e)}")
            raise
    
    def create_dependency_graph(self, tasks: List[Dict[str, Any]],
                               output_file: Optional[str] = None,
                               output_format: OutputFormat = OutputFormat.PNG) -> str:
        """
        Crea un gráfico de dependencias entre tareas.
        
        Args:
            tasks: Lista de tareas con información de dependencias
            output_file: Ruta del archivo de salida (opcional)
            output_format: Formato de salida
            
        Returns:
            Ruta del archivo generado o datos en base64 si no se especifica archivo
        """
        try:
            # Crear grafo dirigido
            G = nx.DiGraph()
            
            # Añadir nodos y aristas
            for task in tasks:
                task_id = task.get('id', 'unknown')
                task_name = task.get('name', task_id)
                task_status = task.get('status', 'pending')
                
                # Añadir nodo con atributos
                G.add_node(task_id, name=task_name, status=task_status)
                
                # Añadir aristas para dependencias
                for dep_id in task.get('dependencies', []):
                    G.add_edge(dep_id, task_id)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
            
            # Configurar fondo
            ax.set_facecolor(self.style['colors']['background'])
            
            # Calcular layout (posicionamiento de nodos)
            pos = nx.spring_layout(G, seed=42)
            
            # Preparar colores de nodos según estado
            node_colors = []
            for node in G.nodes():
                status = G.nodes[node].get('status', 'pending')
                node_colors.append(self.style['colors'].get(status, self.style['colors']['pending']))
            
            # Dibujar nodos
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=500,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                ax=ax
            )
            
            # Dibujar aristas
            nx.draw_networkx_edges(
                G, pos,
                edge_color='gray',
                width=1.0,
                alpha=0.6,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
            
            # Dibujar etiquetas
            nx.draw_networkx_labels(
                G, pos,
                labels={node: G.nodes[node].get('name', node) for node in G.nodes()},
                font_size=8,
                font_family=self.style['font']['family'],
                font_weight='bold',
                ax=ax
            )
            
            # Añadir título
            ax.set_title('Gráfico de Dependencias entre Tareas')
            
            # Quitar ejes
            ax.axis('off')
            
            # Añadir leyenda
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.style['colors']['pending'], 
                          markersize=10, label='Pendiente'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.style['colors']['running'], 
                          markersize=10, label='En ejecución'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.style['colors']['completed'], 
                          markersize=10, label='Completada'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.style['colors']['failed'], 
                          markersize=10, label='Fallida')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Guardar o devolver según configuración
            if output_file:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Guardar en el formato especificado
                if output_format == OutputFormat.PNG:
                    plt.savefig(output_file, format='png')
                elif output_format == OutputFormat.SVG:
                    plt.savefig(output_file, format='svg')
                
                plt.close(fig)
                return output_file
            else:
                # Devolver como base64 para mostrar en web
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close(fig)
                
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            logger.error(f"Error al crear gráfico de dependencias: {str(e)}")
            raise
    
    def create_resource_usage_chart(self, resource_data: Dict[str, List[Dict[str, Any]]],
                                  output_file: Optional[str] = None,
                                  output_format: OutputFormat = OutputFormat.PNG) -> str:
        """
        Crea un gráfico de uso de recursos a lo largo del tiempo.
        
        Args:
            resource_data: Diccionario con datos de uso de recursos por tipo
            output_file: Ruta del archivo de salida (opcional)
            output_format: Formato de salida
            
        Returns:
            Ruta del archivo generado o datos en base64 si no se especifica archivo
        """
        try:
            # Crear figura
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
            
            # Configurar fondo
            ax.set_facecolor(self.style['colors']['background'])
            
            # Colores para diferentes recursos
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            
            # Dibujar líneas para cada recurso
            for i, (resource_name, data_points) in enumerate(resource_data.items()):
                # Extraer tiempos y valores
                times = [dp['timestamp'] if isinstance(dp['timestamp'], datetime.datetime) 
                        else datetime.datetime.fromisoformat(dp['timestamp']) 
                        for dp in data_points]
                
                values = [dp['value'] for dp in data_points]
                
                # Dibujar línea
                ax.plot(
                    times, 
                    values, 
                    label=resource_name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    alpha=0.8
                )
            
            # Configurar ejes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Añadir título y etiquetas
            ax.set_title('Uso de Recursos a lo Largo del Tiempo')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Uso (%)')
            
            # Añadir cuadrícula
            ax.grid(True, linestyle='--', alpha=0.7, color=self.style['colors']['grid'])
            
            # Añadir leyenda
            ax.legend(loc='upper right')
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Guardar o devolver según configuración
            if output_file:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Guardar en el formato especificado
                if output_format == OutputFormat.PNG:
                    plt.savefig(output_file, format='png')
                elif output_format == OutputFormat.SVG:
                    plt.savefig(output_file, format='svg')
                
                plt.close(fig)
                return output_file
            else:
                # Devolver como base64 para mostrar en web
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close(fig)
                
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            logger.error(f"Error al crear gráfico de uso de recursos: {str(e)}")
            raise
    
    def create_performance_heatmap(self, performance_data: List[Dict[str, Any]],
                                 output_file: Optional[str] = None,
                                 output_format: OutputFormat = OutputFormat.PNG) -> str:
        """
        Crea un mapa de calor para visualizar el rendimiento de tareas por hora y día.
        
        Args:
            performance_data: Lista de datos de rendimiento con timestamp y valor
            output_file: Ruta del archivo de salida (opcional)
            output_format: Formato de salida
            
        Returns:
            Ruta del archivo generado o datos en base64 si no se especifica archivo
        """
        try:
            # Preparar datos para el mapa de calor
            # Convertir a DataFrame para facilitar el procesamiento
            df = pd.DataFrame(performance_data)
            
            # Asegurar que timestamp es datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Crear matriz para el mapa de calor
            heatmap_data = pd.pivot_table(
                df, 
                values='value', 
                index='day_of_week', 
                columns='hour', 
                aggfunc='mean',
                fill_value=0
            )
            
            # Crear figura
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
            
            # Dibujar mapa de calor
            im = ax.imshow(heatmap_data, cmap='viridis')
            
            # Añadir barra de color
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Rendimiento', rotation=-90, va="bottom")
            
            # Configurar ejes
            ax.set_xticks(np.arange(24))
            ax.set_yticks(np.arange(7))
            
            # Etiquetas para días de la semana
            day_labels = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            ax.set_yticklabels(day_labels)
            
            # Etiquetas para horas
            ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
            
            # Rotar etiquetas de horas
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Añadir título
            ax.set_title('Mapa de Calor de Rendimiento por Hora y Día')
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Guardar o devolver según configuración
            if output_file:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Guardar en el formato especificado
                if output_format == OutputFormat.PNG:
                    plt.savefig(output_file, format='png')
                elif output_format == OutputFormat.SVG:
                    plt.savefig(output_file, format='svg')
                
                plt.close(fig)
                return output_file
            else:
                # Devolver como base64 para mostrar en web
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close(fig)
                
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            logger.error(f"Error al crear mapa de calor de rendimiento: {str(e)}")
            raise
    
    def export_task_data(self, tasks: List[Dict[str, Any]], 
                        output_file: str,
                        output_format: OutputFormat = OutputFormat.CSV) -> str:
        """
        Exporta datos de tareas a diferentes formatos para análisis externo.
        
        Args:
            tasks: Lista de tareas con información
            output_file: Ruta del archivo de salida
            output_format: Formato de salida (CSV, JSON)
            
        Returns:
            Ruta del archivo generado
        """
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Exportar según formato
            if output_format == OutputFormat.CSV:
                # Convertir a DataFrame
                df = pd.DataFrame(tasks)
                
                # Manejar columnas de fecha
                for col in ['start_time', 'end_time', 'created_at', 'scheduled_for']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                
                # Guardar como CSV
                df.to_csv(output_file, index=False)
                
            elif output_format == OutputFormat.JSON:
                # Preparar datos para JSON
                json_data = []
                for task in tasks:
                    # Crear copia para no modificar original
                    task_copy = task.copy()
                    
                    # Convertir fechas a strings
                    for key, value in task_copy.items():
                        if isinstance(value, datetime.datetime):
                            task_copy[key] = value.isoformat()
                    
                    json_data.append(task_copy)
                
                # Guardar como JSON
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            return output_file
                
        except Exception as e:
            logger.error(f"Error al exportar datos de tareas: {str(e)}")
            raise
    
    def create_timeline_chart(self, events: List[Dict[str, Any]],
                            output_file: Optional[str] = None,
                            output_format: OutputFormat = OutputFormat.PNG) -> str:
        """
        Crea un gráfico de línea de tiempo para visualizar eventos.
        
        Args:
            events: Lista de eventos con timestamp y descripción
            output_file: Ruta del archivo de salida (opcional)
            output_format: Formato de salida
            
        Returns:
            Ruta del archivo generado o datos en base64 si no se especifica archivo
        """
        try:
            # Crear figura
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
            
            # Configurar fondo
            ax.set_facecolor(self.style['colors']['background'])
            
            # Extraer tiempos y descripciones
            times = []
            descriptions = []
            categories = []
            
            for event in events:
                # Convertir timestamp a datetime si es string
                timestamp = event.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                
                times.append(timestamp)
                descriptions.append(event.get('description', ''))
                categories.append(event.get('category', 'default'))
            
            # Crear DataFrame
            df = pd.DataFrame({
                'time': times,
                'description': descriptions,
                'category': categories
            })
            
            # Ordenar por tiempo
            df = df.sort_values('time')
            
            # Obtener categorías únicas
            unique_categories = df['category'].unique()
            
            # Asignar colores a categorías
            category_colors = {}
            for i, category in enumerate(unique_categories):
                category_colors[category] = plt.cm.tab10(i % 10)
            
            # Dibujar línea de tiempo
            for i, (_, event) in enumerate(df.iterrows()):
                # Color según categoría
                color = category_colors.get(event['category'], 'blue')
                
                # Dibujar punto
                ax.scatter(
                    event['time'], 
                    0, 
                    s=100, 
                    color=color, 
                    zorder=2,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                # Añadir etiqueta
                ax.annotate(
                    event['description'],
                    xy=(event['time'], 0),
                    xytext=(0, (-1)**i * 20),  # Alternar arriba/abajo
                    textcoords='offset points',
                    ha='center',
                    va='center',
                    fontsize=8,
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc='white',
                        alpha=0.7,
                        ec='gray',
                        lw=0.5
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0.0',
                        color='gray'
                    )
                )
            
            # Dibujar línea horizontal
            min_time = min(df['time'])
            max_time = max(df['time'])
            
            # Añadir margen
            time_range = (max_time - min_time).total_seconds()
            margin = time_range * 0.1  # 10% de margen
            
            min_time = min_time - datetime.timedelta(seconds=margin)
            max_time = max_time + datetime.timedelta(seconds=margin)
            
            ax.plot(
                [min_time, max_time], 
                [0, 0], 
                'k-', 
                alpha=0.3, 
                zorder=1
            )
            
            # Configurar ejes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Ocultar eje Y
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Añadir título
            ax.set_title('Línea de Tiempo de Eventos')
            
            # Añadir leyenda para categorías
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=10, label=category)
                for category, color in category_colors.items()
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Ajustar diseño
            plt.tight_layout()
            
            # Guardar o devolver según configuración
            if output_file:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Guardar en el formato especificado
                if output_format == OutputFormat.PNG:
                    plt.savefig(output_file, format='png')
                elif output_format == OutputFormat.SVG:
                    plt.savefig(output_file, format='svg')
                
                plt.close(fig)
                return output_file
            else:
                # Devolver como base64 para mostrar en web
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close(fig)
                
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            logger.error(f"Error al crear gráfico de línea de tiempo: {str(e)}")
            raise
    
    def create_dashboard(self, data: Dict[str, Any], 
                       output_dir: str,
                       include_charts: List[ChartType] = None) -> str:
        """
        Crea un dashboard HTML con múltiples visualizaciones.
        
        Args:
            data: Datos para las visualizaciones
            output_dir: Directorio de salida
            include_charts: Lista de tipos de gráficos a incluir
            
        Returns:
            Ruta del archivo HTML generado
        """
        try:
            # Determinar qué gráficos incluir
            if include_charts is None:
                include_charts = [
                    ChartType.GANTT,
                    ChartType.DEPENDENCY,
                    ChartType.RESOURCE_USAGE,
                    ChartType.PERFORMANCE
                ]
            
            # Crear directorio de salida
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar gráficos y obtener rutas de imágenes
            chart_images = {}
            
            # Generar gráfico de Gantt si está incluido
            if ChartType.GANTT in include_charts and 'tasks' in data:
                gantt_path = os.path.join(output_dir, 'gantt_chart.png')
                self.create_gantt_chart(
                    data['tasks'],
                    output_file=gantt_path,
                    output_format=OutputFormat.PNG
                )
                chart_images['gantt'] = os.path.basename(gantt_path)
            
            # Generar gráfico de dependencias si está incluido
            if ChartType.DEPENDENCY in include_charts and 'tasks' in data:
                dependency_path = os.path.join(output_dir, 'dependency_graph.png')
                self.create_dependency_graph(
                    data['tasks'],
                    output_file=dependency_path,
                    output_format=OutputFormat.PNG
                )
                chart_images['dependency'] = os.path.basename(dependency_path)
            
            # Generar gráfico de uso de recursos si está incluido
            if ChartType.RESOURCE_USAGE in include_charts and 'resources' in data:
                resource_path = os.path.join(output_dir, 'resource_usage.png')
                self.create_resource_usage_chart(
                    data['resources'],
                    output_file=resource_path,
                    output_format=OutputFormat.PNG
                )
                chart_images['resource_usage'] = os.path.basename(resource_path)
            
            # Generar mapa de calor de rendimiento si está incluido
            if ChartType.PERFORMANCE in include_charts and 'performance' in data:
                                performance_path = os.path.join(output_dir, 'performance_heatmap.png')
                self.create_performance_heatmap(
                    data['performance'],
                    output_file=performance_path,
                    output_format=OutputFormat.PNG
                )
                chart_images['performance'] = os.path.basename(performance_path)
            
            # Generar línea de tiempo si está incluida
            if ChartType.TIMELINE in include_charts and 'events' in data:
                timeline_path = os.path.join(output_dir, 'timeline_chart.png')
                self.create_timeline_chart(
                    data['events'],
                    output_file=timeline_path,
                    output_format=OutputFormat.PNG
                )
                chart_images['timeline'] = os.path.basename(timeline_path)
            
            # Crear archivo HTML con todos los gráficos
            html_path = os.path.join(output_dir, 'dashboard.html')
            
            # Generar contenido HTML
            html_content = self._generate_dashboard_html(
                chart_images=chart_images,
                data=data,
                title="Panel de Control del Planificador"
            )
            
            # Guardar archivo HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_path
                
        except Exception as e:
            logger.error(f"Error al crear dashboard: {str(e)}")
            raise
    
    def _generate_dashboard_html(self, chart_images: Dict[str, str], 
                               data: Dict[str, Any],
                               title: str = "Dashboard") -> str:
        """
        Genera el contenido HTML para el dashboard.
        
        Args:
            chart_images: Diccionario con rutas relativas a imágenes de gráficos
            data: Datos originales para mostrar estadísticas
            title: Título del dashboard
            
        Returns:
            Contenido HTML como string
        """
        # Calcular estadísticas básicas para mostrar
        stats = self._calculate_dashboard_stats(data)
        
        # Crear HTML con Bootstrap para estilo
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .dashboard-header {{
            background-color: #343a40;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .chart-container {{
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .chart-img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 14px;
        }}
        .status-pending {{
            color: #ffc107;
        }}
        .status-running {{
            color: #17a2b8;
        }}
        .status-completed {{
            color: #28a745;
        }}
        .status-failed {{
            color: #dc3545;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="dashboard-header">
            <h1>{title}</h1>
            <p>Generado el {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        
        <div class="row">
            <!-- Estadísticas -->
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_tasks', 0)}</div>
                    <div class="stat-label">Tareas Totales</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value status-pending">{stats.get('pending_tasks', 0)}</div>
                    <div class="stat-label">Tareas Pendientes</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value status-running">{stats.get('running_tasks', 0)}</div>
                    <div class="stat-label">Tareas en Ejecución</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value status-completed">{stats.get('completed_tasks', 0)}</div>
                    <div class="stat-label">Tareas Completadas</div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value status-failed">{stats.get('failed_tasks', 0)}</div>
                    <div class="stat-label">Tareas Fallidas</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('avg_execution_time', '0.0')} s</div>
                    <div class="stat-label">Tiempo Medio de Ejecución</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('success_rate', '0')}%</div>
                    <div class="stat-label">Tasa de Éxito</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('task_types', 0)}</div>
                    <div class="stat-label">Tipos de Tareas</div>
                </div>
            </div>
        </div>
        
        <!-- Gráficos -->
"""
        
        # Añadir gráfico de Gantt si existe
        if 'gantt' in chart_images:
            html += f"""
        <div class="row mt-4">
            <div class="col-12">
                <div class="chart-container">
                    <h3>Programación de Tareas (Gráfico de Gantt)</h3>
                    <img src="{chart_images['gantt']}" class="chart-img" alt="Gráfico de Gantt">
                </div>
            </div>
        </div>
"""
        
        # Añadir gráfico de dependencias si existe
        if 'dependency' in chart_images:
            html += f"""
        <div class="row mt-4">
            <div class="col-12">
                <div class="chart-container">
                    <h3>Gráfico de Dependencias entre Tareas</h3>
                    <img src="{chart_images['dependency']}" class="chart-img" alt="Gráfico de Dependencias">
                </div>
            </div>
        </div>
"""
        
        # Añadir gráficos de recursos y rendimiento en la misma fila si existen ambos
        if 'resource_usage' in chart_images or 'performance' in chart_images:
            html += f"""
        <div class="row mt-4">
"""
            
            # Columna para uso de recursos
            if 'resource_usage' in chart_images:
                html += f"""
            <div class="col-md-6">
                <div class="chart-container">
                    <h3>Uso de Recursos</h3>
                    <img src="{chart_images['resource_usage']}" class="chart-img" alt="Uso de Recursos">
                </div>
            </div>
"""
            
            # Columna para mapa de calor de rendimiento
            if 'performance' in chart_images:
                html += f"""
            <div class="col-md-6">
                <div class="chart-container">
                    <h3>Rendimiento por Hora y Día</h3>
                    <img src="{chart_images['performance']}" class="chart-img" alt="Mapa de Calor de Rendimiento">
                </div>
            </div>
"""
            
            html += """
        </div>
"""
        
        # Añadir línea de tiempo si existe
        if 'timeline' in chart_images:
            html += f"""
        <div class="row mt-4">
            <div class="col-12">
                <div class="chart-container">
                    <h3>Línea de Tiempo de Eventos</h3>
                    <img src="{chart_images['timeline']}" class="chart-img" alt="Línea de Tiempo">
                </div>
            </div>
        </div>
"""
        
        # Añadir tabla de tareas recientes si hay datos
        if 'tasks' in data and len(data['tasks']) > 0:
            # Limitar a las 10 tareas más recientes
            recent_tasks = sorted(
                data['tasks'], 
                key=lambda x: x.get('start_time', datetime.datetime.now()), 
                reverse=True
            )[:10]
            
            html += """
        <div class="row mt-4">
            <div class="col-12">
                <div class="chart-container">
                    <h3>Tareas Recientes</h3>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Nombre</th>
                                    <th>Tipo</th>
                                    <th>Estado</th>
                                    <th>Inicio</th>
                                    <th>Fin</th>
                                    <th>Duración</th>
                                </tr>
                            </thead>
                            <tbody>
"""
            
            # Añadir filas para cada tarea reciente
            for task in recent_tasks:
                task_id = task.get('id', 'N/A')
                task_name = task.get('name', 'Sin nombre')
                task_type = task.get('type', 'N/A')
                status = task.get('status', 'pending')
                
                # Formatear fechas
                start_time = task.get('start', datetime.datetime.now())
                if isinstance(start_time, str):
                    start_time = datetime.datetime.fromisoformat(start_time)
                
                end_time = task.get('end', datetime.datetime.now())
                if isinstance(end_time, str):
                    end_time = datetime.datetime.fromisoformat(end_time)
                
                # Calcular duración
                duration = end_time - start_time
                duration_str = f"{duration.total_seconds():.1f} s"
                
                # Formatear fechas para mostrar
                start_str = start_time.strftime('%d/%m/%Y %H:%M:%S')
                end_str = end_time.strftime('%d/%m/%Y %H:%M:%S')
                
                # Determinar clase CSS según estado
                status_class = f"status-{status}" if status in ['pending', 'running', 'completed', 'failed'] else ""
                
                html += f"""
                                <tr>
                                    <td>{task_id}</td>
                                    <td>{task_name}</td>
                                    <td>{task_type}</td>
                                    <td class="{status_class}">{status.capitalize()}</td>
                                    <td>{start_str}</td>
                                    <td>{end_str}</td>
                                    <td>{duration_str}</td>
                                </tr>
"""
            
            html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
"""
        
        # Cerrar HTML
        html += """
        <div class="footer">
            <p>Generado por el Sistema de Planificación de Tareas</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        
        return html
    
    def _calculate_dashboard_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula estadísticas para mostrar en el dashboard.
        
        Args:
            data: Datos originales
            
        Returns:
            Diccionario con estadísticas calculadas
        """
        stats = {}
        
        # Estadísticas de tareas
        if 'tasks' in data:
            tasks = data['tasks']
            
            # Contar tareas por estado
            total_tasks = len(tasks)
            pending_tasks = sum(1 for t in tasks if t.get('status') == 'pending')
            running_tasks = sum(1 for t in tasks if t.get('status') == 'running')
            completed_tasks = sum(1 for t in tasks if t.get('status') == 'completed')
            failed_tasks = sum(1 for t in tasks if t.get('status') == 'failed')
            
            # Calcular tasa de éxito
            if completed_tasks + failed_tasks > 0:
                success_rate = round((completed_tasks / (completed_tasks + failed_tasks)) * 100)
            else:
                success_rate = 0
            
            # Calcular tiempo medio de ejecución
            execution_times = []
            for task in tasks:
                if task.get('status') == 'completed' and 'start' in task and 'end' in task:
                    start = task['start']
                    end = task['end']
                    
                    if isinstance(start, str):
                        start = datetime.datetime.fromisoformat(start)
                    if isinstance(end, str):
                        end = datetime.datetime.fromisoformat(end)
                    
                    execution_times.append((end - start).total_seconds())
            
            avg_execution_time = 0
            if execution_times:
                avg_execution_time = round(sum(execution_times) / len(execution_times), 1)
            
            # Contar tipos de tareas únicos
            task_types = len(set(t.get('type', 'unknown') for t in tasks))
            
            # Guardar estadísticas
            stats.update({
                'total_tasks': total_tasks,
                'pending_tasks': pending_tasks,
                'running_tasks': running_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'task_types': task_types
            })
        
        return stats
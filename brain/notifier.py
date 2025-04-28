"""
Notifier - Sistema de notificaciones y alertas

Este módulo gestiona las notificaciones del sistema:
- Alertas de shadowbans y problemas de distribución
- Notificaciones de saturación de nichos
- Alertas de rendimiento (caídas de CTR, oportunidades)
- Notificaciones de tareas completadas
"""

import os
import sys
import json
import logging
import datetime
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'notifier.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Notifier')

class Notifier:
    """
    Sistema de notificaciones que gestiona alertas y mensajes
    para diferentes eventos del sistema de monetización.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Notifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el sistema de notificaciones si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Notifier...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Cargar configuración de plataformas
        self.platforms_file = os.path.join('config', 'platforms.json')
        self.platforms = self._load_platforms()
        
        # Configuración de alertas
        self.alerts_config = self.strategy.get('analytics', {}).get('alerts', {})
        
        # Historial de notificaciones
        self.notification_history = []
        
        # Configuración de canales de notificación
        self.notification_channels = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_app_password',
                'recipients': ['your_email@gmail.com']
            },
            'telegram': {
                'enabled': False,
                'bot_token': 'YOUR_TELEGRAM_BOT_TOKEN',
                'chat_id': 'YOUR_TELEGRAM_CHAT_ID'
            },
            'discord': {
                'enabled': False,
                'webhook_url': 'YOUR_DISCORD_WEBHOOK_URL'
            },
            'console': {
                'enabled': True
            },
            'log': {
                'enabled': True
            }
        }
        
        self._initialized = True
        logger.info("Notifier inicializado correctamente")
    
    def _load_strategy(self) -> Dict:
        """Carga la configuración de estrategia desde el archivo JSON"""
        try:
            if os.path.exists(self.strategy_file):
                with open(self.strategy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estrategia no encontrado: {self.strategy_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar estrategia: {str(e)}")
            return {}
    
    def _load_platforms(self) -> Dict:
        """Carga la configuración de plataformas desde el archivo JSON"""
        try:
            if os.path.exists(self.platforms_file):
                with open(self.platforms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de plataformas no encontrado: {self.platforms_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar plataformas: {str(e)}")
            return {}
    
    def send_notification(self, notification_type: str, subject: str, message: str, 
                         data: Dict = None, urgency: str = 'normal', 
                         channels: List[str] = None) -> str:
        """
        Envía una notificación a través de los canales configurados
        
        Args:
            notification_type: Tipo de notificación (shadowban, performance, niche, etc.)
            subject: Asunto de la notificación
            message: Mensaje principal
            data: Datos adicionales relacionados con la notificación
            urgency: Nivel de urgencia (low, normal, high, critical)
            channels: Lista de canales a utilizar (si es None, usa todos los habilitados)
            
        Returns:
            ID de la notificación enviada
        """
        # Generar ID de notificación
        notification_id = f"{notification_type}_{int(datetime.datetime.now().timestamp())}"
        
        # Crear objeto de notificación
        notification = {
            'id': notification_id,
            'type': notification_type,
            'subject': subject,
            'message': message,
            'data': data or {},
            'urgency': urgency,
            'timestamp': datetime.datetime.now().isoformat(),
            'channels_sent': []
        }
        
        # Determinar canales a utilizar
        if channels is None:
            channels = [channel for channel, config in self.notification_channels.items() 
                       if config.get('enabled', False)]
        
        # Enviar a cada canal
        for channel in channels:
            if channel not in self.notification_channels or not self.notification_channels[channel].get('enabled', False):
                logger.warning(f"Canal de notificación no disponible o deshabilitado: {channel}")
                continue
            
            try:
                self._send_to_channel(channel, notification)
                notification['channels_sent'].append(channel)
            except Exception as e:
                logger.error(f"Error al enviar notificación a {channel}: {str(e)}")
        
        # Registrar en historial
        self.notification_history.append(notification)
        
        # Limitar tamaño del historial
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]
        
        logger.info(f"Notificación {notification_id} enviada a {len(notification['channels_sent'])} canales")
        
        return notification_id
    
    def _send_to_channel(self, channel: str, notification: Dict):
        """Envía una notificación a un canal específico"""
        if channel == 'email':
            self._send_email(notification)
        elif channel == 'telegram':
            self._send_telegram(notification)
        elif channel == 'discord':
            self._send_discord(notification)
        elif channel == 'console':
            self._send_console(notification)
        elif channel == 'log':
            self._send_log(notification)
        else:
            logger.warning(f"Canal de notificación desconocido: {channel}")
    
    def _send_email(self, notification: Dict):
        """Envía notificación por email"""
        config = self.notification_channels['email']
        
        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = config['username']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"[{notification['urgency'].upper()}] {notification['subject']}"
        
        # Cuerpo del mensaje
        body = f"{notification['message']}\n\n"
        
        if notification['data']:
            body += "Detalles adicionales:\n"
            for key, value in notification['data'].items():
                body += f"- {key}: {value}\n"
        
        body += f"\nFecha: {notification['timestamp']}"
        body += f"\nID: {notification['id']}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Enviar email
        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            logger.info(f"Email enviado a {len(config['recipients'])} destinatarios")
        except Exception as e:
            logger.error(f"Error al enviar email: {str(e)}")
            raise
    
    def _send_telegram(self, notification: Dict):
        """Envía notificación por Telegram"""
        config = self.notification_channels['telegram']
        
        # Crear mensaje
        urgency_emoji = self._get_urgency_emoji(notification['urgency'])
        message = f"{urgency_emoji} *{notification['subject']}*\n\n{notification['message']}\n"
        
        if notification['data']:
            message += "\n*Detalles adicionales:*\n"
            for key, value in notification['data'].items():
                message += f"- {key}: {value}\n"
        
        # Enviar a Telegram
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {
            'chat_id': config['chat_id'],
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Error al enviar mensaje a Telegram: {response.text}")
            raise Exception(f"Error de Telegram: {response.status_code}")
        
        logger.info("Mensaje enviado a Telegram correctamente")
    
    def _send_discord(self, notification: Dict):
        """Envía notificación por Discord"""
        config = self.notification_channels['discord']
        
        # Crear mensaje
        color = self._get_urgency_color(notification['urgency'])
        
        embed = {
            'title': notification['subject'],
            'description': notification['message'],
            'color': color,
            'fields': [],
            'timestamp': notification['timestamp']
        }
        
        # Añadir campos de datos
        if notification['data']:
            for key, value in notification['data'].items():
                embed['fields'].append({
                    'name': key,
                    'value': str(value),
                    'inline': True
                })
        
        # Enviar a Discord
        payload = {
            'embeds': [embed],
            'username': 'Content Bot Notifier'
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        
        if response.status_code != 204:
            logger.error(f"Error al enviar mensaje a Discord: {response.text}")
            raise Exception(f"Error de Discord: {response.status_code}")
        
        logger.info("Mensaje enviado a Discord correctamente")
    
    def _send_console(self, notification: Dict):
        """Muestra notificación en consola"""
        urgency_prefix = f"[{notification['urgency'].upper()}]"
        
        print(f"\n{urgency_prefix} {notification['subject']}")
        print("-" * 50)
        print(notification['message'])
        
        if notification['data']:
            print("\nDetalles adicionales:")
            for key, value in notification['data'].items():
                print(f"- {key}: {value}")
        
        print(f"\nFecha: {notification['timestamp']}")
        print("-" * 50)
        
        logger.info("Notificación mostrada en consola")
    
    def _send_log(self, notification: Dict):
        """Registra notificación en el log"""
        log_message = f"[{notification['urgency'].upper()}] {notification['subject']}: {notification['message']}"
        
        if notification['urgency'] == 'critical':
            logger.critical(log_message)
        elif notification['urgency'] == 'high':
            logger.error(log_message)
        elif notification['urgency'] == 'normal':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _get_urgency_emoji(self, urgency: str) -> str:
        """Devuelve emoji según nivel de urgencia"""
        if urgency == 'critical':
            return '🚨'
        elif urgency == 'high':
            return '⚠️'
        elif urgency == 'normal':
            return '📊'
        else:
            return 'ℹ️'
    
    def _get_urgency_color(self, urgency: str) -> int:
        """Devuelve color según nivel de urgencia (formato Discord)"""
        if urgency == 'critical':
            return 16711680  # Rojo
        elif urgency == 'high':
            return 16737280  # Naranja
        elif urgency == 'normal':
            return 16776960  # Amarillo
        else:
            return 5592575  # Azul
    
    def notify_shadowban(self, platform: str, channel_id: str, metrics: Dict):
        """
        Envía notificación de posible shadowban
        
        Args:
            platform: Plataforma afectada
            channel_id: ID del canal
            metrics: Métricas que indican el shadowban
        """
        # Obtener umbrales de la configuración
        thresholds = self.alerts_config.get('risk', {}).get('shadowban_probability', 0.7)
        
        # Calcular probabilidad de shadowban
        views_drop = metrics.get('views_drop', 0)
        engagement_drop = metrics.get('engagement_drop', 0)
        distribution_change = metrics.get('distribution_change', 0)
        
        # Fórmula simple para calcular probabilidad
        shadowban_probability = (views_drop + engagement_drop + distribution_change) / 3
        
        if shadowban_probability >= thresholds:
            subject = f"Posible shadowban detectado en {platform}"
            message = (
                f"Se ha detectado una posible restricción de distribución (shadowban) "
                f"en el canal {channel_id} de {platform}.\n\n"
                f"Probabilidad estimada: {shadowban_probability:.2%}"
            )
            
            # Enviar notificación con alta urgencia
            self.send_notification(
                notification_type="shadowban",
                subject=subject,
                message=message,
                data={
                    'platform': platform,
                    'channel_id': channel_id,
                    'views_drop': f"{views_drop:.2%}",
                    'engagement_drop': f"{engagement_drop:.2%}",
                    'distribution_change': f"{distribution_change:.2%}",
                    'probability': f"{shadowban_probability:.2%}"
                },
                urgency='high'
            )
    
    def notify_niche_saturation(self, niche: str, metrics: Dict):
        """
        Envía notificación de saturación de nicho
        
        Args:
            niche: Nicho afectado
            metrics: Métricas que indican la saturación
        """
        # Obtener configuración de saturación
        saturation_config = self.strategy.get('optimization_strategies', {}).get('niche_saturation', {})
        
        # Verificar si las métricas superan los umbrales
        view_decline = metrics.get('view_decline_rate', 0)
        engagement_decline = metrics.get('engagement_decline_rate', 0)
        competition_increase = metrics.get('competition_increase_rate', 0)
        
        # Verificar si alguna métrica supera su umbral
        threshold_exceeded = (
            view_decline > saturation_config.get('metrics', {}).get('view_decline_rate', 0.2) or
            engagement_decline > saturation_config.get('metrics', {}).get('engagement_decline_rate', 0.15) or
            competition_increase > saturation_config.get('metrics', {}).get('competition_increase_rate', 0.3)
        )
        
        if threshold_exceeded:
            subject = f"Saturación detectada en nicho: {niche}"
            message = (
                f"Se ha detectado una posible saturación en el nicho {niche}.\n\n"
                f"Las métricas indican un aumento de competencia y/o disminución de rendimiento."
            )
            
            # Enviar notificación
            self.send_notification(
                notification_type="niche_saturation",
                subject=subject,
                message=message,
                data={
                    'niche': niche,
                    'view_decline': f"{view_decline:.2%}",
                    'engagement_decline': f"{engagement_decline:.2%}",
                    'competition_increase': f"{competition_increase:.2%}"
                },
                urgency='normal'
            )
            
            # Si está configurado para sugerir pivote, añadir recomendaciones
            if saturation_config.get('actions', {}).get('suggest_pivot', False):
                self._suggest_niche_pivot(niche)
    
    def _suggest_niche_pivot(self, current_niche: str):
        """Sugiere nichos alternativos basados en tendencias actuales"""
        # Aquí se implementaría lógica para recomendar nichos alternativos
        # Por ahora, sugerimos nichos predefinidos
        
        alternative_niches = {
            'finance': ['technology', 'health'],
            'health': ['finance', 'humor'],
            'gaming': ['technology', 'humor'],
            'technology': ['finance', 'gaming'],
            'humor': ['gaming', 'health']
        }
        
        suggestions = alternative_niches.get(current_niche, [])
        
        if suggestions:
            subject = f"Sugerencia de pivote para nicho {current_niche}"
            message = (
                f"Basado en la saturación detectada en el nicho {current_niche}, "
                f"se recomienda considerar un pivote hacia los siguientes nichos:\n\n"
                f"- {suggestions[0]}\n"
                f"- {suggestions[1]}\n\n"
                f"Estos nichos muestran mejor potencial de crecimiento y menor saturación."
            )
            
            self.send_notification(
                notification_type="niche_pivot",
                subject=subject,
                message=message,
                data={
                    'current_niche': current_niche,
                    'suggested_niches': suggestions
                },
                urgency='normal'
            )
    
    def notify_performance_drop(self, channel_id: str, platform: str, metrics: Dict):
        """
        Envía notificación de caída de rendimiento
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            metrics: Métricas de rendimiento
        """
        # Obtener umbrales de la configuración
        thresholds = self.alerts_config.get('performance_drop', {})
        
        # Verificar métricas contra umbrales
        alerts = []
        
        if metrics.get('views_drop', 0) > thresholds.get('views', 0.3):
            alerts.append(f"Caída de vistas: {metrics['views_drop']:.2%}")
        
        if metrics.get('engagement_drop', 0) > thresholds.get('engagement', 0.25):
            alerts.append(f"Caída de engagement: {metrics['engagement_drop']:.2%}")
        
        if metrics.get('conversion_drop', 0) > thresholds.get('conversion', 0.2):
            alerts.append(f"Caída de conversión: {metrics['conversion_drop']:.2%}")
        
        if alerts:
            subject = f"Caída de rendimiento en {platform}"
            message = (
                f"Se ha detectado una caída significativa en el rendimiento "
                f"del canal {channel_id} en {platform}:\n\n"
                f"- {alerts[0]}\n"
            )
            
            if len(alerts) > 1:
                for alert in alerts[1:]:
                    message += f"- {alert}\n"
            
            # Determinar urgencia basada en la magnitud de la caída
            max_drop = max(
                metrics.get('views_drop', 0),
                metrics.get('engagement_drop', 0),
                metrics.get('conversion_drop', 0)
            )
            
            urgency = 'high' if max_drop > 0.5 else 'normal'
            
            self.send_notification(
                notification_type="performance_drop",
                subject=subject,
                message=message,
                data={
                    'channel_id': channel_id,
                    'platform': platform,
                    **{k: f"{v:.2%}" for k, v in metrics.items() if k.endswith('_drop')}
                },
                urgency=urgency
            )
    
    def notify_opportunity(self, opportunity_type: str, data: Dict):
        """
        Envía notificación de oportunidad detectada
        
        Args:
            opportunity_type: Tipo de oportunidad (trending_topic, audience_growth, etc.)
            data: Datos de la oportunidad
        """
                # Mapeo de tipos de oportunidad a mensajes
        opportunity_messages = {
            'trending_topic': {
                'subject': 'Tema tendencia detectado',
                'message': 'Se ha detectado un tema en tendencia que podría ser relevante para tu contenido.',
                'urgency': 'normal'
            },
            'audience_growth': {
                'subject': 'Oportunidad de crecimiento de audiencia',
                'message': 'Se ha detectado un segmento de audiencia con alto potencial de crecimiento.',
                'urgency': 'normal'
            },
            'monetization': {
                'subject': 'Oportunidad de monetización',
                'message': 'Se ha detectado una nueva oportunidad para monetizar tu contenido.',
                'urgency': 'high'
            },
            'collaboration': {
                'subject': 'Oportunidad de colaboración',
                'message': 'Se ha identificado un creador potencial para colaboración.',
                'urgency': 'normal'
            },
            'platform_feature': {
                'subject': 'Nueva característica de plataforma',
                'message': 'Se ha lanzado una nueva característica que podría beneficiar a tu canal.',
                'urgency': 'low'
            },
            'content_gap': {
                'subject': 'Brecha de contenido identificada',
                'message': 'Se ha identificado una brecha de contenido que podrías aprovechar.',
                'urgency': 'normal'
            }
        }
        
        # Obtener configuración para el tipo de oportunidad
        opportunity_config = opportunity_messages.get(opportunity_type, {
            'subject': f'Nueva oportunidad: {opportunity_type}',
            'message': 'Se ha detectado una nueva oportunidad para tu contenido.',
            'urgency': 'normal'
        })
        
        # Personalizar mensaje con datos específicos
        subject = opportunity_config['subject']
        message = opportunity_config['message'] + '\n\n'
        
        # Añadir detalles específicos según el tipo de oportunidad
        if opportunity_type == 'trending_topic':
            message += (
                f"Tema: {data.get('topic', 'No especificado')}\n"
                f"Volumen de búsqueda: {data.get('search_volume', 'No disponible')}\n"
                f"Crecimiento: {data.get('growth_rate', '0')}%\n\n"
                f"Este tema está ganando popularidad y se alinea con tu contenido. "
                f"Considera crear contenido relacionado pronto para aprovechar esta tendencia."
            )
        
        elif opportunity_type == 'audience_growth':
            message += (
                f"Segmento: {data.get('segment', 'No especificado')}\n"
                f"Tamaño estimado: {data.get('estimated_size', 'No disponible')}\n"
                f"Tasa de crecimiento: {data.get('growth_rate', '0')}%\n\n"
                f"Este segmento de audiencia está creciendo rápidamente y muestra interés "
                f"en temas relacionados con tu contenido. Considera adaptar tu estrategia "
                f"para atraer a este público."
            )
        
        elif opportunity_type == 'monetization':
            message += (
                f"Tipo: {data.get('type', 'No especificado')}\n"
                f"Potencial estimado: {data.get('estimated_revenue', 'No disponible')}\n\n"
                f"Esta oportunidad de monetización podría aumentar tus ingresos. "
                f"Revisa los detalles y considera implementarla en tu estrategia."
            )
        
        elif opportunity_type == 'collaboration':
            message += (
                f"Creador: {data.get('creator_name', 'No especificado')}\n"
                f"Plataforma: {data.get('platform', 'No disponible')}\n"
                f"Audiencia: {data.get('audience_size', 'No disponible')}\n\n"
                f"Este creador tiene una audiencia similar y complementaria a la tuya. "
                f"Una colaboración podría beneficiar a ambos canales."
            )
        
        elif opportunity_type == 'platform_feature':
            message += (
                f"Plataforma: {data.get('platform', 'No especificado')}\n"
                f"Característica: {data.get('feature_name', 'No disponible')}\n\n"
                f"Esta nueva característica podría ayudarte a mejorar tu alcance o monetización. "
                f"Considera cómo integrarla en tu estrategia de contenido."
            )
        
        elif opportunity_type == 'content_gap':
            message += (
                f"Tema: {data.get('topic', 'No especificado')}\n"
                f"Demanda estimada: {data.get('demand_level', 'No disponible')}\n"
                f"Competencia: {data.get('competition_level', 'No disponible')}\n\n"
                f"Existe una demanda significativa para este tema con poca oferta de contenido. "
                f"Considera crear contenido para cubrir esta brecha."
            )
        
        # Enviar notificación
        self.send_notification(
            notification_type=f"opportunity_{opportunity_type}",
            subject=subject,
            message=message,
            data=data,
            urgency=opportunity_config['urgency']
        )
    
    def notify_task_completion(self, task_type: str, task_data: Dict, success: bool = True):
        """
        Envía notificación de finalización de tarea
        
        Args:
            task_type: Tipo de tarea completada
            task_data: Datos de la tarea
            success: Indica si la tarea se completó con éxito
        """
        # Determinar mensaje según tipo de tarea
        task_messages = {
            'content_creation': 'Creación de contenido',
            'content_optimization': 'Optimización de contenido',
            'analytics_report': 'Informe de análisis',
            'audience_research': 'Investigación de audiencia',
            'competitor_analysis': 'Análisis de competencia',
            'trend_research': 'Investigación de tendencias',
            'monetization_setup': 'Configuración de monetización',
            'engagement_campaign': 'Campaña de engagement',
            'data_backup': 'Respaldo de datos'
        }
        
        task_name = task_messages.get(task_type, f'Tarea: {task_type}')
        
        # Crear asunto y mensaje
        if success:
            subject = f"✅ {task_name} completada"
            message = f"La tarea de {task_name.lower()} se ha completado con éxito."
        else:
            subject = f"❌ {task_name} fallida"
            message = f"La tarea de {task_name.lower()} ha fallado."
        
        # Añadir detalles específicos
        if 'name' in task_data:
            message += f"\nNombre: {task_data['name']}"
        
        if 'duration' in task_data:
            message += f"\nDuración: {task_data['duration']} segundos"
        
        if not success and 'error' in task_data:
            message += f"\n\nError: {task_data['error']}"
        
        # Determinar urgencia
        urgency = 'high' if not success else 'low'
        
        # Enviar notificación
        self.send_notification(
            notification_type=f"task_{task_type}",
            subject=subject,
            message=message,
            data=task_data,
            urgency=urgency
        )
    
    def notify_content_performance(self, content_id: str, platform: str, metrics: Dict):
        """
        Envía notificación sobre el rendimiento de un contenido específico
        
        Args:
            content_id: ID del contenido
            platform: Plataforma
            metrics: Métricas de rendimiento
        """
        # Determinar si el rendimiento es bueno o malo
        performance_threshold = self.alerts_config.get('content_performance', {}).get('threshold', 0.2)
        
        # Calcular desviación del rendimiento esperado
        views_performance = metrics.get('views_vs_expected', 0)
        engagement_performance = metrics.get('engagement_vs_expected', 0)
        conversion_performance = metrics.get('conversion_vs_expected', 0)
        
        # Promedio de desviación
        avg_performance = (views_performance + engagement_performance + conversion_performance) / 3
        
        # Determinar tipo de rendimiento
        if avg_performance > performance_threshold:
            performance_type = "positive"
            subject = f"📈 Contenido con rendimiento excepcional en {platform}"
            message = f"El contenido está superando las expectativas en {platform}."
            urgency = "normal"
        elif avg_performance < -performance_threshold:
            performance_type = "negative"
            subject = f"📉 Contenido con bajo rendimiento en {platform}"
            message = f"El contenido está por debajo de las expectativas en {platform}."
            urgency = "normal"
        else:
            # Rendimiento dentro de lo esperado, no notificar
            return
        
        # Añadir detalles de métricas
        message += "\n\nMétricas de rendimiento:"
        
        if 'views' in metrics:
            message += f"\n- Vistas: {metrics['views']}"
            if 'views_vs_expected' in metrics:
                deviation = metrics['views_vs_expected'] * 100
                direction = "por encima" if deviation > 0 else "por debajo"
                message += f" ({abs(deviation):.1f}% {direction} de lo esperado)"
        
        if 'engagement_rate' in metrics:
            message += f"\n- Tasa de engagement: {metrics['engagement_rate']:.2%}"
            if 'engagement_vs_expected' in metrics:
                deviation = metrics['engagement_vs_expected'] * 100
                direction = "por encima" if deviation > 0 else "por debajo"
                message += f" ({abs(deviation):.1f}% {direction} de lo esperado)"
        
        if 'conversion_rate' in metrics:
            message += f"\n- Tasa de conversión: {metrics['conversion_rate']:.2%}"
            if 'conversion_vs_expected' in metrics:
                deviation = metrics['conversion_vs_expected'] * 100
                direction = "por encima" if deviation > 0 else "por debajo"
                message += f" ({abs(deviation):.1f}% {direction} de lo esperado)"
        
        # Añadir recomendaciones según el tipo de rendimiento
        if performance_type == "positive":
            message += "\n\nRecomendaciones:"
            message += "\n- Considera crear más contenido similar"
            message += "\n- Analiza qué factores contribuyeron al éxito"
            message += "\n- Promociona este contenido en otras plataformas"
        else:
            message += "\n\nRecomendaciones:"
            message += "\n- Revisa el título, miniaturas y optimización"
            message += "\n- Considera ajustar la estrategia de distribución"
            message += "\n- Evalúa si el contenido se alinea con los intereses de tu audiencia"
        
        # Enviar notificación
        self.send_notification(
            notification_type=f"content_performance_{performance_type}",
            subject=subject,
            message=message,
            data={
                'content_id': content_id,
                'platform': platform,
                **metrics
            },
            urgency=urgency
        )
    
    def notify_system_status(self, status: str, details: Dict = None):
        """
        Envía notificación sobre el estado del sistema
        
        Args:
            status: Estado del sistema (ok, warning, error, critical)
            details: Detalles adicionales
        """
        details = details or {}
        
        # Mapeo de estados a mensajes y urgencia
        status_config = {
            'ok': {
                'subject': '✅ Sistema funcionando correctamente',
                'message': 'Todos los componentes del sistema están funcionando correctamente.',
                'urgency': 'low'
            },
            'warning': {
                'subject': '⚠️ Advertencia del sistema',
                'message': 'Se han detectado problemas menores en el sistema.',
                'urgency': 'normal'
            },
            'error': {
                'subject': '❌ Error en el sistema',
                'message': 'Se han detectado errores en el sistema que requieren atención.',
                'urgency': 'high'
            },
            'critical': {
                'subject': '🚨 Error crítico en el sistema',
                'message': 'Se han detectado errores críticos que afectan el funcionamiento del sistema.',
                'urgency': 'critical'
            }
        }
        
        # Obtener configuración para el estado
        config = status_config.get(status, status_config['warning'])
        
        # Crear mensaje
        message = config['message'] + '\n\n'
        
        # Añadir componentes afectados
        if 'affected_components' in details:
            message += "Componentes afectados:\n"
            for component in details['affected_components']:
                message += f"- {component}\n"
            message += "\n"
        
        # Añadir errores
        if 'errors' in details:
            message += "Errores detectados:\n"
            for error in details['errors']:
                message += f"- {error}\n"
            message += "\n"
        
        # Añadir uso de recursos
        if 'resource_usage' in details:
            message += "Uso de recursos:\n"
            for resource, usage in details['resource_usage'].items():
                message += f"- {resource}: {usage}\n"
            message += "\n"
        
        # Añadir acciones recomendadas
        if 'recommended_actions' in details:
            message += "Acciones recomendadas:\n"
            for action in details['recommended_actions']:
                message += f"- {action}\n"
        
        # Enviar notificación
        self.send_notification(
            notification_type=f"system_status_{status}",
            subject=config['subject'],
            message=message,
            data=details,
            urgency=config['urgency']
        )
    
    def get_notification_history(self, limit: int = 50, notification_type: str = None, 
                               urgency: str = None, start_date: str = None, 
                               end_date: str = None) -> List[Dict]:
        """
        Obtiene historial de notificaciones con filtros opcionales
        
        Args:
            limit: Número máximo de notificaciones a devolver
            notification_type: Filtrar por tipo de notificación
            urgency: Filtrar por nivel de urgencia
            start_date: Fecha de inicio (formato ISO)
            end_date: Fecha de fin (formato ISO)
            
        Returns:
            Lista de notificaciones filtradas
        """
        # Convertir fechas a objetos datetime si están presentes
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                logger.warning(f"Formato de fecha de inicio no válido: {start_date}")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
            except ValueError:
                logger.warning(f"Formato de fecha de fin no válido: {end_date}")
        
        # Filtrar notificaciones
        filtered_notifications = []
        
        for notification in reversed(self.notification_history):
            # Filtrar por tipo
            if notification_type and not notification['type'].startswith(notification_type):
                continue
            
            # Filtrar por urgencia
            if urgency and notification['urgency'] != urgency:
                continue
            
            # Filtrar por fecha de inicio
            if start_datetime:
                notification_datetime = datetime.datetime.fromisoformat(notification['timestamp'])
                if notification_datetime < start_datetime:
                    continue
            
            # Filtrar por fecha de fin
            if end_datetime:
                notification_datetime = datetime.datetime.fromisoformat(notification['timestamp'])
                if notification_datetime > end_datetime:
                    continue
            
            # Añadir a resultados filtrados
            filtered_notifications.append(notification)
            
            # Limitar resultados
            if len(filtered_notifications) >= limit:
                break
        
        return filtered_notifications
    
    def update_notification_channels(self, channel_updates: Dict) -> bool:
        """
        Actualiza la configuración de canales de notificación
        
        Args:
            channel_updates: Diccionario con actualizaciones de configuración
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            # Actualizar cada canal especificado
            for channel, config in channel_updates.items():
                if channel in self.notification_channels:
                    # Actualizar configuración existente
                    self.notification_channels[channel].update(config)
                    logger.info(f"Canal de notificación actualizado: {channel}")
                else:
                    # Añadir nuevo canal
                    self.notification_channels[channel] = config
                    logger.info(f"Nuevo canal de notificación añadido: {channel}")
            
            # Guardar configuración actualizada
            config_file = os.path.join('config', 'notification_channels.json')
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.notification_channels, f, indent=4)
            
            logger.info("Configuración de canales de notificación guardada")
            return True
        
        except Exception as e:
            logger.error(f"Error al actualizar canales de notificación: {str(e)}")
            return False

# Ejemplo de uso
if __name__ == "__main__":
    # Crear directorio de logs si no existe
    os.makedirs('logs', exist_ok=True)
    
    # Inicializar notificador
    notifier = Notifier()
    
    # Ejemplo de notificación
    notifier.send_notification(
        notification_type="test",
        subject="Prueba de notificación",
        message="Este es un mensaje de prueba del sistema de notificaciones.",
        data={"test_key": "test_value"},
        urgency="normal"
    )
    
    print("Notificación enviada correctamente")
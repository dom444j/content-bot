"""
Notifier - Sistema de notificaciones y alertas

Este módulo gestiona las notificaciones del sistema:
- Alertas de shadowbans y problemas de distribución
- Notificaciones de saturación de nichos
- Alertas de rendimiento (caídas de CTR, oportunidades)
- Notificaciones de tareas completadas
- Soporte para niveles de alerta (info, warning, critical)
- Escalado de alertas, supresión de duplicados y reintentos
- Personalización por usuario, agrupación inteligente y webhooks personalizados
"""

import os
import sys
import json
import logging
import datetime
import smtplib
import requests
import time
import random
import threading
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from urllib.parse import urlparse

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
    Implementa Singleton, niveles de alerta, personalización por usuario,
    agrupación inteligente y webhooks personalizados.
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
        
        # Cargar configuración de usuarios
        self.user_configs_file = os.path.join('config', 'user_configs.json')
        self.user_configs = self._load_user_configs()
        
        # Configuración de alertas
        self.alerts_config = self.strategy.get('analytics', {}).get('alerts', {})
        
        # Configuración de niveles de alerta
        self.alert_levels_config = {
            'info': {
                'priority': 3,
                'channels': ['console', 'log'],
                'emoji': 'ℹ️',
                'color': 5592575,  # Azul (Discord)
                'prefix': '[INFO]',
                'suppression_window': 3600,  # 1 hora
                'escalation': None
            },
            'warning': {
                'priority': 2,
                'channels': ['console', 'log', 'email', 'telegram'],
                'emoji': '⚠️',
                'color': 16776960,  # Amarillo (Discord)
                'prefix': '[WARNING]',
                'suppression_window': 1800,  # 30 minutos
                'escalation': {
                    'level': 'critical',
                    'delay': 3600  # Escalar a critical tras 1 hora
                }
            },
            'critical': {
                'priority': 1,
                'channels': ['console', 'log', 'email', 'telegram', 'discord', 'custom_webhook'],
                'emoji': '🚨',
                'color': 16711680,  # Rojo (Discord)
                'prefix': '[CRITICAL]',
                'suppression_window': 900,  # 15 minutos
                'escalation': None
            }
        }
        
        # Configuración de agrupación inteligente
        self.grouping_config = {
            'time_window': self.alerts_config.get('grouping', {}).get('time_window', 300),  # 5 minutos
            'max_group_size': self.alerts_config.get('grouping', {}).get('max_group_size', 10),
            'type_similarity_threshold': 0.8
        }
        
        # Historial de notificaciones
        self.notification_history = []
        
        # Cache para supresión de notificaciones
        self.suppression_cache = defaultdict(list)
        
        # Métricas de notificaciones
        self.notification_metrics = {
            'total_sent': 0,
            'success_by_channel': defaultdict(int),
            'failures_by_channel': defaultdict(int),
            'escalated_notifications': 0,
            'suppressed_notifications': 0,
            'grouped_notifications': 0,
            'average_delivery_time': 0,
            'user_metrics': defaultdict(lambda: {
                'total_sent': 0,
                'success_by_channel': defaultdict(int),
                'failures_by_channel': defaultdict(int)
            })
        }
        
        # Configuración de reintentos
        self.max_retries = 3
        self.base_retry_delay = 5
        
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
            'custom_webhook': {
                'enabled': False,
                'webhooks': [],  # Lista de {url, format, headers}
                'timeout': 10
            },
            'console': {
                'enabled': True
            },
            'log': {
                'enabled': True
            }
        }
        
        # Configuración de llamadas de voz (para escalado crítico)
        self.call_config = {
            'enabled': False,
            'twilio_account_sid': '',
            'twilio_auth_token': '',
            'from_number': '',
            'to_numbers': []
        }
        
        # Inicializar métricas
        self._initialize_metrics()
        
        # Cargar plantillas de notificación
        self.notification_templates = self._load_notification_templates()
        
        # Validar configuración
        self._validate_channels_config()
        self._validate_user_configs()
        
        # Iniciar hilo para escalado de alertas
        threading.Thread(target=self._alert_escalation_loop, daemon=True).start()
        
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
    
    def _load_user_configs(self) -> Dict:
        """Carga la configuración de usuarios desde el archivo JSON"""
        try:
            if os.path.exists(self.user_configs_file):
                with open(self.user_configs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración de usuarios no encontrado: {self.user_configs_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración de usuarios: {str(e)}")
            return {}
    
    def _validate_channels_config(self):
        """Valida la configuración de los canales de notificación"""
        for channel, config in self.notification_channels.items():
            if config.get('enabled', False):
                if channel == 'email':
                    required = ['smtp_server', 'smtp_port', 'username', 'password', 'recipients']
                    if not all(key in config for key in required) or not config['recipients']:
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal email: configuración incompleta")
                elif channel == 'telegram':
                    required = ['bot_token', 'chat_id']
                    if not all(key in config for key in required):
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal telegram: configuración incompleta")
                elif channel == 'discord':
                    if 'webhook_url' not in config or not config['webhook_url']:
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal discord: configuración incompleta")
                elif channel == 'custom_webhook':
                    if not config.get('webhooks') or not all(
                        'url' in wh and wh['url'] for wh in config['webhooks']
                    ):
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal custom_webhook: configuración incompleta")
    
    def _validate_user_configs(self):
        """Valida la configuración de usuarios"""
        for user_id, config in list(self.user_configs.items()):
            if not config.get('channels') or not config.get('alert_preferences'):
                logger.warning(f"Configuración inválida para usuario {user_id}, eliminando")
                del self.user_configs[user_id]
                continue
            
            # Validar canales
            for channel in config['channels']:
                if channel not in self.notification_channels:
                    logger.warning(f"Canal {channel} no soportado para usuario {user_id}")
                    config['channels'].remove(channel)
            
            # Validar niveles de alerta
            for level in config['alert_preferences'].get('levels', {}):
                if level not in self.alert_levels_config:
                    logger.warning(f"Nivel {level} no soportado para usuario {user_id}")
                    del config['alert_preferences']['levels'][level]
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calcula el retraso para reintentos con backoff exponencial"""
        return self.base_retry_delay * (2 ** retry_count) + random.uniform(0, 0.1)
    
    def _get_user_alert_level(self, user_id: str, level: str) -> str:
        """Obtiene el nivel de alerta personalizado para un usuario"""
        if user_id not in self.user_configs:
            return level
        
        user_level_map = self.user_configs[user_id].get('alert_preferences', {}).get('levels', {})
        return user_level_map.get(level, level)
    
    def _get_user_channels(self, user_id: str, level: str) -> List[str]:
        """Obtiene los canales preferidos para un usuario y nivel"""
        if user_id not in self.user_configs:
            return self.alert_levels_config[level]['channels']
        
        user_channels = self.user_configs[user_id].get('channels', [])
        level_channels = self.alert_levels_config[level]['channels']
        return [ch for ch in user_channels if ch in level_channels and self.notification_channels[ch].get('enabled', False)]
    
    def _apply_user_template(self, user_id: str, message: str, subject: str, data: Dict) -> tuple:
        """Aplica una plantilla personalizada para el mensaje de un usuario"""
        if user_id not in self.user_configs:
            return subject, message
        
        template = self.user_configs[user_id].get('message_template', {
            'subject': '{subject}',
            'message': '{message}\n\nDetalles:\n{details}'
        })
        
        details = '\n'.join(f"- {k}: {v}" for k, v in data.items()) if data else 'Ninguno'
        new_subject = template['subject'].format(subject=subject, **data)
        new_message = template['message'].format(message=message, details=details, **data)
        
        return new_subject, new_message
    
    def send_notification(self, notification_type: str, subject: str, message: str, 
                         data: Dict = None, level: str = 'info', 
                         channels: List[str] = None, user_id: str = None) -> str:
        """
        Envía una notificación a través de los canales configurados
        """
        if level not in self.alert_levels_config:
            logger.warning(f"Nivel de alerta no válido: {level}, usando 'info'")
            level = 'info'
        
        # Aplicar personalización por usuario
        level = self._get_user_alert_level(user_id, level) if user_id else level
        subject, message = self._apply_user_template(user_id, message, subject, data or {})
        
        # Generar ID de notificación
        notification_id = f"{notification_type}_{int(datetime.datetime.now().timestamp())}"
        
        # Verificar supresión
        if self._is_notification_suppressed(notification_type, level):
            self.notification_metrics['suppressed_notifications'] += 1
            logger.info(f"Notificación {notification_id} suprimida por política de supresión")
            return notification_id
        
        # Crear objeto de notificación
        notification = {
            'id': notification_id,
            'type': notification_type,
            'subject': subject,
            'message': message,
            'data': data or {},
            'level': level,
            'timestamp': datetime.datetime.now().isoformat(),
            'channels_sent': [],
            'delivery_time': 0,
            'escalation_status': 'none',
            'user_id': user_id
        }
        
        # Verificar si debe suprimirse por duplicado
        if self._should_suppress_duplicate(notification):
            if hasattr(self, 'notification_metrics'):
                self.notification_metrics['suppressed_count'] += 1
            logger.info(f"Notificación {notification_id} suprimida por duplicado")
            return notification_id
        
        # Determinar canales a utilizar
        level_config = self.alert_levels_config[level]
        if channels is None:
            channels = self._get_user_channels(user_id, level) if user_id else level_config['channels']
        
        # Registrar en cache de supresión
        self.suppression_cache[notification_type].append({
            'timestamp': datetime.datetime.now().timestamp(),
            'notification_id': notification_id
        })
        
        # Enviar a canales configurados
        results = self._send_to_channels(notification, level_config)
        
        # Verificar si debe escalarse
        if self._should_escalate(notification):
            self._escalate_alert(notification)
        
        # Enviar a cada canal
        start_time = time.time()
        for channel in channels:
            if channel not in self.notification_channels or not self.notification_channels[channel].get('enabled', False):
                logger.warning(f"Canal de notificación no disponible o deshabilitado: {channel}")
                continue
            
            try:
                self._send_to_channel(channel, notification, user_id)
                notification['channels_sent'].append(channel)
                self.notification_metrics['success_by_channel'][channel] += 1
                if user_id:
                    self.notification_metrics['user_metrics'][user_id]['success_by_channel'][channel] += 1
                    self.notification_metrics['user_metrics'][user_id]['total_sent'] += 1
            except Exception as e:
                logger.error(f"Error al enviar notificación a {channel}: {str(e)}")
                self.notification_metrics['failures_by_channel'][channel] += 1
                if user_id:
                    self.notification_metrics['user_metrics'][user_id]['failures_by_channel'][channel] += 1
                threading.Thread(target=self._retry_channel, 
                               args=(channel, notification, 0, user_id), 
                               daemon=True).start()
        
        # Calcular tiempo de entrega
        notification['delivery_time'] = time.time() - start_time
        self._update_delivery_metrics(notification['delivery_time'])
        
        # Registrar en historial
        self.notification_history.append(notification)
        self.notification_metrics['total_sent'] += 1
        if user_id:
            self.notification_metrics['user_metrics'][user_id]['total_sent'] += 1
        
        # Programar escalado si aplica
        if level_config['escalation']:
            threading.Thread(target=self._schedule_escalation, 
                           args=(notification_id, level_config['escalation']), 
                           daemon=True).start()
        
        logger.info(f"Notificación {notification_id} enviada a {len(notification['channels_sent'])} canales")
        return notification_id
    
    def _retry_channel(self, channel: str, notification: Dict, retry_count: int, user_id: str = None):
        """Reintenta enviar una notificación a un canal fallido"""
        if retry_count >= self.max_retries:
            logger.error(f"Reintentos agotados para canal {channel}, notificación {notification['id']}")
            return
        
        try:
            delay = self._calculate_retry_delay(retry_count)
            time.sleep(delay)
            self._send_to_channel(channel, notification, user_id)
            notification['channels_sent'].append(f"{channel}_retry_{retry_count + 1}")
            self.notification_metrics['success_by_channel'][channel] += 1
            if user_id:
                self.notification_metrics['user_metrics'][user_id]['success_by_channel'][channel] += 1
            logger.info(f"Reintento exitoso para canal {channel}, notificación {notification['id']}")
        except Exception as e:
            logger.warning(f"Reintento {retry_count + 1}/{self.max_retries} fallido para canal {channel}: {str(e)}")
            threading.Thread(target=self._retry_channel, 
                           args=(channel, notification, retry_count + 1, user_id), 
                           daemon=True).start()
    
    def _update_delivery_metrics(self, delivery_time: float):
        """Actualiza las métricas de tiempo de entrega"""
        total_notifications = self.notification_metrics['total_sent']
        if total_notifications == 0:
            return
        
        current_avg = self.notification_metrics['average_delivery_time']
        new_avg = ((current_avg * (total_notifications - 1)) + delivery_time) / total_notifications
        self.notification_metrics['average_delivery_time'] = new_avg
    
    def _is_notification_suppressed(self, notification_type: str, level: str) -> bool:
        """Verifica si una notificación debe ser suprimida"""
        if notification_type not in self.suppression_cache:
            return False
        
        suppression_window = self.alert_levels_config[level]['suppression_window']
        current_time = datetime.datetime.now().timestamp()
        
        self.suppression_cache[notification_type] = [
            entry for entry in self.suppression_cache[notification_type]
            if current_time - entry['timestamp'] <= suppression_window
        ]
        
        return len(self.suppression_cache[notification_type]) > 0
    
    def _schedule_escalation(self, notification_id: str, escalation_config: Dict):
        """Programa el escalado de una notificación"""
        time.sleep(escalation_config['delay'])
        
        for notification in self.notification_history:
            if notification['id'] == notification_id and notification['escalation_status'] == 'none':
                self._escalate_alert(notification, escalation_config['level'])
                break
    
    def _escalate_alert(self, notification: Dict, new_level: str = None):
        """Escala una notificación a un nivel superior"""
        if not new_level:
            # Determinar nivel de escalado
            current_level = notification['level']
            if current_level == 'info':
                new_level = 'warning'
            elif current_level == 'warning':
                new_level = 'critical'
            else:
                # Ya está en nivel crítico, no escalar más
                return
        
        notification['escalation_status'] = 'escalated'
        if hasattr(self, 'notification_metrics'):
            self.notification_metrics['escalated_notifications'] += 1
        
        new_subject = f"[ESCALATED] {notification['subject']}"
        new_message = (
            f"⚠️ ALERTA ESCALADA desde {notification['level'].upper()} a {new_level.upper()}\n\n"
            f"Original: {notification['message']}\n\n"
            f"Razón: No resuelta tras {self.alert_levels_config[notification['level']]['escalation']['delay']} segundos"
        )
        
        # Si es crítico, considerar llamada de voz
        if new_level == 'critical' and self.call_config.get('enabled'):
            threading.Thread(target=self._send_voice_call, 
                           args=(notification,), 
                           daemon=True).start()
        
        self.send_notification(
            notification_type=f"{notification['type']}_escalated",
            subject=new_subject,
            message=new_message,
            data={
                **notification['data'],
                'original_level': notification['level'],
                'escalation_time': datetime.datetime.now().isoformat()
            },
            level=new_level,
            user_id=notification['user_id']
        )
        logger.info(f"Notificación {notification['id']} escalada a {new_level}")
    
    def _alert_escalation_loop(self):
        """Bucle para manejar escalado periódico"""
        while True:
            try:
                current_time = datetime.datetime.now().timestamp()
                for notification in self.notification_history:
                    if notification['escalation_status'] != 'none':
                        continue
                    
                    level_config = self.alert_levels_config.get(notification['level'], {})
                    escalation = level_config.get('escalation')
                    if not escalation:
                        continue
                    
                    notification_time = datetime.datetime.fromisoformat(notification['timestamp']).timestamp()
                    if current_time - notification_time >= escalation['delay']:
                        self._escalate_alert(notification, escalation['level'])
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error en bucle de escalado: {str(e)}")
                time.sleep(60)
    
    def _send_to_channel(self, channel: str, notification: Dict, user_id: str = None):
        """Envía una notificación a un canal específico"""
        level_config = self.alert_levels_config[notification['level']]
        if channel == 'email':
            self._send_email(notification, level_config, user_id)
        elif channel == 'telegram':
            self._send_telegram(notification, level_config, user_id)
        elif channel == 'discord':
            self._send_discord(notification, level_config)
        elif channel == 'custom_webhook':
            self._send_custom_webhook(notification, level_config, user_id)
        elif channel == 'console':
            self._send_console(notification, level_config)
        elif channel == 'log':
            self._send_log(notification, level_config)
        else:
            logger.warning(f"Canal de notificación desconocido: {channel}")
    
    def _send_email(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación por email"""
        config = self.notification_channels['email']
        recipients = config['recipients']
        
        if user_id and user_id in self.user_configs:
            user_recipients = self.user_configs[user_id].get('email_recipients', [])
            if user_recipients:
                recipients = user_recipients
        
        msg = MIMEMultipart()
        msg['From'] = config['username']
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"{level_config['prefix']} {notification['subject']}"
        
        body = f"{level_config['emoji']} {notification['message']}\n\n"
        
        if notification['data']:
            body += "Detalles adicionales:\n"
            for key, value in notification['data'].items():
                body += f"- {key}: {value}\n"
        
        body += f"\nFecha: {notification['timestamp']}"
        body += f"\nID: {notification['id']}"
        if notification['escalation_status'] == 'escalated':
            body += f"\nEstado: Escalada desde {notification['data'].get('original_level', 'desconocido')}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            logger.info(f"Email enviado a {len(recipients)} destinatarios")
        except Exception as e:
            logger.error(f"Error al enviar email: {str(e)}")
            raise
    
    def _send_telegram(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación por Telegram"""
        config = self.notification_channels['telegram']
        chat_id = config['chat_id']
        
        if user_id and user_id in self.user_configs:
            user_chat_id = self.user_configs[user_id].get('telegram_chat_id')
            if user_chat_id:
                chat_id = user_chat_id
        
        message = f"{level_config['emoji']} *{notification['subject']}*\n\n{notification['message']}\n"
        
        if notification['data']:
            message += "\n*Detalles adicionales:*\n"
            for key, value in notification['data'].items():
                message += f"- {key}: {value}\n"
        
        if notification['escalation_status'] == 'escalated':
            message += f"\n*Escalada desde*: {notification['data'].get('original_level', 'desconocido')}"
        
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Error al enviar mensaje a Telegram: {response.text}")
            raise Exception(f"Error de Telegram: {response.status_code}")
        
        logger.info("Mensaje enviado a Telegram correctamente")
    
    def _send_discord(self, notification: Dict, level_config: Dict):
        """Envía notificación por Discord"""
        config = self.notification_channels['discord']
        
        embed = {
            'title': f"{level_config['emoji']} {notification['subject']}",
            'description': notification['message'],
            'color': level_config['color'],
            'fields': [],
            'timestamp': notification['timestamp']
        }
        
        if notification['data']:
            for key, value in notification['data'].items():
                embed['fields'].append({
                    'name': key,
                    'value': str(value),
                    'inline': True
                })
        
        if notification['escalation_status'] == 'escalated':
            embed['fields'].append({
                'name': 'Estado',
                'value': f"Escalada desde {notification['data'].get('original_level', 'desconocido')}",
                'inline': True
            })
        
        payload = {
            'embeds': [embed],
            'username': 'Content Bot Notifier'
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        
        if response.status_code != 204:
            logger.error(f"Error al enviar mensaje a Discord: {response.text}")
            raise Exception(f"Error de Discord: {response.status_code}")
        
        logger.info("Mensaje enviado a Discord correctamente")
    
    def _send_custom_webhook(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación a webhooks personalizados"""
        config = self.notification_channels['custom_webhook']
        webhooks = config['webhooks']
        
        if user_id and user_id in self.user_configs:
            user_webhooks = self.user_configs[user_id].get('custom_webhooks', [])
            if user_webhooks:
                webhooks = user_webhooks
        
        for webhook in webhooks:
                        url = webhook['url']
            format_type = webhook.get('format', 'json')
            headers = webhook.get('headers', {})
            timeout = config.get('timeout', 10)
            
            # Preparar payload según formato
            if format_type == 'json':
                payload = {
                    'id': notification['id'],
                    'type': notification['type'],
                    'subject': notification['subject'],
                    'message': notification['message'],
                    'level': notification['level'],
                    'timestamp': notification['timestamp'],
                    'data': notification['data'],
                    'emoji': level_config['emoji'],
                    'user_id': notification['user_id']
                }
                headers['Content-Type'] = 'application/json'
                data = json.dumps(payload)
            elif format_type == 'form':
                payload = {
                    'notification_id': notification['id'],
                    'notification_type': notification['type'],
                    'subject': notification['subject'],
                    'message': notification['message'],
                    'level': notification['level'],
                    'timestamp': notification['timestamp'],
                    'user_id': notification['user_id'] or ''
                }
                # Añadir datos adicionales
                for key, value in notification['data'].items():
                    payload[f'data_{key}'] = str(value)
                data = payload
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
            elif format_type == 'xml':
                root = ET.Element('notification')
                ET.SubElement(root, 'id').text = notification['id']
                ET.SubElement(root, 'type').text = notification['type']
                ET.SubElement(root, 'subject').text = notification['subject']
                ET.SubElement(root, 'message').text = notification['message']
                ET.SubElement(root, 'level').text = notification['level']
                ET.SubElement(root, 'timestamp').text = notification['timestamp']
                ET.SubElement(root, 'emoji').text = level_config['emoji']
                if notification['user_id']:
                    ET.SubElement(root, 'user_id').text = notification['user_id']
                
                # Añadir datos adicionales
                data_elem = ET.SubElement(root, 'data')
                for key, value in notification['data'].items():
                    item = ET.SubElement(data_elem, 'item')
                    ET.SubElement(item, 'key').text = key
                    ET.SubElement(item, 'value').text = str(value)
                
                data = ET.tostring(root, encoding='utf-8')
                headers['Content-Type'] = 'application/xml'
            else:
                # Formato desconocido, usar JSON por defecto
                payload = {
                    'id': notification['id'],
                    'subject': notification['subject'],
                    'message': notification['message'],
                    'level': notification['level'],
                    'timestamp': notification['timestamp']
                }
                headers['Content-Type'] = 'application/json'
                data = json.dumps(payload)
            
            # Enviar webhook
            response = requests.post(url, data=data, headers=headers, timeout=timeout)
            
            if response.status_code >= 400:
                logger.error(f"Error al enviar webhook a {url}: {response.status_code} {response.text}")
                raise Exception(f"Error de webhook: {response.status_code}")
            
            logger.info(f"Webhook enviado a {url} correctamente")
    
    def _send_console(self, notification: Dict, level_config: Dict):
        """Muestra notificación en consola"""
        prefix = level_config['prefix']
        emoji = level_config['emoji']
        
        # Formatear mensaje para consola
        console_message = f"\n{prefix} {emoji} {notification['subject']}\n"
        console_message += f"{'-' * 50}\n"
        console_message += f"{notification['message']}\n"
        
        if notification['data']:
            console_message += "\nDetalles adicionales:\n"
            for key, value in notification['data'].items():
                console_message += f"- {key}: {value}\n"
        
        console_message += f"\nFecha: {notification['timestamp']}"
        console_message += f"\nID: {notification['id']}"
        
        if notification['user_id']:
            console_message += f"\nUsuario: {notification['user_id']}"
        
        if notification['escalation_status'] == 'escalated':
            console_message += f"\nEstado: Escalada desde {notification['data'].get('original_level', 'desconocido')}"
        
        console_message += f"\n{'-' * 50}\n"
        
        # Imprimir en consola
        print(console_message)
    
    def _send_log(self, notification: Dict, level_config: Dict):
        """Registra notificación en el log"""
        log_message = f"{level_config['prefix']} {notification['subject']}: {notification['message']}"
        if notification['user_id']:
            log_message += f" (Usuario: {notification['user_id']})"
        
        if notification['level'] == 'critical':
            logger.critical(log_message)
        elif notification['level'] == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_push_notification(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación push a dispositivos móviles"""
        try:
            # Verificar si Firebase está configurado
            if not hasattr(self, 'firebase_config') or not self.firebase_config.get('enabled'):
                logger.warning("Firebase no está configurado para notificaciones push")
                return
            
            # Importar Firebase Admin SDK
            import firebase_admin
            from firebase_admin import messaging
            
            # Inicializar Firebase si no está inicializado
            if not hasattr(self, 'firebase_app'):
                cred = firebase_admin.credentials.Certificate(self.firebase_config['credentials_file'])
                self.firebase_app = firebase_admin.initialize_app(cred)
            
            # Determinar tokens de dispositivo
            device_tokens = []
            
            if user_id and user_id in self.user_configs:
                user_tokens = self.user_configs[user_id].get('device_tokens', [])
                if user_tokens:
                    device_tokens.extend(user_tokens)
            else:
                # Usar tokens predeterminados si no hay usuario específico
                device_tokens.extend(self.firebase_config.get('default_tokens', []))
            
            if not device_tokens:
                logger.warning("No hay tokens de dispositivo para enviar notificación push")
                return
            
            # Crear mensaje
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=f"{level_config['emoji']} {notification['subject']}",
                    body=notification['message']
                ),
                data={
                    'notification_id': notification['id'],
                    'notification_type': notification['type'],
                    'level': notification['level'],
                    'timestamp': notification['timestamp']
                },
                tokens=device_tokens,
            )
            
            # Enviar mensaje
            response = messaging.send_multicast(message)
            logger.info(f"Notificación push enviada a {response.success_count} de {len(device_tokens)} dispositivos")
            
            # Registrar fallos
            if response.failure_count > 0:
                for idx, resp in enumerate(response.responses):
                    if not resp.success:
                        logger.error(f"Error al enviar push a token {device_tokens[idx]}: {resp.exception}")
        
        except ImportError:
            logger.error("No se pudo importar Firebase Admin SDK. Instale con: pip install firebase-admin")
        except Exception as e:
            logger.error(f"Error al enviar notificación push: {str(e)}")
            raise
    
    def _send_telegram(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación por Telegram"""
        config = self.notification_channels['telegram']
        chat_id = config['chat_id']
        
        if user_id and user_id in self.user_configs:
            user_chat_id = self.user_configs[user_id].get('telegram_chat_id')
            if user_chat_id:
                chat_id = user_chat_id
        
        message = f"{level_config['emoji']} *{notification['subject']}*\n\n{notification['message']}\n"
        
        if notification['data']:
            message += "\n*Detalles adicionales:*\n"
            for key, value in notification['data'].items():
                message += f"- {key}: {value}\n"
        
        if notification['escalation_status'] == 'escalated':
            message += f"\n*Escalada desde*: {notification['data'].get('original_level', 'desconocido')}"
        
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Error al enviar mensaje a Telegram: {response.text}")
            raise Exception(f"Error de Telegram: {response.status_code}")
        
        logger.info("Mensaje enviado a Telegram correctamente")
    
    def _send_discord(self, notification: Dict, level_config: Dict):
        """Envía notificación por Discord"""
        config = self.notification_channels['discord']
        
        embed = {
            'title': f"{level_config['emoji']} {notification['subject']}",
            'description': notification['message'],
            'color': level_config['color'],
            'fields': [],
            'timestamp': notification['timestamp']
        }
        
        if notification['data']:
            for key, value in notification['data'].items():
                embed['fields'].append({
                    'name': key,
                    'value': str(value),
                    'inline': True
                })
        
        if notification['escalation_status'] == 'escalated':
            embed['fields'].append({
                'name': 'Estado',
                'value': f"Escalada desde {notification['data'].get('original_level', 'desconocido')}",
                'inline': True
            })
        
        payload = {
            'embeds': [embed],
            'username': 'Content Bot Notifier'
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        
        if response.status_code != 204:
            logger.error(f"Error al enviar mensaje a Discord: {response.text}")
            raise Exception(f"Error de Discord: {response.status_code}")
        
        logger.info("Mensaje enviado a Discord correctamente")
    
    def _send_teams(self, notification: Dict, level_config: Dict):
        """Envía notificación a Microsoft Teams"""
        try:
            # Verificar si Teams está configurado
            if not hasattr(self, 'teams_config') or not self.teams_config.get('enabled'):
                logger.warning("Microsoft Teams no está configurado")
                return
            
            webhook_url = self.teams_config.get('webhook_url')
            if not webhook_url:
                logger.warning("URL de webhook de Teams no configurada")
                return
            
            # Crear tarjeta adaptativa para Teams
            card = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": hex(level_config['color'])[2:],
                "summary": notification['subject'],
                "sections": [
                    {
                        "activityTitle": f"{level_config['emoji']} {notification['subject']}",
                        "activitySubtitle": f"Nivel: {notification['level'].upper()}",
                        "activityImage": self._get_level_image(notification['level']),
                        "text": notification['message'],
                        "facts": [
                            {
                                "name": "ID",
                                "value": notification['id']
                            },
                            {
                                "name": "Tipo",
                                "value": notification['type']
                            },
                            {
                                "name": "Fecha",
                                "value": notification['timestamp']
                            }
                        ]
                    }
                ]
            }
            
            # Añadir datos adicionales
            if notification['data']:
                facts = []
                for key, value in notification['data'].items():
                    facts.append({
                        "name": key,
                        "value": str(value)
                    })
                
                card['sections'].append({
                    "title": "Detalles adicionales",
                    "facts": facts
                })
            
            # Añadir acciones si es crítico
            if notification['level'] == 'critical':
                card['potentialAction'] = [
                    {
                        "@type": "OpenUri",
                        "name": "Ver en panel",
                        "targets": [
                            {
                                "os": "default",
                                "uri": f"https://dashboard.example.com/notifications/{notification['id']}"
                            }
                        ]
                    }
                ]
            
            # Enviar a Teams
            response = requests.post(webhook_url, json=card)
            
            if response.status_code >= 400:
                logger.error(f"Error al enviar a Teams: {response.status_code} {response.text}")
                raise Exception(f"Error de Teams: {response.status_code}")
            
            logger.info("Mensaje enviado a Microsoft Teams correctamente")
        
        except Exception as e:
            logger.error(f"Error al enviar notificación a Teams: {str(e)}")
            raise
    
    def _get_level_image(self, level: str) -> str:
        """Obtiene URL de imagen para nivel de alerta"""
        # Imágenes predeterminadas para cada nivel
        level_images = {
            'info': 'https://example.com/images/info.png',
            'warning': 'https://example.com/images/warning.png',
            'critical': 'https://example.com/images/critical.png'
        }
        
        # Usar imágenes personalizadas si están configuradas
        if hasattr(self, 'notification_images') and level in self.notification_images:
            return self.notification_images[level]
        
        return level_images.get(level, level_images['info'])
    
    def _send_to_channels(self, notification: Dict, level_config: Dict) -> Dict:
        """
        Envía notificación a todos los canales activos configurados
        
        Args:
            notification: Datos de la notificación
            level_config: Configuración del nivel de alerta
            
        Returns:
            Diccionario con resultados por canal
        """
        results = {
            'success': [],
            'failure': []
        }
        
        # Obtener canales activos para este nivel
        channels = self._get_active_channels(notification['level'], notification['user_id'])
        
        # Enviar a cada canal
        for channel in channels:
            try:
                self._send_to_channel(channel, notification, notification['user_id'])
                results['success'].append(channel)
                
                # Actualizar métricas
                if hasattr(self, 'notification_metrics'):
                    self.notification_metrics['success_channels'] += 1
                    self.notification_metrics['total_channels'] += 1
            except Exception as e:
                logger.error(f"Error al enviar a canal {channel}: {str(e)}")
                results['failure'].append(channel)
                
                # Actualizar métricas
                if hasattr(self, 'notification_metrics'):
                    self.notification_metrics['total_channels'] += 1
        
        return results
    
    def _get_active_channels(self, level: str, user_id: str = None) -> List[str]:
        """
        Obtiene canales activos para un nivel y usuario
        
        Args:
            level: Nivel de alerta
            user_id: ID de usuario opcional
            
        Returns:
            Lista de canales activos
        """
        # Canales predeterminados para el nivel
        default_channels = self.alert_levels_config[level]['channels']
        
        # Filtrar solo canales habilitados
        active_channels = [
            ch for ch in default_channels 
            if ch in self.notification_channels and self.notification_channels[ch].get('enabled', False)
        ]
        
        # Aplicar preferencias de usuario si existe
        if user_id and user_id in self.user_configs:
            user_channels = self.user_configs[user_id].get('channels', [])
            # Intersección de canales de usuario y canales activos para este nivel
            active_channels = [ch for ch in user_channels if ch in active_channels]
        
        return active_channels
    
    def _should_suppress_duplicate(self, notification: Dict) -> bool:
        """
        Determina si una notificación debe suprimirse por ser duplicada
        
        Args:
            notification: Datos de la notificación
            
        Returns:
            True si debe suprimirse, False en caso contrario
        """
        # Verificar historial reciente
        recent_notifications = []
        current_time = datetime.datetime.now().timestamp()
        
        # Obtener ventana de supresión para este nivel
        suppression_window = self.alert_levels_config[notification['level']]['suppression_window']
        
        # Filtrar notificaciones recientes dentro de la ventana de tiempo
        for notif in self.notification_history:
            notif_time = datetime.datetime.fromisoformat(notif['timestamp']).timestamp()
            if current_time - notif_time <= suppression_window:
                recent_notifications.append(notif)
        
        # Buscar duplicados
        for recent in recent_notifications:
            if self._are_notifications_similar(notification, recent):
                logger.info(f"Notificación similar encontrada: {recent['id']}, suprimiendo nueva notificación")
                return True
        
        return False
    
    def _are_notifications_similar(self, notif1: Dict, notif2: Dict) -> bool:
        """
        Compara dos notificaciones para determinar si son similares
        
        Args:
            notif1: Primera notificación
            notif2: Segunda notificación
            
        Returns:
            True si son similares, False en caso contrario
        """
        # Si son del mismo tipo y nivel, comparar contenido
        if notif1['type'] == notif2['type'] and notif1['level'] == notif2['level']:
            # Comparar asunto (puede tener pequeñas variaciones)
            from difflib import SequenceMatcher
            subject_similarity = SequenceMatcher(None, notif1['subject'], notif2['subject']).ratio()
            
            # Si el asunto es muy similar, considerar duplicado
            if subject_similarity > 0.8:
                return True
            
            # Comparar datos clave si existen
            if notif1.get('data') and notif2.get('data'):
                # Verificar si tienen las mismas claves principales
                keys1 = set(notif1['data'].keys())
                keys2 = set(notif2['data'].keys())
                common_keys = keys1.intersection(keys2)
                
                # Si tienen claves en común, verificar valores
                if common_keys:
                    matches = 0
                    for key in common_keys:
                        if str(notif1['data'][key]) == str(notif2['data'][key]):
                            matches += 1
                    
                    # Si más del 70% de los valores coinciden, considerar duplicado
                    if matches / len(common_keys) > 0.7:
                        return True
        
        return False
    
    def _should_escalate(self, notification: Dict) -> bool:
        """
        Determina si una notificación debe escalarse inmediatamente
        
        Args:
            notification: Datos de la notificación
            
        Returns:
            True si debe escalarse, False en caso contrario
        """
        # Verificar si el nivel actual permite escalado
        current_level = notification['level']
        if current_level == 'critical':
            # Ya está en nivel máximo
            return False
        
        # Verificar condiciones de escalado inmediato
        
        # 1. Verificar si hay muchas notificaciones similares recientes
        similar_count = 0
        current_time = datetime.datetime.now().timestamp()
        
        # Contar notificaciones similares en la última hora
        for notif in self.notification_history:
            notif_time = datetime.datetime.fromisoformat(notif['timestamp']).timestamp()
            if current_time - notif_time <= 3600 and notif['type'] == notification['type']:
                similar_count += 1
        
        # Si hay más de 5 notificaciones similares en la última hora, escalar
        if similar_count >= 5:
            logger.info(f"Escalando notificación debido a frecuencia alta: {similar_count} en la última hora")
            return True
        
        # 2. Verificar palabras clave de escalado en el mensaje
        escalation_keywords = [
            'crítico', 'crítica', 'urgente', 'inmediato', 'inmediata',
            'fallo grave', 'error crítico', 'shadowban', 'suspensión',
            'violación', 'bloqueo', 'pérdida de datos'
        ]
        
        message_lower = notification['message'].lower()
        subject_lower = notification['subject'].lower()
        
        for keyword in escalation_keywords:
            if keyword in message_lower or keyword in subject_lower:
                logger.info(f"Escalando notificación debido a palabra clave: '{keyword}'")
                return True
        
        # 3. Verificar datos específicos que indiquen gravedad
        data = notification.get('data', {})
        
        # Verificar métricas críticas
        if 'error_count' in data and data['error_count'] > 10:
            return True
        
        if 'impact_level' in data and data['impact_level'] in ['high', 'critical', 'alto', 'crítico']:
            return True
        
        if 'revenue_impact' in data and data['revenue_impact'] > 100:
            return True
        
        return False
    
    def _escalate_alert(self, notification: Dict, new_level: str = None):
        """Escala una notificación a un nivel superior"""
        if not new_level:
            # Determinar nivel de escalado
            current_level = notification['level']
            if current_level == 'info':
                new_level = 'warning'
            elif current_level == 'warning':
                new_level = 'critical'
            else:
                # Ya está en nivel crítico, no escalar más
                return
        
        notification['escalation_status'] = 'escalated'
        if hasattr(self, 'notification_metrics'):
            self.notification_metrics['escalated_notifications'] += 1
        
        new_subject = f"[ESCALATED] {notification['subject']}"
        new_message = (
            f"⚠️ ALERTA ESCALADA desde {notification['level'].upper()} a {new_level.upper()}\n\n"
            f"Original: {notification['message']}\n\n"
            f"Razón: No resuelta tras {self.alert_levels_config[notification['level']]['escalation']['delay']} segundos"
        )
        
        # Si es crítico, considerar llamada de voz
        if new_level == 'critical' and self.call_config.get('enabled'):
            threading.Thread(target=self._send_voice_call, 
                           args=(notification,), 
                           daemon=True).start()
        
        self.send_notification(
            notification_type=f"{notification['type']}_escalated",
            subject=new_subject,
            message=new_message,
            data={
                **notification['data'],
                'original_level': notification['level'],
                'escalation_time': datetime.datetime.now().isoformat()
            },
            level=new_level,
            user_id=notification['user_id']
        )
        logger.info(f"Notificación {notification['id']} escalada a {new_level}")
    
    def _send_voice_call(self, notification: Dict):
        """
        Realiza llamada de voz para alertas críticas
        
        Args:
            notification: Datos de la notificación
        
        Returns:
            True si la llamada se realizó correctamente, False en caso contrario
        """
        try:
            # Verificar si Twilio está configurado
            if not self.call_config.get('enabled'):
                logger.warning("Llamadas de voz no están habilitadas")
                return False
            
            # Importar Twilio
            from twilio.rest import Client
            
            # Configuración de Twilio
            account_sid = self.call_config['twilio_account_sid']
            auth_token = self.call_config['twilio_auth_token']
            from_number = self.call_config['from_number']
            
            # Determinar destinatarios
            recipients = self.call_config['to_numbers']
            
            # Si hay un usuario específico, usar sus números
            if notification.get('user_id') and notification['user_id'] in self.user_configs:
                user_numbers = self.user_configs[notification['user_id']].get('phone_numbers', [])
                if user_numbers:
                    recipients = user_numbers
            
            if not recipients:
                logger.warning("No hay destinatarios para llamada de voz")
                return False
            
            # Crear cliente Twilio
            client = Client(account_sid, auth_token)
            
            # Preparar mensaje TwiML
            twiml = f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Say voice="woman" language="es-ES">
                    Alerta crítica del sistema de monetización de contenido.
                    {notification['subject']}.
                    {notification['message']}.
                    Esta alerta requiere atención inmediata.
                </Say>
                <Pause length="1"/>
                <Say voice="woman" language="es-ES">
                    Repito, alerta crítica del sistema.
                    {notification['subject']}.
                    Por favor, revise el panel de control para más detalles.
                </Say>
            </Response>
            """
            
            # Determinar URL de TwiML
            twiml_url = None
            
            # Opción 1: Usar Twilio Bin (recomendado para producción)
            if hasattr(self, 'twilio_bin_id') and self.twilio_bin_id:
                # Actualizar TwiML Bin existente
                try:
                    client.twiml_bins(self.twilio_bin_id).update(twiml=twiml)
                    twiml_url = f"https://handler.twilio.com/twiml/{self.twilio_bin_id}"
                except Exception as e:
                    logger.error(f"Error al actualizar TwiML Bin: {str(e)}")
                    # Continuar con opción alternativa
            
            # Opción 2: Guardar TwiML localmente y servir desde servidor web
            if not twiml_url:
                # Crear directorio temporal si no existe
                twiml_path = os.path.join('temp', f"call_{notification['id']}.xml")
                os.makedirs(os.path.dirname(twiml_path), exist_ok=True)
                
                with open(twiml_path, 'w') as f:
                    f.write(twiml)
                
                # Usar URL local (requiere servidor web)
                twiml_url = f"http://localhost:8000/temp/{os.path.basename(twiml_path)}"
            
            # Realizar llamada a cada destinatario
            success = True
            for recipient in recipients:
                try:
                    call = client.calls.create(
                        to=recipient,
                        from_=self.call_config['from_number'],
                        url=twiml_url,
                        method='GET'
                    )
                    logger.info(f"Llamada iniciada a {recipient}: {call.sid}")
                except Exception as e:
                    logger.error(f"Error al realizar llamada a {recipient}: {str(e)}")
                    success = False
            
            return success
            
        except ImportError:
            logger.error("No se pudo importar Twilio. Instale con: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Error al realizar llamada: {str(e)}")
            # Incrementar contador de fallos
            if hasattr(self, 'notification_metrics'):
                self.notification_metrics['failure_count'] += 1
            return False
    
    def _initialize_metrics(self):
        """Inicializa métricas de notificaciones"""
        self.notification_metrics = {
            'sent_count': 0,
            'suppressed_count': 0,
            'failure_count': 0,
            'success_channels': 0,
            'total_channels': 0,
            'start_time': datetime.datetime.now()
        }
    
    def get_metrics(self) -> Dict:
        """
        Obtiene métricas del sistema de notificaciones
        
        Returns:
            Diccionario con métricas
        """
        if not hasattr(self, 'notification_metrics'):
            self._initialize_metrics()
        
        # Calcular métricas adicionales
        metrics = self.notification_metrics.copy()
        
        # Calcular tiempo de actividad
        uptime = (datetime.datetime.now() - metrics['start_time']).total_seconds()
        metrics['uptime_seconds'] = uptime
        metrics['uptime_formatted'] = str(datetime.timedelta(seconds=int(uptime)))
        
        # Calcular tasa de éxito
        if metrics['total_channels'] > 0:
            metrics['success_rate'] = (metrics['success_channels'] / metrics['total_channels']) * 100
        else:
            metrics['success_rate'] = 100.0
        
        # Calcular tasa de supresión
        total_attempted = metrics['sent_count'] + metrics['suppressed_count']
        if total_attempted > 0:
            metrics['suppression_rate'] = (metrics['suppressed_count'] / total_attempted) * 100
        else:
            metrics['suppression_rate'] = 0.0
        
        return metrics
    
    def reset_metrics(self):
        """Reinicia las métricas del sistema de notificaciones"""
        self._initialize_metrics()
        logger.info("Métricas de notificaciones reiniciadas")
    
        def _load_notification_templates(self) -> Dict:
        """
        Carga plantillas de notificaciones desde archivo
        
        Returns:
            Diccionario con plantillas de notificaciones
        """
        templates_path = os.path.join('config', 'notification_templates.json')
        
        if not os.path.exists(templates_path):
            logger.warning(f"Archivo de plantillas no encontrado: {templates_path}")
            return {}
        
        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            logger.info(f"Cargadas {len(templates)} plantillas de notificaciones")
            return templates
        except Exception as e:
            logger.error(f"Error al cargar plantillas: {str(e)}")
            return {}
    
    def _apply_template(self, template_name: str, data: Dict) -> Dict:
        """
        Aplica una plantilla de notificación con datos específicos
        
        Args:
            template_name: Nombre de la plantilla
            data: Datos para rellenar la plantilla
            
        Returns:
            Diccionario con notificación formateada
        """
        if not hasattr(self, 'notification_templates'):
            self.notification_templates = self._load_notification_templates()
        
        if template_name not in self.notification_templates:
            logger.warning(f"Plantilla no encontrada: {template_name}")
            return {
                'subject': f"Notificación sin plantilla: {template_name}",
                'message': "No se encontró la plantilla para esta notificación.",
                'level': 'info'
            }
        
        template = self.notification_templates[template_name]
        
        # Formatear asunto y mensaje con datos
        subject = template['subject']
        message = template['message']
        
        # Reemplazar variables en asunto y mensaje
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            if placeholder in subject:
                subject = subject.replace(placeholder, str(value))
            if placeholder in message:
                message = message.replace(placeholder, str(value))
        
        return {
            'subject': subject,
            'message': message,
            'level': template.get('level', 'info'),
            'channels': template.get('channels', None)
        }
    
    def _format_monetization_alert(self, data: Dict) -> Dict:
        """
        Formatea alerta específica de monetización
        
        Args:
            data: Datos de monetización
            
        Returns:
            Diccionario con alerta formateada
        """
        # Determinar tipo de alerta de monetización
        alert_type = data.get('alert_type', 'general')
        
        # Formatear según tipo
        if alert_type == 'revenue_drop':
            return self._apply_template('monetization_revenue_drop', data)
        elif alert_type == 'opportunity':
            return self._apply_template('monetization_opportunity', data)
        elif alert_type == 'threshold':
            return self._apply_template('monetization_threshold', data)
        elif alert_type == 'affiliate_performance':
            return self._apply_template('monetization_affiliate', data)
        else:
            return self._apply_template('monetization_general', data)
    
    def _format_content_alert(self, data: Dict) -> Dict:
        """
        Formatea alerta específica de contenido
        
        Args:
            data: Datos de contenido
            
        Returns:
            Diccionario con alerta formateada
        """
        # Determinar tipo de alerta de contenido
        alert_type = data.get('alert_type', 'general')
        
        # Formatear según tipo
        if alert_type == 'shadowban':
            return self._apply_template('content_shadowban', data)
        elif alert_type == 'distribution':
            return self._apply_template('content_distribution', data)
        elif alert_type == 'engagement':
            return self._apply_template('content_engagement', data)
        elif alert_type == 'viral_potential':
            return self._apply_template('content_viral', data)
        else:
            return self._apply_template('content_general', data)
    
    def _format_system_alert(self, data: Dict) -> Dict:
        """
        Formatea alerta específica del sistema
        
        Args:
            data: Datos del sistema
            
        Returns:
            Diccionario con alerta formateada
        """
        # Determinar tipo de alerta del sistema
        alert_type = data.get('alert_type', 'general')
        
        # Formatear según tipo
        if alert_type == 'api_limit':
            return self._apply_template('system_api_limit', data)
        elif alert_type == 'error':
            return self._apply_template('system_error', data)
        elif alert_type == 'update':
            return self._apply_template('system_update', data)
        elif alert_type == 'security':
            return self._apply_template('system_security', data)
        else:
            return self._apply_template('system_general', data)
    
    def _format_task_notification(self, data: Dict) -> Dict:
        """
        Formatea notificación de tarea
        
        Args:
            data: Datos de la tarea
            
        Returns:
            Diccionario con notificación formateada
        """
        # Determinar estado de la tarea
        task_status = data.get('status', 'completed')
        
        # Formatear según estado
        if task_status == 'completed':
            return self._apply_template('task_completed', data)
        elif task_status == 'failed':
            return self._apply_template('task_failed', data)
        elif task_status == 'started':
            return self._apply_template('task_started', data)
        elif task_status == 'progress':
            return self._apply_template('task_progress', data)
        else:
            return self._apply_template('task_general', data)
    
    def _format_audience_notification(self, data: Dict) -> Dict:
        """
        Formatea notificación de audiencia
        
        Args:
            data: Datos de audiencia
            
        Returns:
            Diccionario con notificación formateada
        """
        # Determinar tipo de notificación de audiencia
        notification_type = data.get('notification_type', 'general')
        
        # Formatear según tipo
        if notification_type == 'milestone':
            return self._apply_template('audience_milestone', data)
        elif notification_type == 'growth':
            return self._apply_template('audience_growth', data)
        elif notification_type == 'engagement':
            return self._apply_template('audience_engagement', data)
        elif notification_type == 'demographic':
            return self._apply_template('audience_demographic', data)
        else:
            return self._apply_template('audience_general', data)
    
    def _send_batch_notifications(self, notifications: List[Dict]):
        """
        Envía un lote de notificaciones agrupadas
        
        Args:
            notifications: Lista de notificaciones a enviar
        """
        if not notifications:
            return
        
        # Agrupar notificaciones inteligentemente
        grouped = self._group_notifications_intelligently(notifications)
        
        # Enviar cada grupo
        for group in grouped:
            if len(group) == 1:
                # Si solo hay una notificación, enviarla normalmente
                self.send_notification(
                    notification_type=group[0]['type'],
                    subject=group[0]['subject'],
                    message=group[0]['message'],
                    data=group[0]['data'],
                    level=group[0]['level'],
                    user_id=group[0].get('user_id')
                )
            else:
                # Si hay múltiples, crear una notificación agrupada
                self._send_grouped_notification(group)
    
    def _send_grouped_notification(self, notifications: List[Dict]):
        """
        Envía una notificación agrupada
        
        Args:
            notifications: Lista de notificaciones a agrupar
        """
        # Determinar nivel más alto
        highest_level = 'info'
        for notif in notifications:
            if self.alert_levels_config[notif['level']]['priority'] > self.alert_levels_config[highest_level]['priority']:
                highest_level = notif['level']
        
        # Crear asunto agrupado
        subject = f"Resumen de {len(notifications)} notificaciones"
        
        # Crear mensaje agrupado
        message = f"Se han recibido {len(notifications)} notificaciones:\n\n"
        
        for i, notif in enumerate(notifications, 1):
            message += f"{i}. [{notif['level'].upper()}] {notif['subject']}\n"
            message += f"   {notif['message']}\n\n"
        
        # Crear datos agrupados
        data = {
            'grouped': True,
            'count': len(notifications),
            'notification_ids': [n['id'] for n in notifications],
            'types': [n['type'] for n in notifications]
        }
        
        # Enviar notificación agrupada
        self.send_notification(
            notification_type='grouped',
            subject=subject,
            message=message,
            data=data,
            level=highest_level
        )
    
    def _schedule_escalation(self, notification: Dict):
        """
        Programa la escalación de una notificación
        
        Args:
            notification: Notificación a escalar
        """
        # Verificar si el nivel permite escalado
        level = notification['level']
        if level not in self.alert_levels_config or level == 'critical':
            return
        
        # Obtener configuración de escalado
        escalation_config = self.alert_levels_config[level].get('escalation')
        if not escalation_config or not escalation_config.get('enabled', False):
            return
        
        # Determinar tiempo de espera
        delay = escalation_config.get('delay', 3600)  # 1 hora por defecto
        
        # Programar escalado
        threading.Timer(
            delay,
            self._check_and_escalate,
            args=[notification['id']]
        ).start()
        
        logger.info(f"Escalado programado para notificación {notification['id']} en {delay} segundos")
    
    def _check_and_escalate(self, notification_id: str):
        """
        Verifica si una notificación debe escalarse
        
        Args:
            notification_id: ID de la notificación
        """
        # Buscar notificación en historial
        notification = None
        for notif in self.notification_history:
            if notif['id'] == notification_id:
                notification = notif
                break
        
        if not notification:
            logger.warning(f"Notificación {notification_id} no encontrada para escalado")
            return
        
        # Verificar si ya fue resuelta
        if notification.get('resolved', False):
            logger.info(f"Notificación {notification_id} ya resuelta, no se escala")
            return
        
        # Verificar si ya fue escalada
        if notification.get('escalation_status') == 'escalated':
            logger.info(f"Notificación {notification_id} ya escalada")
            return
        
        # Escalar notificación
        self._escalate_alert(notification)
    
    def mark_as_resolved(self, notification_id: str, resolution_note: str = None) -> bool:
        """
        Marca una notificación como resuelta
        
        Args:
            notification_id: ID de la notificación
            resolution_note: Nota opcional sobre la resolución
            
        Returns:
            True si se marcó correctamente, False en caso contrario
        """
        # Buscar notificación en historial
        for notif in self.notification_history:
            if notif['id'] == notification_id:
                notif['resolved'] = True
                notif['resolution_time'] = datetime.datetime.now().isoformat()
                if resolution_note:
                    notif['resolution_note'] = resolution_note
                
                logger.info(f"Notificación {notification_id} marcada como resuelta")
                return True
        
        logger.warning(f"Notificación {notification_id} no encontrada para marcar como resuelta")
        return False
    
    def get_active_notifications(self, level: str = None, user_id: str = None) -> List[Dict]:
        """
        Obtiene notificaciones activas (no resueltas)
        
        Args:
            level: Filtrar por nivel (opcional)
            user_id: Filtrar por usuario (opcional)
            
        Returns:
            Lista de notificaciones activas
        """
        active = []
        
        for notif in self.notification_history:
            if notif.get('resolved', False):
                continue
            
            if level and notif['level'] != level:
                continue
            
            if user_id and notif.get('user_id') != user_id:
                continue
            
            active.append(notif)
        
        return active
    
    def get_notification_by_id(self, notification_id: str) -> Optional[Dict]:
        """
        Obtiene una notificación por su ID
        
        Args:
            notification_id: ID de la notificación
            
        Returns:
            Notificación o None si no se encuentra
        """
        for notif in self.notification_history:
            if notif['id'] == notification_id:
                return notif
        
        return None
    
    def get_notification_history(self, limit: int = 100, 
                               level: str = None, 
                               user_id: str = None,
                               type_filter: str = None,
                               resolved: bool = None) -> List[Dict]:
        """
        Obtiene historial de notificaciones con filtros
        
        Args:
            limit: Número máximo de notificaciones a devolver
            level: Filtrar por nivel (opcional)
            user_id: Filtrar por usuario (opcional)
            type_filter: Filtrar por tipo (opcional)
            resolved: Filtrar por estado de resolución (opcional)
            
        Returns:
            Lista de notificaciones filtradas
        """
        filtered = []
        
        for notif in self.notification_history:
            if level and notif['level'] != level:
                continue
            
            if user_id and notif.get('user_id') != user_id:
                continue
            
            if type_filter and notif['type'] != type_filter:
                continue
            
            if resolved is not None and notif.get('resolved', False) != resolved:
                continue
            
            filtered.append(notif)
        
        # Ordenar por fecha (más recientes primero)
        filtered.sort(
            key=lambda n: datetime.datetime.fromisoformat(n['timestamp']).timestamp(),
            reverse=True
        )
        
        # Limitar resultados
        return filtered[:limit]
    
    def clear_notification_history(self, days_old: int = 30) -> int:
        """
        Limpia notificaciones antiguas del historial
        
        Args:
            days_old: Eliminar notificaciones más antiguas que estos días
            
        Returns:
            Número de notificaciones eliminadas
        """
        if not self.notification_history:
            return 0
        
        current_time = datetime.datetime.now().timestamp()
        cutoff_time = current_time - (days_old * 86400)  # días a segundos
        
        original_count = len(self.notification_history)
        
        # Filtrar notificaciones más recientes que el límite
        self.notification_history = [
            notif for notif in self.notification_history
            if datetime.datetime.fromisoformat(notif['timestamp']).timestamp() > cutoff_time
        ]
        
        removed_count = original_count - len(self.notification_history)
        logger.info(f"Eliminadas {removed_count} notificaciones antiguas del historial")
        
        return removed_count
    
    def save_notification_history(self) -> bool:
        """
        Guarda historial de notificaciones en archivo
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        history_path = os.path.join('data', 'notification_history.json')
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.notification_history, f, indent=2)
            
            logger.info(f"Historial de notificaciones guardado: {len(self.notification_history)} entradas")
            return True
        except Exception as e:
            logger.error(f"Error al guardar historial de notificaciones: {str(e)}")
            return False
    
    def load_notification_history(self) -> bool:
        """
        Carga historial de notificaciones desde archivo
        
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        history_path = os.path.join('data', 'notification_history.json')
        
        if not os.path.exists(history_path):
            logger.info("No existe archivo de historial de notificaciones")
            return False
        
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                self.notification_history = json.load(f)
            
            logger.info(f"Historial de notificaciones cargado: {len(self.notification_history)} entradas")
            return True
        except Exception as e:
            logger.error(f"Error al cargar historial de notificaciones: {str(e)}")
            self.notification_history = []
            return False
    
    def send_monetization_alert(self, alert_data: Dict, level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta específica de monetización
        
        Args:
            alert_data: Datos de la alerta de monetización
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Formatear alerta
        formatted = self._format_monetization_alert(alert_data)
        
        # Usar nivel de plantilla si no se especifica
        if level == 'warning' and 'level' in formatted:
            level = formatted['level']
        
        # Enviar notificación
        return self.send_notification(
            notification_type='monetization_alert',
            subject=formatted['subject'],
            message=formatted['message'],
            data=alert_data,
            level=level,
            channels=formatted.get('channels'),
            user_id=user_id
        )
    
    def send_content_alert(self, alert_data: Dict, level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta específica de contenido
        
        Args:
            alert_data: Datos de la alerta de contenido
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Formatear alerta
        formatted = self._format_content_alert(alert_data)
        
        # Usar nivel de plantilla si no se especifica
        if level == 'warning' and 'level' in formatted:
            level = formatted['level']
        
        # Enviar notificación
        return self.send_notification(
            notification_type='content_alert',
            subject=formatted['subject'],
            message=formatted['message'],
            data=alert_data,
            level=level,
            channels=formatted.get('channels'),
            user_id=user_id
        )
    
    def send_system_alert(self, alert_data: Dict, level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta específica del sistema
        
        Args:
            alert_data: Datos de la alerta del sistema
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Formatear alerta
        formatted = self._format_system_alert(alert_data)
        
        # Usar nivel de plantilla si no se especifica
        if level == 'warning' and 'level' in formatted:
            level = formatted['level']
        
        # Enviar notificación
        return self.send_notification(
            notification_type='system_alert',
            subject=formatted['subject'],
            message=formatted['message'],
            data=alert_data,
            level=level,
            channels=formatted.get('channels'),
            user_id=user_id
        )
    
    def send_task_notification(self, task_data: Dict, level: str = 'info', user_id: str = None) -> str:
        """
        Envía notificación de tarea
        
        Args:
            task_data: Datos de la tarea
            level: Nivel de notificación
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Formatear notificación
        formatted = self._format_task_notification(task_data)
        
        # Usar nivel de plantilla si no se especifica
        if level == 'info' and 'level' in formatted:
            level = formatted['level']
        
        # Enviar notificación
        return self.send_notification(
            notification_type='task_notification',
            subject=formatted['subject'],
            message=formatted['message'],
            data=task_data,
            level=level,
            channels=formatted.get('channels'),
            user_id=user_id
        )
    
    def send_audience_notification(self, audience_data: Dict, level: str = 'info', user_id: str = None) -> str:
        """
        Envía notificación de audiencia
        
        Args:
            audience_data: Datos de audiencia
            level: Nivel de notificación
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Formatear notificación
        formatted = self._format_audience_notification(audience_data)
        
        # Usar nivel de plantilla si no se especifica
        if level == 'info' and 'level' in formatted:
            level = formatted['level']
        
        # Enviar notificación
        return self.send_notification(
            notification_type='audience_notification',
            subject=formatted['subject'],
            message=formatted['message'],
            data=audience_data,
            level=level,
            channels=formatted.get('channels'),
            user_id=user_id
        )
    
    def send_shadowban_alert(self, platform: str, content_id: str, 
                           metrics: Dict, level: str = 'critical', 
                           user_id: str = None) -> str:
        """
        Envía alerta específica de shadowban
        
        Args:
            platform: Plataforma afectada
            content_id: ID del contenido afectado
            metrics: Métricas que indican shadowban
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Crear datos de alerta
        alert_data = {
            'alert_type': 'shadowban',
            'platform': platform,
            'content_id': content_id,
            'metrics': metrics,
            'detection_time': datetime.datetime.now().isoformat()
        }
        
        # Enviar alerta de contenido
        return self.send_content_alert(alert_data, level, user_id)
    
    def send_revenue_alert(self, platform: str, content_id: str, 
                         metrics: Dict, threshold: float,
                         level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta específica de ingresos
        
        Args:
            platform: Plataforma afectada
            content_id: ID del contenido afectado
            metrics: Métricas de ingresos
            threshold: Umbral que activó la alerta
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Crear datos de alerta
        alert_data = {
            'alert_type': 'revenue_drop',
            'platform': platform,
            'content_id': content_id,
            'metrics': metrics,
            'threshold': threshold,
            'detection_time': datetime.datetime.now().isoformat()
        }
        
        # Enviar alerta de monetización
        return self.send_monetization_alert(alert_data, level, user_id)
    
    def send_api_limit_alert(self, service: str, limit_type: str, 
                           current: int, max_limit: int,
                           level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta específica de límite de API
        
        Args:
            service: Servicio afectado
            limit_type: Tipo de límite
            current: Uso actual
            max_limit: Límite máximo
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Determinar nivel según cercanía al límite
        usage_percent = (current / max_limit) * 100
        
        if usage_percent >= 95:
            level = 'critical'
        elif usage_percent >= 80:
            level = 'warning'
        else:
            level = 'info'
        
        # Crear datos de alerta
        alert_data = {
            'alert_type': 'api_limit',
            'service': service,
            'limit_type': limit_type,
            'current_usage': current,
            'max_limit': max_limit,
            'usage_percent': usage_percent,
            'detection_time': datetime.datetime.now().isoformat()
        }
        
        # Enviar alerta del sistema
        return self.send_system_alert(alert_data, level, user_id)
    
    def send_niche_saturation_alert(self, niche: str, metrics: Dict,
                                  level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta específica de saturación de nicho
        
        Args:
            niche: Nicho afectado
            metrics: Métricas de saturación
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Crear datos de alerta
        alert_data = {
            'alert_type': 'saturation',
            'niche': niche,
            'metrics': metrics,
            'detection_time': datetime.datetime.now().isoformat()
        }
        
        # Enviar alerta de contenido
        return self.send_content_alert(alert_data, level, user_id)
    
    def send_viral_opportunity_alert(self, content_id: str, platform: str,
                                   metrics: Dict, opportunity_score: float,
                                   level: str = 'info', user_id: str = None) -> str:
        """
        Envía alerta específica de oportunidad viral
        
        Args:
            content_id: ID del contenido
            platform: Plataforma
            metrics: Métricas de viralidad
            opportunity_score: Puntuación de oportunidad
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Determinar nivel según puntuación
        if opportunity_score >= 0.8:
            level = 'warning'  # Más urgente para no perder la oportunidad
        
        # Crear datos de alerta
        alert_data = {
            'alert_type': 'viral_potential',
            'content_id': content_id,
            'platform': platform,
            'metrics': metrics,
            'opportunity_score': opportunity_score,
            'detection_time': datetime.datetime.now().isoformat()
        }
        
        # Enviar alerta de contenido
        return self.send_content_alert(alert_data, level, user_id)
    
    def send_milestone_notification(self, milestone_type: str, platform: str,
                                  value: Union[int, float], previous: Union[int, float],
                                  level: str = 'info', user_id: str = None) -> str:
        """
        Envía notificación de hito alcanzado
        
        Args:
            milestone_type: Tipo de hito (seguidores, ingresos, etc.)
            platform: Plataforma
            value: Valor actual
            previous: Valor anterior
            level: Nivel de notificación
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Crear datos de notificación
        notification_data = {
            'notification_type': 'milestone',
            'milestone_type': milestone_type,
            'platform': platform,
            'value': value,
            'previous': previous,
            'growth': value - previous,
            'growth_percent': ((value - previous) / previous) * 100 if previous > 0 else 100,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Enviar notificación de audiencia
        return self.send_audience_notification(notification_data, level, user_id)
    
    def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """
        Actualiza preferencias de notificación para un usuario
        
        Args:
            user_id: ID del usuario
            preferences: Preferencias de notificación
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        if not user_id:
            logger.error("ID de usuario requerido para actualizar preferencias")
            return False
        
        # Inicializar configuración de usuario si no existe
        if user_id not in self.user_configs:
            self.user_configs[user_id] = {}
        
        # Actualizar preferencias
        for key, value in preferences.items():
            self.user_configs[user_id][key] = value
        
        logger.info(f"Preferencias actualizadas para usuario {user_id}")
        
        # Guardar configuración
        self._save_user_configs()
        
        return True
    
    def _save_user_configs(self) -> bool:
        """
        Guarda configuraciones de usuario en archivo
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
                config_path = os.path.join('config', 'user_notification_configs.json')
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_configs, f, indent=2)
            
            logger.info(f"Configuraciones de usuario guardadas: {len(self.user_configs)} usuarios")
            return True
        except Exception as e:
            logger.error(f"Error al guardar configuraciones de usuario: {str(e)}")
            return False
    
    def _load_user_configs(self) -> bool:
        """
        Carga configuraciones de usuario desde archivo
        
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        config_path = os.path.join('config', 'user_notification_configs.json')
        
        if not os.path.exists(config_path):
            logger.info("No existe archivo de configuraciones de usuario")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.user_configs = json.load(f)
            
            logger.info(f"Configuraciones de usuario cargadas: {len(self.user_configs)} usuarios")
            return True
        except Exception as e:
            logger.error(f"Error al cargar configuraciones de usuario: {str(e)}")
            self.user_configs = {}
            return False
    
    def _escalate_alert(self, notification: Dict):
        """
        Escala una alerta a un nivel superior
        
        Args:
            notification: Notificación a escalar
        """
        # Determinar nivel actual y siguiente
        current_level = notification['level']
        
        if current_level == 'info':
            next_level = 'warning'
        elif current_level == 'warning':
            next_level = 'critical'
        else:
            logger.warning(f"No se puede escalar notificación de nivel {current_level}")
            return
        
        # Actualizar estado de escalado
        notification['escalation_status'] = 'escalated'
        notification['escalation_time'] = datetime.datetime.now().isoformat()
        notification['previous_level'] = current_level
        notification['level'] = next_level
        
        # Crear mensaje de escalado
        escalation_subject = f"ESCALADO: {notification['subject']}"
        escalation_message = (
            f"Esta alerta ha sido escalada de {current_level.upper()} a {next_level.upper()}.\n\n"
            f"Alerta original: {notification['message']}\n\n"
            f"Motivo: No se ha resuelto en el tiempo establecido."
        )
        
        # Enviar notificación escalada
        self.send_notification(
            notification_type=notification['type'],
            subject=escalation_subject,
            message=escalation_message,
            data=notification['data'],
            level=next_level,
            user_id=notification.get('user_id'),
            related_id=notification['id']
        )
        
        logger.warning(f"Notificación {notification['id']} escalada de {current_level} a {next_level}")
    
    def _is_duplicate(self, notification_type: str, subject: str, level: str, user_id: str = None) -> Optional[str]:
        """
        Verifica si una notificación es duplicada
        
        Args:
            notification_type: Tipo de notificación
            subject: Asunto de la notificación
            level: Nivel de notificación
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación duplicada o None si no hay duplicado
        """
        # Verificar si la supresión de duplicados está habilitada
        if not self.deduplication_config['enabled']:
            return None
        
        # Obtener ventana de tiempo para duplicados
        time_window = self.deduplication_config['time_window']
        current_time = datetime.datetime.now().timestamp()
        
        # Buscar duplicados recientes
        for notif in self.notification_history:
            # Verificar si es del mismo tipo y nivel
            if notif['type'] != notification_type or notif['level'] != level:
                continue
            
            # Verificar si es para el mismo usuario
            if user_id and notif.get('user_id') != user_id:
                continue
            
            # Verificar si el asunto es similar
            if subject.lower() != notif['subject'].lower():
                continue
            
            # Verificar si está dentro de la ventana de tiempo
            notif_time = datetime.datetime.fromisoformat(notif['timestamp']).timestamp()
            if (current_time - notif_time) <= time_window:
                return notif['id']
        
        return None
    
    def _format_alert_message(self, alert_type: str, subject: str, message: str, data: Dict) -> tuple:
        """
        Formatea mensaje de alerta con información adicional
        
        Args:
            alert_type: Tipo de alerta
            subject: Asunto original
            message: Mensaje original
            data: Datos adicionales
            
        Returns:
            Tupla con (asunto_formateado, mensaje_formateado)
        """
        # Formatear asunto según tipo de alerta
        formatted_subject = f"[{alert_type.upper()}] {subject}"
        
        # Formatear mensaje con detalles adicionales
        formatted_message = message + "\n\n"
        
        # Añadir detalles según tipo de alerta
        if alert_type == 'shadowban':
            formatted_message += (
                f"Plataforma: {data.get('platform', 'Desconocida')}\n"
                f"Contenido: {data.get('content_id', 'Desconocido')}\n"
                f"Detectado: {data.get('detection_time', datetime.datetime.now().isoformat())}\n\n"
                f"Métricas anómalas:\n"
            )
            
            # Añadir métricas
            metrics = data.get('metrics', {})
            for key, value in metrics.items():
                formatted_message += f"- {key}: {value}\n"
                
        elif alert_type == 'revenue_drop':
            formatted_message += (
                f"Plataforma: {data.get('platform', 'Desconocida')}\n"
                f"Contenido: {data.get('content_id', 'Desconocido')}\n"
                f"Umbral: {data.get('threshold', 0)}%\n"
                f"Detectado: {data.get('detection_time', datetime.datetime.now().isoformat())}\n\n"
                f"Métricas de ingresos:\n"
            )
            
            # Añadir métricas
            metrics = data.get('metrics', {})
            for key, value in metrics.items():
                formatted_message += f"- {key}: {value}\n"
                
        elif alert_type == 'saturation':
            formatted_message += (
                f"Nicho: {data.get('niche', 'Desconocido')}\n"
                f"Detectado: {data.get('detection_time', datetime.datetime.now().isoformat())}\n\n"
                f"Métricas de saturación:\n"
            )
            
            # Añadir métricas
            metrics = data.get('metrics', {})
            for key, value in metrics.items():
                formatted_message += f"- {key}: {value}\n"
                
        elif alert_type == 'api_limit':
            formatted_message += (
                f"Servicio: {data.get('service', 'Desconocido')}\n"
                f"Tipo de límite: {data.get('limit_type', 'Desconocido')}\n"
                f"Uso actual: {data.get('current_usage', 0)}/{data.get('max_limit', 0)} "
                f"({data.get('usage_percent', 0):.1f}%)\n"
                f"Detectado: {data.get('detection_time', datetime.datetime.now().isoformat())}\n"
            )
            
        # Añadir recomendaciones si existen
        if 'recommendations' in data:
            formatted_message += "\nRecomendaciones:\n"
            for rec in data['recommendations']:
                formatted_message += f"- {rec}\n"
        
        return formatted_subject, formatted_message
    
    def _get_user_channel_config(self, user_id: str, channel: str) -> Dict:
        """
        Obtiene configuración de canal para un usuario
        
        Args:
            user_id: ID del usuario
            channel: Nombre del canal
            
        Returns:
            Configuración del canal para el usuario
        """
        # Si no hay ID de usuario, usar configuración global
        if not user_id:
            return self.channels_config.get(channel, {})
        
        # Obtener configuración de usuario
        user_config = self.user_configs.get(user_id, {})
        
        # Obtener configuración de canal específica del usuario
        user_channel_config = user_config.get('channels', {}).get(channel)
        
        # Si no hay configuración específica, usar global
        if not user_channel_config:
            return self.channels_config.get(channel, {})
        
        # Combinar configuración global con específica del usuario
        channel_config = self.channels_config.get(channel, {}).copy()
        channel_config.update(user_channel_config)
        
        return channel_config
    
    def _should_send_to_channel(self, user_id: str, channel: str, level: str) -> bool:
        """
        Determina si se debe enviar a un canal específico
        
        Args:
            user_id: ID del usuario
            channel: Nombre del canal
            level: Nivel de notificación
            
        Returns:
            True si se debe enviar, False en caso contrario
        """
        # Obtener configuración de canal
        channel_config = self._get_user_channel_config(user_id, channel)
        
        # Verificar si el canal está habilitado
        if not channel_config.get('enabled', False):
            return False
        
        # Verificar nivel mínimo para el canal
        min_level = channel_config.get('min_level', 'info')
        
        # Comparar prioridades
        level_priority = self.alert_levels_config[level]['priority']
        min_priority = self.alert_levels_config[min_level]['priority']
        
        return level_priority >= min_priority
    
    def _send_to_email(self, notification: Dict, user_id: str = None):
        """
        Envía notificación por correo electrónico
        
        Args:
            notification: Notificación a enviar
            user_id: ID del usuario destinatario
        """
        # Verificar si se debe enviar por email
        if not self._should_send_to_channel(user_id, 'email', notification['level']):
            return
        
        # Obtener configuración de email
        email_config = self._get_user_channel_config(user_id, 'email')
        
        # Obtener destinatario
        recipient = None
        if user_id and 'address' in email_config:
            recipient = email_config['address']
        elif 'default_address' in email_config:
            recipient = email_config['default_address']
        
        if not recipient:
            logger.warning("No se pudo determinar destinatario de email")
            return
        
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_address', 'notificaciones@contentbot.com')
            msg['To'] = recipient
            msg['Subject'] = notification['subject']
            
            # Añadir cuerpo del mensaje
            body = notification['message']
            msg.attach(MIMEText(body, 'plain'))
            
            # Conectar al servidor SMTP
            server = smtplib.SMTP(
                email_config.get('smtp_server', 'smtp.gmail.com'),
                email_config.get('smtp_port', 587)
            )
            server.starttls()
            
            # Iniciar sesión
            server.login(
                email_config.get('smtp_username', ''),
                email_config.get('smtp_password', '')
            )
            
            # Enviar email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email enviado a {recipient}: {notification['subject']}")
        except Exception as e:
            logger.error(f"Error al enviar email: {str(e)}")
    
    def _send_to_webhook(self, notification: Dict, user_id: str = None):
        """
        Envía notificación a webhook
        
        Args:
            notification: Notificación a enviar
            user_id: ID del usuario destinatario
        """
        # Verificar si se debe enviar por webhook
        if not self._should_send_to_channel(user_id, 'webhook', notification['level']):
            return
        
        # Obtener configuración de webhook
        webhook_config = self._get_user_channel_config(user_id, 'webhook')
        
        # Obtener URL del webhook
        webhook_url = None
        if user_id and 'url' in webhook_config:
            webhook_url = webhook_config['url']
        elif 'default_url' in webhook_config:
            webhook_url = webhook_config['default_url']
        
        if not webhook_url:
            logger.warning("No se pudo determinar URL de webhook")
            return
        
        try:
            # Preparar payload
            payload = {
                'id': notification['id'],
                'type': notification['type'],
                'subject': notification['subject'],
                'message': notification['message'],
                'level': notification['level'],
                'timestamp': notification['timestamp'],
                'data': notification['data']
            }
            
            # Añadir usuario si existe
            if user_id:
                payload['user_id'] = user_id
            
            # Enviar a webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=webhook_config.get('timeout', 5)
            )
            
            # Verificar respuesta
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook enviado a {webhook_url}: {notification['subject']}")
            else:
                logger.warning(f"Error al enviar webhook: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error al enviar webhook: {str(e)}")
    
    def _send_to_slack(self, notification: Dict, user_id: str = None):
        """
        Envía notificación a Slack
        
        Args:
            notification: Notificación a enviar
            user_id: ID del usuario destinatario
        """
        # Verificar si se debe enviar por Slack
        if not self._should_send_to_channel(user_id, 'slack', notification['level']):
            return
        
        # Obtener configuración de Slack
        slack_config = self._get_user_channel_config(user_id, 'slack')
        
        # Obtener webhook URL de Slack
        webhook_url = None
        if user_id and 'webhook_url' in slack_config:
            webhook_url = slack_config['webhook_url']
        elif 'default_webhook_url' in slack_config:
            webhook_url = slack_config['default_webhook_url']
        
        if not webhook_url:
            logger.warning("No se pudo determinar webhook URL de Slack")
            return
        
        try:
            # Determinar color según nivel
            color = '#36a64f'  # verde para info
            if notification['level'] == 'warning':
                color = '#ffcc00'  # amarillo
            elif notification['level'] == 'critical':
                color = '#ff0000'  # rojo
            
            # Preparar payload
            payload = {
                'attachments': [
                    {
                        'fallback': notification['subject'],
                        'color': color,
                        'title': notification['subject'],
                        'text': notification['message'],
                        'fields': [
                            {
                                'title': 'Nivel',
                                'value': notification['level'].upper(),
                                'short': True
                            },
                            {
                                'title': 'Tipo',
                                'value': notification['type'],
                                'short': True
                            }
                        ],
                        'footer': f"ID: {notification['id']}",
                        'ts': int(datetime.datetime.fromisoformat(notification['timestamp']).timestamp())
                    }
                ]
            }
            
            # Añadir canal específico si existe
            if 'channel' in slack_config:
                payload['channel'] = slack_config['channel']
            
            # Enviar a Slack
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=slack_config.get('timeout', 5)
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                logger.info(f"Slack enviado: {notification['subject']}")
            else:
                logger.warning(f"Error al enviar Slack: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error al enviar Slack: {str(e)}")
    
    def _send_to_telegram(self, notification: Dict, user_id: str = None):
        """
        Envía notificación a Telegram
        
        Args:
            notification: Notificación a enviar
            user_id: ID del usuario destinatario
        """
        # Verificar si se debe enviar por Telegram
        if not self._should_send_to_channel(user_id, 'telegram', notification['level']):
            return
        
        # Obtener configuración de Telegram
        telegram_config = self._get_user_channel_config(user_id, 'telegram')
        
        # Obtener token y chat_id
        token = telegram_config.get('bot_token')
        chat_id = None
        
        if user_id and 'chat_id' in telegram_config:
            chat_id = telegram_config['chat_id']
        elif 'default_chat_id' in telegram_config:
            chat_id = telegram_config['default_chat_id']
        
        if not token or not chat_id:
            logger.warning("Faltan credenciales para Telegram")
            return
        
        try:
            # Formatear mensaje
            level_emoji = "ℹ️"
            if notification['level'] == 'warning':
                level_emoji = "⚠️"
            elif notification['level'] == 'critical':
                level_emoji = "🚨"
            
            message = (
                f"{level_emoji} *{notification['subject']}*\n\n"
                f"{notification['message']}\n\n"
                f"*Nivel:* {notification['level'].upper()}\n"
                f"*Tipo:* {notification['type']}\n"
                f"*ID:* `{notification['id']}`"
            )
            
            # Enviar a Telegram
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=telegram_config.get('timeout', 5)
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                logger.info(f"Telegram enviado: {notification['subject']}")
            else:
                logger.warning(f"Error al enviar Telegram: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error al enviar Telegram: {str(e)}")
    
    def _send_to_desktop(self, notification: Dict, user_id: str = None):
        """
        Envía notificación al escritorio
        
        Args:
            notification: Notificación a enviar
            user_id: ID del usuario destinatario
        """
        # Verificar si se debe enviar al escritorio
        if not self._should_send_to_channel(user_id, 'desktop', notification['level']):
            return
        
        # Obtener configuración de escritorio
        desktop_config = self._get_user_channel_config(user_id, 'desktop')
        
        # Verificar si está habilitado
        if not desktop_config.get('enabled', False):
            return
        
        try:
            # Intentar importar módulo de notificaciones
            import plyer.platforms.win.notification
            from plyer import notification as plyer_notification
            
            # Enviar notificación
            plyer_notification.notify(
                title=notification['subject'],
                message=notification['message'],
                app_name="ContentBot",
                timeout=desktop_config.get('timeout', 10)
            )
            
            logger.info(f"Notificación de escritorio enviada: {notification['subject']}")
        except ImportError:
            logger.warning("Módulo plyer no encontrado para notificaciones de escritorio")
        except Exception as e:
            logger.error(f"Error al enviar notificación de escritorio: {str(e)}")
    
    def _send_to_database(self, notification: Dict):
        """
        Guarda notificación en base de datos
        
        Args:
            notification: Notificación a guardar
        """
        # Añadir a historial
        self.notification_history.append(notification)
        
        # Guardar historial periódicamente
        self._save_history_if_needed()
    
    def _save_history_if_needed(self):
        """Guarda historial si es necesario según configuración"""
        # Verificar si está habilitado el guardado automático
        if not self.history_config.get('auto_save', {}).get('enabled', False):
            return
        
        # Verificar si es momento de guardar
        current_time = time.time()
        last_save = getattr(self, '_last_history_save', 0)
        save_interval = self.history_config.get('auto_save', {}).get('interval', 3600)
        
        if (current_time - last_save) >= save_interval:
            self.save_notification_history()
            self._last_history_save = current_time
    
    def _retry_failed_notifications(self):
        """Reintenta enviar notificaciones fallidas"""
        # Verificar si hay notificaciones para reintentar
        if not hasattr(self, 'failed_notifications'):
            self.failed_notifications = []
        
        if not self.failed_notifications:
            return
        
        # Obtener configuración de reintentos
        retry_config = self.retry_config
        max_retries = retry_config.get('max_retries', 3)
        
        # Crear copia para iterar mientras modificamos
        to_retry = self.failed_notifications.copy()
        self.failed_notifications = []
        
        for notif in to_retry:
            # Verificar si se alcanzó el máximo de reintentos
            if notif.get('retry_count', 0) >= max_retries:
                logger.warning(f"Máximo de reintentos alcanzado para notificación {notif['id']}")
                continue
            
            # Incrementar contador de reintentos
            notif['retry_count'] = notif.get('retry_count', 0) + 1
            
            # Reintentar envío
            logger.info(f"Reintentando envío de notificación {notif['id']} (intento {notif['retry_count']})")
            
            try:
                # Enviar a canales configurados
                self._send_to_channels(notif, notif.get('user_id'))
            except Exception as e:
                logger.error(f"Error al reintentar notificación {notif['id']}: {str(e)}")
                # Volver a añadir a la lista de fallidos
                self.failed_notifications.append(notif)
    
    def _schedule_retry(self):
        """Programa reintento de notificaciones fallidas"""
        # Verificar si los reintentos están habilitados
        if not self.retry_config.get('enabled', False):
            return
        
        # Obtener intervalo de reintentos
        retry_interval = self.retry_config.get('interval', 300)  # 5 minutos por defecto
        
        # Programar reintento
        threading.Timer(
            retry_interval,
            self._retry_and_reschedule
        ).start()
    
    def _retry_and_reschedule(self):
        """Reintenta notificaciones y reprograma"""
        self._retry_failed_notifications()
        self._schedule_retry()
    
    def _group_notifications_intelligently(self, notifications: List[Dict]) -> List[List[Dict]]:
        """
        Agrupa notificaciones de manera inteligente
        
        Args:
            notifications: Lista de notificaciones a agrupar
            
        Returns:
            Lista de grupos de notificaciones
        """
        if not notifications:
            return []
        
        grouped = []
        current_group = []
        last_time = None
        time_window = self.grouping_config['time_window']
        max_group_size = self.grouping_config['max_group_size']
        
        # Ordenar por prioridad y timestamp
        sorted_notifications = sorted(
            notifications,
            key=lambda n: (
                self.alert_levels_config[n['level']]['priority'],
                datetime.datetime.fromisoformat(n['timestamp']).timestamp()
            )
        )
        
        for notif in sorted_notifications:
            notif_time = datetime.datetime.fromisoformat(notif['timestamp']).timestamp()
            
            # Verificar si entra en la ventana temporal
            if last_time is None or (notif_time - last_time) <= time_window:
                if len(current_group) < max_group_size:
                    current_group.append(notif)
                else:
                    grouped.append(current_group)
                    current_group = [notif]
            else:
                grouped.append(current_group)
                current_group = [notif]
            
            last_time = notif_time
        
        # Añadir último grupo
        if current_group:
            grouped.append(current_group)
        
        return grouped
    
    def _send_log(self, notification: Dict, level_config: Dict):
        """
        Registra notificación en el log
        
        Args:
            notification: Notificación a registrar
            level_config: Configuración del nivel de alerta
        """
        log_message = f"{level_config['prefix']} {notification['subject']}: {notification['message']}"
        if notification.get('user_id'):
            log_message += f" (Usuario: {notification['user_id']})"
        
        if notification['level'] == 'critical':
            logger.critical(log_message)
        elif notification['level'] == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_notification_stats(self, days: int = 7) -> Dict:
        """
        Obtiene estadísticas de notificaciones
        
        Args:
            days: Número de días para las estadísticas
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'total': 0,
            'by_level': {
                'info': 0,
                'warning': 0,
                'critical': 0
            },
            'by_type': {},
            'by_day': {},
            'resolution_time': {
                'average': 0,
                'by_level': {
                    'info': 0,
                    'warning': 0,
                    'critical': 0
                }
            }
        }
        
        # Calcular fecha límite
        cutoff_time = (datetime.datetime.now() - datetime.timedelta(days=days)).timestamp()
        
        # Contadores para tiempo de resolución
        resolution_times = {
            'info': [],
            'warning': [],
            'critical': []
        }
        
        # Procesar notificaciones
        for notif in self.notification_history:
            # Verificar si está dentro del rango de tiempo
            notif_time = datetime.datetime.fromisoformat(notif['timestamp']).timestamp()
            if notif_time < cutoff_time:
                continue
            
            # Incrementar contador total
            stats['total'] += 1
            
            # Incrementar contador por nivel
            level = notif['level']
            stats['by_level'][level] += 1
            
            # Incrementar contador por tipo
            notif_type = notif['type']
            if notif_type not in stats['by_type']:
                stats['by_type'][notif_type] = 0
            stats['by_type'][notif_type] += 1
            
            # Incrementar contador por día
            day = datetime.datetime.fromtimestamp(notif_time).strftime('%Y-%m-%d')
            if day not in stats['by_day']:
                stats['by_day'][day] = 0
            stats['by_day'][day] += 1
            
            # Calcular tiempo de resolución si está resuelto
            if notif.get('resolved', False) and 'resolution_time' in notif:
                resolution_time = (
                    datetime.datetime.fromisoformat(notif['resolution_time']).timestamp() - notif_time
                )
                resolution_times[level].append(resolution_time)
        
        # Calcular tiempos promedio de resolución
        total_resolution_time = 0
        total_resolved = 0
        
        for level, times in resolution_times.items():
            if times:
                avg_time = sum(times) / len(times)
                stats['resolution_time']['by_level'][level] = avg_time
                total_resolution_time += sum(times)
                total_resolved += len(times)
        
        if total_resolved > 0:
            stats['resolution_time']['average'] = total_resolution_time / total_resolved
        
        return stats
    
        def send_notification_stats(self, days: int = 7, user_id: str = None) -> str:
        """
        Envía estadísticas de notificaciones como notificación
        
        Args:
            days: Número de días para las estadísticas
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Obtener estadísticas
        stats = self.get_notification_stats(days)
        
        # Crear asunto
        subject = f"Resumen de notificaciones ({days} días)"
        
        # Crear mensaje
        message = f"Resumen de notificaciones de los últimos {days} días:\n\n"
        
        # Añadir total
        message += f"Total de notificaciones: {stats['total']}\n\n"
        
        # Añadir por nivel
        message += "Por nivel:\n"
        for level, count in stats['by_level'].items():
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            message += f"- {level.upper()}: {count} ({percentage:.1f}%)\n"
        
        message += "\nPor tipo:\n"
        for notif_type, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            message += f"- {notif_type}: {count} ({percentage:.1f}%)\n"
        
        # Añadir tiempo de resolución
        if stats['resolution_time']['average'] > 0:
            avg_time = stats['resolution_time']['average']
            hours = int(avg_time / 3600)
            minutes = int((avg_time % 3600) / 60)
            
            message += f"\nTiempo promedio de resolución: {hours}h {minutes}m\n"
            message += "Por nivel:\n"
            
            for level, time_avg in stats['resolution_time']['by_level'].items():
                if time_avg > 0:
                    hours = int(time_avg / 3600)
                    minutes = int((time_avg % 3600) / 60)
                    message += f"- {level.upper()}: {hours}h {minutes}m\n"
        
        # Añadir tendencia por día
        message += "\nTendencia diaria:\n"
        for day, count in sorted(stats['by_day'].items()):
            message += f"- {day}: {count}\n"
        
        # Enviar notificación
        return self.send_notification(
            notification_type='stats',
            subject=subject,
            message=message,
            data=stats,
            level='info',
            user_id=user_id
        )
    
    def send_system_alert(self, alert_data: Dict, level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta del sistema
        
        Args:
            alert_data: Datos de la alerta
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Extraer tipo de alerta
        alert_type = alert_data.get('alert_type', 'system')
        
        # Crear asunto según tipo
        if alert_type == 'api_limit':
            subject = f"Límite de API cercano: {alert_data.get('service')}"
        elif alert_type == 'error':
            subject = f"Error del sistema: {alert_data.get('error_type', 'Desconocido')}"
        elif alert_type == 'performance':
            subject = f"Problema de rendimiento: {alert_data.get('component', 'Sistema')}"
        else:
            subject = f"Alerta del sistema: {alert_type}"
        
        # Crear mensaje según tipo
        if alert_type == 'api_limit':
            message = (
                f"Se está alcanzando el límite de API para {alert_data.get('service')}.\n"
                f"Uso actual: {alert_data.get('current_usage')} de {alert_data.get('max_limit')} "
                f"({alert_data.get('usage_percent', 0):.1f}%)."
            )
        elif alert_type == 'error':
            message = (
                f"Se ha producido un error en el sistema: {alert_data.get('error_message', 'Desconocido')}.\n"
                f"Componente: {alert_data.get('component', 'Desconocido')}"
            )
        elif alert_type == 'performance':
            message = (
                f"Se ha detectado un problema de rendimiento en {alert_data.get('component', 'Sistema')}.\n"
                f"Métrica: {alert_data.get('metric', 'Desconocida')} = {alert_data.get('value', 'Desconocido')}"
            )
        else:
            message = f"Alerta del sistema: {alert_data.get('message', 'Sin detalles')}"
        
        # Formatear mensaje con detalles adicionales
        subject, message = self._format_alert_message(alert_type, subject, message, alert_data)
        
        # Enviar notificación
        return self.send_notification(
            notification_type=f"system_{alert_type}",
            subject=subject,
            message=message,
            data=alert_data,
            level=level,
            user_id=user_id
        )
    
    def send_content_alert(self, alert_data: Dict, level: str = 'warning', user_id: str = None) -> str:
        """
        Envía alerta relacionada con contenido
        
        Args:
            alert_data: Datos de la alerta
            level: Nivel de alerta
            user_id: ID del usuario destinatario
            
        Returns:
            ID de la notificación enviada
        """
        # Extraer tipo de alerta
        alert_type = alert_data.get('alert_type', 'content')
        
        # Crear asunto según tipo
        if alert_type == 'shadowban':
            subject = f"Posible shadowban detectado: {alert_data.get('platform')}"
        elif alert_type == 'saturation':
            subject = f"Saturación de nicho: {alert_data.get('niche')}"
        elif alert_type == 'viral_potential':
            subject = f"Oportunidad viral: {alert_data.get('content_id')}"
        elif alert_type == 'distribution':
            subject = f"Problema de distribución: {alert_data.get('platform')}"
        else:
            subject = f"Alerta de contenido: {alert_type}"
        
        # Crear mensaje según tipo
        if alert_type == 'shadowban':
            message = (
                f"Se ha detectado un posible shadowban en {alert_data.get('platform')}.\n"
                f"Contenido afectado: {alert_data.get('content_id')}\n"
                f"Métricas anómalas detectadas."
            )
        elif alert_type == 'saturation':
            message = (
                f"Se ha detectado saturación en el nicho {alert_data.get('niche')}.\n"
                f"Recomendamos diversificar o encontrar un ángulo diferente."
            )
        elif alert_type == 'viral_potential':
            message = (
                f"Se ha detectado potencial viral en el contenido {alert_data.get('content_id')}.\n"
                f"Puntuación de oportunidad: {alert_data.get('opportunity_score', 0):.2f}/1.0\n"
                f"Recomendamos promoción adicional."
            )
        elif alert_type == 'distribution':
            message = (
                f"Se ha detectado un problema de distribución en {alert_data.get('platform')}.\n"
                f"Contenido afectado: {alert_data.get('content_id')}\n"
                f"Métricas por debajo de lo esperado."
            )
        else:
            message = f"Alerta de contenido: {alert_data.get('message', 'Sin detalles')}"
        
        # Formatear mensaje con detalles adicionales
        subject, message = self._format_alert_message(alert_type, subject, message, alert_data)
        
        # Enviar notificación
        return self.send_notification(
            notification_type=f"content_{alert_type}",
            subject=subject,
            message=message,
            data=alert_data,
            level=level,
            user_id=user_id
        )
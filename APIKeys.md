# Claves API Requeridas para Content-Bot

## Plataformas de Contenido

### YouTube
- **API Key**: Para acceso básico a la API de YouTube
- **OAuth 2.0 Client ID**: Para autenticación de cuenta
- **OAuth 2.0 Client Secret**: Para autenticación de cuenta
- **Refresh Token**: Para mantener acceso sin reautenticación
- **Proceso de Obtención**: 
  1. Crear proyecto en [Google Cloud Console](https://console.cloud.google.com/)
  2. Habilitar YouTube Data API v3
  3. Crear credenciales OAuth 2.0
  4. Configurar pantalla de consentimiento
  5. Obtener tokens mediante flujo de autorización

### TikTok
- **App ID**: Identificador de aplicación
- **App Secret**: Clave secreta de aplicación
- **Access Token**: Token de acceso (temporal)
- **Proceso de Obtención**:
  1. Registrarse en [TikTok for Developers](https://developers.tiktok.com/)
  2. Crear una aplicación
  3. Solicitar acceso a Content Publishing API (requiere aprobación)
  4. Implementar flujo de autenticación OAuth

### Instagram
- **App ID**: Identificador de aplicación de Facebook
- **App Secret**: Clave secreta de aplicación
- **Access Token**: Token de acceso de larga duración
- **Proceso de Obtención**:
  1. Crear cuenta de desarrollador en [Meta for Developers](https://developers.facebook.com/)
  2. Crear una aplicación
  3. Configurar Instagram Graph API
  4. Solicitar permisos necesarios (requiere revisión)
  5. Implementar flujo de autenticación OAuth

### Twitter/X
- **API Key**: Clave de API
- **API Secret**: Clave secreta de API
- **Access Token**: Token de acceso
- **Access Token Secret**: Secreto del token de acceso
- **Proceso de Obtención**:
  1. Solicitar acceso a [Twitter API v2](https://developer.twitter.com/en/portal/dashboard)
  2. Crear un proyecto y aplicación
  3. Solicitar elevación a nivel 2 para acceso a funcionalidades avanzadas
  4. Generar tokens de acceso

### Bluesky
- **Handle**: Nombre de usuario
- **Password**: Contraseña (o App Password)
- **Proceso de Obtención**:
  1. Crear cuenta en [Bluesky](https://bsky.app/)
  2. Generar contraseña de aplicación en configuración

## Servicios de IA

### OpenAI (GPT-4o)
- **API Key**: Clave de API para acceso
- **Organization ID**: ID de organización (opcional)
- **Proceso de Obtención**:
  1. Registrarse en [OpenAI Platform](https://platform.openai.com/)
  2. Navegar a sección de API Keys
  3. Generar nueva clave

### Anthropic (Claude)
- **API Key**: Clave de API para acceso
- **Proceso de Obtención**:
  1. Registrarse en [Anthropic Console](https://console.anthropic.com/)
  2. Solicitar acceso a API
  3. Generar clave API

### ElevenLabs
- **API Key**: Clave de API para síntesis de voz
- **Proceso de Obtención**:
  1. Crear cuenta en [ElevenLabs](https://elevenlabs.io/)
  2. Suscribirse a un plan (gratuito disponible)
  3. Generar clave API en configuración

### Leonardo.ai
- **API Key**: Clave de API para generación de imágenes
- **Proceso de Obtención**:
  1. Registrarse en [Leonardo.ai](https://leonardo.ai/)
  2. Navegar a configuración de API
  3. Generar nueva clave

### Midjourney
- **Discord Bot Token**: Token para interactuar con Midjourney vía Discord
- **Proceso de Obtención**:
  1. Suscribirse a [Midjourney](https://www.midjourney.com/)
  2. Unirse al servidor de Discord
  3. Configurar bot personalizado (avanzado)

## Servicios de Análisis

### Google Analytics
- **Measurement ID**: ID para tracking
- **API Secret**: Clave secreta para API
- **Proceso de Obtención**:
  1. Crear cuenta en [Google Analytics](https://analytics.google.com/)
  2. Configurar propiedad
  3. Obtener ID de medición y secreto API

### Amplitude
- **API Key**: Clave para envío de eventos
- **Proceso de Obtención**:
  1. Registrarse en [Amplitude](https://amplitude.com/)
  2. Crear proyecto
  3. Obtener clave API

## Almacenamiento

### AWS S3
- **Access Key ID**: ID de clave de acceso
- **Secret Access Key**: Clave secreta de acceso
- **Region**: Región de AWS
- **Bucket Name**: Nombre del bucket
- **Proceso de Obtención**:
  1. Crear cuenta en [AWS](https://aws.amazon.com/)
  2. Crear usuario IAM con permisos S3
  3. Generar credenciales
  4. Crear bucket S3

## Bases de Datos

### MongoDB Atlas
- **Connection String**: URI de conexión
- **Username**: Usuario de base de datos
- **Password**: Contraseña
- **Proceso de Obtención**:
  1. Crear cuenta en [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
  2. Configurar cluster
  3. Crear usuario de base de datos
  4. Obtener string de conexión

### TimescaleDB
- **Host**: Dirección del servidor
- **Port**: Puerto (generalmente 5432)
- **Database**: Nombre de base de datos
- **Username**: Usuario
- **Password**: Contraseña
- **Proceso de Obtención**:
  1. Configurar instancia TimescaleDB (cloud o self-hosted)
  2. Crear usuario y base de datos
  3. Configurar reglas de firewall si es necesario
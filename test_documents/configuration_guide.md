# Configuration Guide

## Overview

This guide covers all configuration options for the application, including environment variables, configuration files, and runtime settings.

## Environment Variables

### Required Variables

The following environment variables must be set for the application to function:

#### DATABASE_URL
Database connection string.

**Format:** `postgresql://username:password@host:port/database`
**Example:** `postgresql://myuser:mypass@localhost:5432/myapp`

#### API_KEY
API key for external service authentication.

**Format:** String (32-64 characters)
**Example:** `sk-1234567890abcdef1234567890abcdef`

#### SECRET_KEY
Secret key for session management and encryption.

**Format:** String (minimum 32 characters)
**Example:** `your-secret-key-here-make-it-long-and-random`

### Optional Variables

#### LOG_LEVEL
Logging verbosity level.

**Values:** `DEBUG`, `INFO`, `WARN`, `ERROR`
**Default:** `INFO`
**Example:** `LOG_LEVEL=DEBUG`

#### PORT
Application port number.

**Format:** Integer (1-65535)
**Default:** `8000`
**Example:** `PORT=3000`

#### REDIS_URL
Redis connection string for caching.

**Format:** `redis://host:port/database`
**Default:** `redis://localhost:6379/0`
**Example:** `REDIS_URL=redis://redis-server:6379/1`

#### SMTP_HOST
SMTP server for email notifications.

**Format:** Hostname or IP address
**Example:** `SMTP_HOST=smtp.gmail.com`

#### SMTP_PORT
SMTP server port.

**Format:** Integer
**Default:** `587`
**Example:** `SMTP_PORT=465`

#### SMTP_USERNAME
SMTP authentication username.

**Example:** `SMTP_USERNAME=your-email@gmail.com`

#### SMTP_PASSWORD
SMTP authentication password.

**Example:** `SMTP_PASSWORD=your-app-password`

## Configuration Files

### app.yaml

Main application configuration file located at `config/app.yaml`.

```yaml
# Application settings
app:
  name: "My Application"
  version: "1.0.0"
  debug: false
  timezone: "UTC"

# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30

# Database settings
database:
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo: false

# Cache settings
cache:
  ttl: 3600  # Time to live in seconds
  max_entries: 10000
  cleanup_interval: 300

# Security settings
security:
  session_timeout: 1800  # 30 minutes
  max_login_attempts: 5
  lockout_duration: 900  # 15 minutes
  password_min_length: 8
  require_special_chars: true
```

### database.yaml

Database-specific configuration at `config/database.yaml`.

```yaml
# Database connection settings
connection:
  driver: "postgresql"
  host: "localhost"
  port: 5432
  database: "myapp"
  username: "myuser"
  password: "mypass"
  ssl_mode: "prefer"

# Connection pool settings
pool:
  min_connections: 5
  max_connections: 20
  connection_timeout: 10
  idle_timeout: 300
  max_lifetime: 3600

# Migration settings
migrations:
  auto_migrate: false
  migration_path: "./migrations"
  schema_version_table: "schema_versions"

# Backup settings
backup:
  enabled: true
  interval: "daily"
  retention_days: 30
  backup_path: "./backups"
```

### logging.yaml

Logging configuration at `config/logging.yaml`.

```yaml
# Logging configuration
logging:
  version: 1
  disable_existing_loggers: false
  
  formatters:
    standard:
      format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    detailed:
      format: "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    json:
      format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
  
  handlers:
    console:
      class: logging.StreamHandler
      level: INFO
      formatter: standard
      stream: ext://sys.stdout
    
    file:
      class: logging.handlers.RotatingFileHandler
      level: DEBUG
      formatter: detailed
      filename: ./logs/app.log
      maxBytes: 10485760  # 10MB
      backupCount: 5
    
    error_file:
      class: logging.handlers.RotatingFileHandler
      level: ERROR
      formatter: json
      filename: ./logs/error.log
      maxBytes: 10485760
      backupCount: 10
  
  loggers:
    myapp:
      level: DEBUG
      handlers: [console, file]
      propagate: false
    
    myapp.database:
      level: INFO
      handlers: [file]
      propagate: false
    
    uvicorn:
      level: INFO
      handlers: [console]
      propagate: false
  
  root:
    level: WARNING
    handlers: [console, error_file]
```

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "myapp"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  postgres_data:
  redis_data:
```

## Deployment Configuration

### Production Settings

For production deployment, use these additional configurations:

#### Environment Variables
```bash
# Production environment
NODE_ENV=production
DEBUG=false
LOG_LEVEL=WARN

# Security
SECURE_COOKIES=true
HTTPS_ONLY=true
CSRF_PROTECTION=true

# Performance
CACHE_TTL=7200
WORKER_PROCESSES=auto
```

#### Nginx Configuration

```nginx
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Configuration Validation

The application validates configuration on startup:

### Required Configuration Check
- All required environment variables present
- Database connectivity
- Redis connectivity (if enabled)
- File system permissions

### Configuration Errors

Common configuration errors and solutions:

#### Database Connection Failed
- Verify DATABASE_URL format
- Check database server status
- Confirm credentials and permissions

#### Redis Connection Failed  
- Verify REDIS_URL format
- Check Redis server status
- Confirm network connectivity

#### Invalid Configuration Values
- Check data types (string, integer, boolean)
- Verify allowed values for enums
- Confirm required fields are present

## Configuration Best Practices

### Security
- Use environment variables for sensitive data
- Never commit secrets to version control
- Rotate API keys and passwords regularly
- Use strong, unique secret keys

### Performance
- Set appropriate connection pool sizes
- Configure cache TTL based on data patterns
- Monitor resource usage and adjust limits
- Use compression for large configurations

### Monitoring
- Enable appropriate log levels for environment
- Set up log rotation to prevent disk space issues
- Monitor configuration changes
- Alert on configuration errors

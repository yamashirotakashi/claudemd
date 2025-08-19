# API Reference

## Overview

This API provides comprehensive access to the system's functionality through RESTful endpoints.

## Base URL

```
https://api.example.com/v1
```

## Authentication

All API requests require authentication using an API key or JWT token.

### API Key Authentication

Include the API key in the request header:

```
Authorization: Bearer YOUR_API_KEY
```

### JWT Authentication

For user-specific operations, use JWT tokens:

```
Authorization: JWT YOUR_JWT_TOKEN
```

## Endpoints

### Users

#### GET /users

Retrieve a list of users.

**Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Number of users per page (default: 20)
- `search` (optional): Search term for filtering users

**Response:**
```json
{
  "users": [
    {
      "id": "12345",
      "username": "john_doe",
      "email": "john@example.com",
      "created_at": "2023-01-15T10:30:00Z",
      "last_login": "2023-08-15T14:22:00Z"
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 20
}
```

#### POST /users

Create a new user.

**Request Body:**
```json
{
  "username": "jane_doe",
  "email": "jane@example.com",
  "password": "secure_password123",
  "first_name": "Jane",
  "last_name": "Doe"
}
```

**Response:**
```json
{
  "id": "12346",
  "username": "jane_doe",
  "email": "jane@example.com",
  "created_at": "2023-08-19T15:45:00Z"
}
```

#### GET /users/{id}

Retrieve a specific user by ID.

**Parameters:**
- `id` (required): User ID

**Response:**
```json
{
  "id": "12345",
  "username": "john_doe",
  "email": "john@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "created_at": "2023-01-15T10:30:00Z",
  "last_login": "2023-08-15T14:22:00Z",
  "profile": {
    "bio": "Software developer",
    "website": "https://johndoe.dev",
    "location": "San Francisco, CA"
  }
}
```

#### PUT /users/{id}

Update a user.

**Parameters:**
- `id` (required): User ID

**Request Body:**
```json
{
  "first_name": "Johnny",
  "last_name": "Doe",
  "profile": {
    "bio": "Senior Software Developer",
    "website": "https://johnnydoe.dev"
  }
}
```

#### DELETE /users/{id}

Delete a user.

**Parameters:**
- `id` (required): User ID

**Response:**
```json
{
  "message": "User deleted successfully"
}
```

### Data Records

#### GET /data

Retrieve data records.

**Parameters:**
- `page` (optional): Page number
- `limit` (optional): Records per page
- `type` (optional): Data type filter
- `from_date` (optional): Start date filter (ISO 8601)
- `to_date` (optional): End date filter (ISO 8601)

**Response:**
```json
{
  "records": [
    {
      "id": "record_001",
      "type": "measurement",
      "value": 42.5,
      "unit": "celsius",
      "timestamp": "2023-08-19T10:15:00Z",
      "metadata": {
        "sensor_id": "temp_001",
        "location": "room_a"
      }
    }
  ],
  "total": 1000,
  "page": 1,
  "limit": 50
}
```

#### POST /data

Create a new data record.

**Request Body:**
```json
{
  "type": "measurement",
  "value": 38.2,
  "unit": "celsius",
  "metadata": {
    "sensor_id": "temp_002",
    "location": "room_b"
  }
}
```

### Notifications

#### GET /notifications

Retrieve user notifications.

**Parameters:**
- `page` (optional): Page number
- `limit` (optional): Notifications per page
- `read` (optional): Filter by read status (true/false)

**Response:**
```json
{
  "notifications": [
    {
      "id": "notif_001",
      "title": "System Alert",
      "message": "Temperature threshold exceeded",
      "type": "alert",
      "read": false,
      "created_at": "2023-08-19T12:30:00Z"
    }
  ]
}
```

#### POST /notifications

Send a notification.

**Request Body:**
```json
{
  "user_id": "12345",
  "title": "Welcome",
  "message": "Welcome to our platform!",
  "type": "info"
}
```

#### PUT /notifications/{id}/read

Mark a notification as read.

**Parameters:**
- `id` (required): Notification ID

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error responses include a JSON body with error details:

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "The 'email' parameter is required",
    "details": {
      "parameter": "email",
      "expected_format": "valid email address"
    }
  }
}
```

## Rate Limiting

API requests are rate-limited:

- **Standard users**: 1000 requests per hour
- **Premium users**: 5000 requests per hour
- **Enterprise users**: 10000 requests per hour

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1692456000
```

## SDK and Libraries

Official SDKs available for:

### Python
```python
from myapi import Client

client = Client(api_key="your_api_key")
users = client.users.list(page=1, limit=20)
```

### JavaScript
```javascript
import { ApiClient } from '@myapi/sdk';

const client = new ApiClient('your_api_key');
const users = await client.users.list({ page: 1, limit: 20 });
```

### cURL Examples

#### Get users
```bash
curl -X GET "https://api.example.com/v1/users?page=1&limit=20" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Create user
```bash
curl -X POST "https://api.example.com/v1/users" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "new_user",
    "email": "new@example.com",
    "password": "secure_pass"
  }'
```

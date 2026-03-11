# Deployment Guide — Legally Platform

This guide covers deploying all three layers of the Legally platform to production:

| Layer | Service | Port |
|---|---|---|
| Python AI Engine | Render / Railway / VPS | 8000 |
| Go Backend | Render / Railway / VPS | 8080 |
| React Frontend | Vercel / Netlify / Static hosting | 80/443 |

---

## Prerequisites

Before deploying, you must have:
- A Pinecone index built and populated (run `build_vector_db.py` locally first)
- Valid API keys: `PINECONE_API_KEY`, `GROQ_API_KEY`
- A MongoDB Atlas cluster URI (`MONGO_URI`)
- A strong `JWT_SECRET` (generate: `openssl rand -hex 64`)

---

## Part 1 — Python AI Engine (FastAPI)

### Option A: Render.com (Recommended)

**1. Create a Web Service on Render**
- Connect your Git repo
- Branch: `main` (or your production branch)
- Environment: **Python**
- Build Command:
  ```
  pip install -r ai_service/requirements.txt
  ```
- Start Command:
  ```
  uvicorn ai_service.api.api:app --host 0.0.0.0 --port $PORT
  ```

**2. Set Environment Variables in Render dashboard:**

| Variable | Required | Value |
|---|---|---|
| `PINECONE_API_KEY` | ✅ | Your Pinecone key |
| `PINECONE_INDEX_NAME` | ✅ | `legally-index` |
| `PINECONE_NAMESPACE` | ✅ | `default` |
| `GROQ_API_KEY` | ✅ | Your Groq key |
| `HF_TOKEN` | recommended | Your HuggingFace token |
| `HF_HUB_OFFLINE` | optional | `1` if model is pre-cached |
| `LEGAL_RAG_HF_LOCAL_ONLY` | optional | `1` if model is pre-cached |
| `LEGAL_RAG_USE_RERANKER` | optional | `0` to disable reranker (saves RAM) |
| `LEGAL_RAG_LLM` | optional | `llama-3.1-8b-instant` |

> **Note:** Render's free tier has ephemeral disks. The HuggingFace model (`multilingual-e5-large`) downloads on every cold start (~1GB). Use `HF_TOKEN` to avoid anonymous rate limits.

**3. `render.yaml` Blueprint (optional):**
```yaml
services:
  - type: web
    name: legally-ai-engine
    env: python
    plan: starter
    buildCommand: pip install -r ai_service/requirements.txt
    startCommand: uvicorn ai_service.api.api:app --host 0.0.0.0 --port $PORT
    autoDeploy: true
    envVars:
      - key: PINECONE_API_KEY
        sync: false
      - key: PINECONE_INDEX_NAME
        value: legally-index
      - key: GROQ_API_KEY
        sync: false
      - key: HF_TOKEN
        sync: false
      - key: LEGAL_RAG_USE_RERANKER
        value: "0"
```

### Option B: Docker

**`Dockerfile` for AI Engine:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY ai_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "ai_service.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t legally-ai .
docker run -p 8000:8000 --env-file .env legally-ai
```

### Option C: VPS (Ubuntu/Fedora)

```bash
# Clone repo
git clone https://github.com/zloo00/LegalRAG.git
cd LegalRAG

# Create venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r ai_service/requirements.txt

# Create systemd service
sudo nano /etc/systemd/system/legally-ai.service
```

```ini
[Unit]
Description=Legally AI Engine
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/LegalRAG
EnvironmentFile=/home/ubuntu/LegalRAG/.env
ExecStart=/home/ubuntu/LegalRAG/venv/bin/uvicorn ai_service.api.api:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable legally-ai
sudo systemctl start legally-ai
sudo systemctl status legally-ai
```

---

## Part 2 — Go Backend

### Option A: Render.com

- Environment: **Go**
- Build Command: `go build -o server ./backend/legally`
- Start Command: `./server`
- Root Directory: `backend/legally`

Environment variables:

| Variable | Required | Description |
|---|---|---|
| `MONGO_URI` | ✅ | MongoDB Atlas connection string |
| `DB_NAME` | ✅ | Database name (`legally_bot`) |
| `JWT_SECRET` | ✅ | Long random secret for signing JWTs |
| `ADMIN_IDS` | ✅ | Comma-separated admin user IDs |
| `AI_ENGINE_URL` | ✅ | Internal URL of the Python engine (`http://localhost:8000`) |

### Option B: Docker

```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY backend/legally/ .
RUN go build -o server .

FROM alpine:latest
WORKDIR /root/
COPY --from=builder /app/server .
EXPOSE 8080
CMD ["./server"]
```

### Option C: VPS (systemd)

```ini
[Unit]
Description=Legally Go Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/LegalRAG/backend/legally
EnvironmentFile=/home/ubuntu/LegalRAG/.env
ExecStart=/home/ubuntu/LegalRAG/backend/legally/server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## Part 3 — React Frontend

### Option A: Vercel (Recommended)

```bash
cd frontend/legally-app
npm run build
# Push to GitHub → Vercel auto-deploys on merge to main
```

Set environment variables in Vercel dashboard:
```
REACT_APP_API_URL=https://your-go-backend.onrender.com
```

### Option B: Netlify

```bash
npm run build
# Drag and drop the build/ folder to netlify.com
# Or connect GitHub repo: Build command: npm run build, Publish dir: build
```

### Option C: Serve from Go Backend (Self-Contained)

The Go backend already serves the React build at `GET /` via its static file handler. To use this:

```bash
cd frontend/legally-app
npm run build
cp -r build/ ../../backend/legally/static/
```

Then access the full app on `http://localhost:8080`.

---

## Nginx Reverse Proxy (VPS Production)

Put all services behind Nginx for SSL termination and routing:

```nginx
server {
    listen 443 ssl;
    server_name legally.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/legally.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/legally.yourdomain.com/privkey.pem;

    # React Frontend (or serve Go directly)
    location / {
        root /var/www/legally-app/build;
        try_files $uri /index.html;
    }

    # Go Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Python AI Engine (internal only — DO NOT expose publicly)
    # location /ai/ { proxy_pass http://127.0.0.1:8000; }
}
```

> ⚠️ The Python AI Engine (`port 8000`) must **never** be publicly accessible — it has no authentication layer. Use a firewall rule (`ufw deny 8000`) and communicate Go→Python only on `localhost`.

---

## Pinecone Index Setup (One-Time)

Must be run before any deployment. Run locally with your Pinecone key:

```bash
source venv/bin/activate
export PINECONE_API_KEY="pcsk_..."
export PINECONE_INDEX_NAME="legally-index"

# Index all documents
./venv/bin/python ai_service/retrieval/build_vector_db.py

# Build BM25 corpus
./venv/bin/python -m ai_service.processing.prepare_data
```

Expected result in Pinecone console: **~4,200 vectors**, dimension `1024`, metric `cosine`.

---

## HuggingFace Embedding Strategy

The `multilingual-e5-large` model (~560M params, ~1.1GB) is downloaded from HuggingFace Hub at first startup.

| Strategy | Speed | Reliability | Setup |
|---|---|---|---|
| **Online HF download** | Slow on cold start | Depends on HF | Set `HF_TOKEN` |
| **Pre-cached, online verify** | Medium | High | Download once, set `HF_TOKEN` |
| **Offline (cached)** | Fast | Very High | Mount cached model, set `HF_HUB_OFFLINE=1` |

**Pre-cache locally:**
```bash
./venv/bin/python -c "
from langchain_huggingface import HuggingFaceEmbeddings
HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
print('Model cached.')
"
```

Then on the server: set `HF_HUB_OFFLINE=1` and `LEGAL_RAG_HF_LOCAL_ONLY=1`.

---

## Health Checks

| Service | Endpoint | Expected |
|---|---|---|
| Python AI Engine | `GET http://localhost:8000/api/v1/stats` | `200 OK` with JSON |
| Go Backend | `GET http://localhost:8080/health` | `200 OK` |
| React Frontend | `GET http://localhost:3000` | HTML page |

---

## Production Checklist

- [ ] Pinecone index built and populated (`~4,200 vectors`)
- [ ] All required env vars set in deployment platform
- [ ] `JWT_SECRET` is long (≥64 chars) and unique
- [ ] Python AI Engine is NOT publicly exposed (firewall rule on port 8000)
- [ ] HuggingFace token set to avoid anonymous rate limits
- [ ] CORS configured in Go backend to allow your frontend domain only
- [ ] MongoDB Atlas access list includes deployment server IP
- [ ] Nginx SSL configured with valid certificate
- [ ] Systemd services set to restart on failure

---

## Rollback

**Render:** Use the "Rollback" button in the Render dashboard to any previous deploy.

**VPS / Git:**
```bash
git log --oneline -10          # find the last good commit
git checkout <commit-hash>     # switch to it
sudo systemctl restart legally-ai legally-go
```

---

## Common Failure Modes

| Error | Cause | Fix |
|---|---|---|
| `CRITICAL ERROR: Missing Configuration` | `.env` not loaded / key missing | Check all required vars are set in platform |
| `ReadTimeoutError` from HuggingFace | HF Hub slow or blocked | Set `HF_TOKEN`, or enable offline mode |
| `ValueError: Pinecone API key must be provided` | Old code path | Ensure using updated `get_vector_store()` lazy init |
| `cannot import name 'is_torch_fx_available'` | transformers/FlagEmbedding mismatch | Set `LEGAL_RAG_USE_RERANKER=0` or upgrade transformers |
| `dial tcp: connection refused` (Go→Python) | AI engine not running | Start Python engine first, check `AI_ENGINE_URL` |
| `401 Unauthorized` on all API calls | JWT secret mismatch between deploys | Ensure `JWT_SECRET` is identical across restarts |
| MongoDB `authentication failed` | Wrong URI or IP not whitelisted | Check Atlas access list and MONGO_URI format |

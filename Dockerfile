# Image for the web app: the FastAPI backend plus the static page it serves.
# FalkorDB is a separate service (see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend

# server.py serves ../frontend, so the process runs from backend/.
WORKDIR /app/backend

# The app does not need root.
RUN useradd --create-home app
USER app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

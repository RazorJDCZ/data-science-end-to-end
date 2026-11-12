FROM python:3.11-slim

WORKDIR /app

COPY servicio/requirements.txt ./servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt

COPY modelado/ ./modelado/
COPY servicio/api.py servicio/modelo.joblib ./servicio/

EXPOSE 8000

CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]

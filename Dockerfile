# Oficjalny obraz Pythona
FROM python:3.11-slim

# Katalog roboczy
WORKDIR /app

# Kopiowanie plików
COPY . .

# Instalacja zależności z pliku requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Serwer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


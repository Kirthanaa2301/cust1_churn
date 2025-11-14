FROM python:3.10-slim
WORKDIR /app
COPY requirements_fastapi.txt /app/
RUN apt-get update && apt-get install -y build-essential && pip install --no-cache-dir -r requirements_fastapi.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

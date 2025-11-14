FROM python:3.10-slim
WORKDIR /app
COPY bento_requirements.txt /app/
RUN apt-get update && apt-get install -y build-essential && pip install --no-cache-dir -r bento_requirements.txt
COPY . /app
# expose Bento default (we will use bentoml serve when container built in CI)
EXPOSE 3000
CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]

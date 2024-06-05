FROM python:3.9-slim

WORKDIR /app

COPY src/visualisation/dockerF/requirements.txt ./

COPY src/visualisation/onEdgeDocker ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src/models/ ./model/

EXPOSE 5000

CMD ["python", "app.py"]
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Users must add torch geometric per-platform
COPY . /app
ENV PYTHONPATH=/app

CMD ["bash"]

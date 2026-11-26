FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 STARPLATE_DATA=/data OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
ARG STARPLATE_PIP_INDEX=https://pypi.org/simple
COPY requirements.txt .
RUN python -m pip install --index-url "$STARPLATE_PIP_INDEX" --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt
COPY starplate starplate
COPY static static
COPY samples samples
COPY app.py .
RUN mkdir -p /data
EXPOSE 8080
CMD ["python", "app.py"]

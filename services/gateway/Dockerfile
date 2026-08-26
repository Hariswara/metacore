FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY gateway ./gateway
RUN uv pip install --system --no-cache .
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 1: train ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py          # produces artifacts/model.pkl + artifacts/metrics.json

# ── Stage 2: serve ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app.py run_model.py ./

# Copy trained model from build stage
COPY --from=trainer /app/artifacts ./artifacts

EXPOSE 5001

CMD ["python", "app.py"]

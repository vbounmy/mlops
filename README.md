# Vélib MLOps — Prédiction de disponibilité des vélos

Projet MLOps complet basé sur le dataset **Vélib - Vélos et bornes - Disponibilité temps réel** ([opendata.paris.fr](https://opendata.paris.fr)).

## Objectif

Prédire si une station Vélib a **au moins un vélo disponible** (classification binaire).

| Feature | Description |
|---|---|
| `capacity` | Capacité totale de la station |
| `numbikesavailable` | Vélos disponibles actuellement |
| `numdocksavailable` | Bornes libres actuellement |
| `mechanical` | Vélos mécaniques disponibles |
| `ebike` | Vélos électriques disponibles |

**Target** : `is_available` → 1 si ≥ 1 vélo dispo, sinon 0

## Lancement rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle
python train.py

# 3. Démarrer l'API
python app.py
```

## API

```bash
# Health check
curl http://localhost:5001/health

# Prédiction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [30, 5, 10, 3, 2]}'
```

## Docker

```bash
docker build -t velib-mlops .
docker run -p 5001:5001 velib-mlops
```

## CI/CD (GitHub Actions)

| Branche | Jobs |
|---|---|
| `feature/*` | install → train |
| `develop` | install → train → build Docker → push GHCR |

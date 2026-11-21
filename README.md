# Resource Optimization System — Neural Backend

This repository contains the neural forecasting backend for the Resource Optimization System. It accepts authenticated system telemetry, predicts near-term CPU utilization, and exposes account, status, telemetry, and installer-download endpoints through Flask.

## Repository contents

- `Backend/app.py` — Flask API, authentication, telemetry processing, and model inference.
- `Backend/model/model.h5` — trained Keras model loaded by the API.
- `Backend/model/scaler.save` — fitted feature scaler used before inference.
- `Backend/model/train.py` — model training script.
- `Backend/collector_buffer.py` — sliding-window buffer helper.
- `Backend/downloads/` — Linux `.deb` and Windows `.exe` agent installers.

## Requirements

- Python 3.10 or later (use a Python version supported by the pinned TensorFlow release).
- A MongoDB connection for persistent accounts, or opt-in local SQLite fallback for development.
- A configured JWT secret.
- Model and scaler artifacts in `Backend/model/` (included in this repository).

## Local setup

```bash
cd Backend
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and configure environment variables:

```bash
pip install -r requirements.txt
```

Set `SECRET_KEY` to a long, random value before starting the server. Configure either `MONGO_URI` (recommended) or enable local development storage with `LOCAL_DB_FALLBACK=true`. Optional variables include `PORT`, `CORS_ORIGINS`, `MODEL_PATH`, `MONGO_CONNECT_TIMEOUT_MS`, `MONGO_SERVER_SELECTION_TIMEOUT_MS`, and `LOCAL_DB_PATH`.

Example for macOS/Linux development (do not use this sample secret in production):

```bash
export SECRET_KEY="replace-with-a-long-random-secret"
export LOCAL_DB_FALLBACK=true
python app.py
```

The API listens on `http://localhost:5000` by default. Keep secrets in environment variables or a local `.env` file that is not committed. If using a `.env` file, the backend loader reads `KEY=value` lines from `Backend/.env`.

## API overview

| Method | Path | Authentication | Purpose |
| --- | --- | --- | --- |
| `GET` | `/status` | No | Reports API, database, and model readiness. |
| `POST` | `/signup` | No | Creates an account. JSON body: `name`, `email`, `password`. |
| `POST` | `/login` | No | Authenticates an account and returns a JWT. |
| `POST` | `/analyze` | Bearer token | Accepts system telemetry and returns a CPU forecast and recommendations. |
| `GET` | `/client-system` | Bearer token | Returns most recent telemetry and in-memory history. |
| `GET` | `/predicted` | Bearer token | Returns the most recent prediction. |
| `DELETE` | `/delete-account` | Bearer token | Deletes the authenticated account. |
| `GET` | `/download/<filename>` | No | Downloads one of the allowlisted agent installers. |

For authenticated routes, send `Authorization: Bearer <token>`. The telemetry payload to `/analyze` should include numeric values for `cpu_usage`, `memory_usage`, `net_upload_mbps`, `net_download_mbps`, `disk_read_mbps`, and `disk_write_mbps`. The model uses a 10-observation sequence; early observations are padded from the first available sample.

The `/status` response includes internal readiness details for deployment diagnostics. Do not expose this endpoint publicly if those details are sensitive in your deployment.

## Model notes

The API loads `MODEL_PATH` when set, otherwise it checks `/models/model.h5` and then `Backend/model/model.h5`. The scaler is loaded from `Backend/model/scaler.save`. Ensure the model and scaler were trained with the same six telemetry features and feature ordering used by `/analyze`.

To train or regenerate the artifacts, inspect `Backend/model/train.py` and provide its expected input data. Training is separate from API startup; existing model artifacts are used for inference.

## Security and operations

- Use a unique, high-entropy `SECRET_KEY`; do not commit credentials or secrets.
- Configure `CORS_ORIGINS` as a comma-separated allowlist of exact frontend origins in production.
- Keep `LOCAL_DB_FALLBACK=true` for local development only. Production should use a persistent MongoDB service.
- Telemetry and prediction history are currently held in process memory and are lost on restart; with multiple workers, each worker has separate in-memory state.
- Use a production WSGI server such as Gunicorn rather than Flask's development server when deploying. Set `PORT` as required by the hosting platform.
- Installer binaries are executable artifacts; distribute them only through trusted release and deployment channels.

## License

This project is distributed under the MIT License. See `LICENSE`.

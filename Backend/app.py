from flask import Flask, jsonify, request, send_from_directory, abort
import os
import sqlite3
import numpy as np
import datetime
import jwt
from functools import wraps
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.security import generate_password_hash, check_password_hash
import joblib

def load_local_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"')

load_local_env()

try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
CORS(app, origins=allowed_origins)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Vary"] = "Origin"
    return response

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be configured in the environment")
mongo_uri_file = os.getenv("MONGO_URI_FILE")
if mongo_uri_file:
    if not os.path.isabs(mongo_uri_file):
        mongo_uri_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), mongo_uri_file)
        )
    try:
        with open(mongo_uri_file, "r", encoding="utf-8") as uri_file:
            uri_lines = [
                line.strip()
                for line in uri_file
                if line.strip() and not line.lstrip().startswith("#")
            ]
        MONGO_URI = next(
            (
                line
                for line in uri_lines
                if line.startswith(("mongodb://", "mongodb+srv://"))
            ),
            None,
        )
        if not MONGO_URI:
            raise ValueError("No MongoDB URI found in MONGO_URI_FILE")
    except OSError as exc:
        raise RuntimeError("MONGO_URI_FILE could not be read") from exc
else:
    MONGO_URI = os.getenv("MONGO_URI")
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "downloads")
users_collection = None
mongo_client = None
database_error = None
database_backend = None


class SQLiteUsersCollection:
    """Local auth-store fallback for development when MongoDB is unavailable."""

    def __init__(self, database_path):
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def create_index(self, *_args, **_kwargs):
        return None

    def find_one(self, query):
        row = self.connection.execute(
            "SELECT id, name, email, password, created_at FROM users WHERE email = ? LIMIT 1",
            (query.get("email"),),
        ).fetchone()
        if row is None:
            return None
        return {
            "_id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "password": row["password"],
            "created_at": row["created_at"],
        }

    def insert_one(self, document):
        self.connection.execute(
            "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (
                document["name"],
                document["email"],
                document["password"],
                document["created_at"].isoformat(),
            ),
        )
        self.connection.commit()

    def delete_one(self, query):
        cursor = self.connection.execute(
            "DELETE FROM users WHERE email = ?", (query.get("email"),)
        )
        self.connection.commit()
        return type("DeleteResult", (), {"deleted_count": cursor.rowcount})()


def connect_to_local_database():
    global users_collection, database_backend
    database_path = os.getenv(
        "LOCAL_DB_PATH",
        os.path.join(os.path.dirname(__file__), "local_users.sqlite3"),
    )
    users_collection = SQLiteUsersCollection(database_path)
    database_backend = "sqlite-local"
    print("INFO: Using local SQLite database for development")
    return True


def connect_to_mongodb():
    """Connect to MongoDB and keep the failure state safe for the API."""
    global mongo_client, users_collection, database_error, database_backend

    if not MONGO_URI:
        database_error = "MONGO_URI is not configured"
        users_collection = None
        database_backend = None
        return connect_to_local_database() if os.getenv("LOCAL_DB_FALLBACK", "false").lower() == "true" else False

    try:
        mongo_client = MongoClient(
            MONGO_URI,
            connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "5000")),
            serverSelectionTimeoutMS=int(
                os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "10000")
            ),
        )
        mongo_client.admin.command("ping")
        db = mongo_client["ros"]
        collection = db["userdata"]
        collection.create_index("email", unique=True)
        users_collection = collection
        database_error = None
        database_backend = "mongodb"
        print("SUCCESS: MongoDB connected successfully")
        return True
    except PyMongoError as exc:
        database_error = type(exc).__name__
        users_collection = None
        print(f"ERROR: MongoDB connection failed ({database_error})")
        return connect_to_local_database() if os.getenv("LOCAL_DB_FALLBACK", "false").lower() == "true" else False

print(f"DEBUG: Attempting to connect to MongoDB URI")
connect_to_mongodb()
DEFAULT_MODEL_CANDIDATES = [
    "/models/model.h5",
    os.path.join(os.path.dirname(__file__), "model", "model.h5"),
]
SEQUENCE_LENGTH = 10
MODEL_LOADED = False
MODEL_LOAD_ERROR = None
MODEL_PATH = None
try:
    env_model_path = os.getenv("MODEL_PATH")
    candidate_paths = [env_model_path] if env_model_path else DEFAULT_MODEL_CANDIDATES
    MODEL_PATH = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not MODEL_PATH:
        raise FileNotFoundError("Model file not found")
    model = load_model(MODEL_PATH, compile=False)
    MODEL_LOADED = True
    print("AI Model loaded successfully")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print("Model load failed:", e)
    model = None
try:
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.save")
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully from:", scaler_path)
except Exception as e:
    print("Scaler load failed:", e)
    scaler = None
USER_SESSIONS = {}
def get_user_session(email):
    if email not in USER_SESSIONS:
        USER_SESSIONS[email] = {
            "LAST_SYSTEM_DATA": None,
            "SYSTEM_HISTORY": [],
            "LAST_PREDICTION": None,
            "FEATURE_BUFFER": [],
            "LAST_TELEMETRY_AT": None,
        }
    return USER_SESSIONS[email]

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"message": "Token missing"}), 401
        try:
            token = auth_header.split(" ")[1]
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = decoded
        except Exception:
            return jsonify({"message": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/download/<path:filename>")
def download_file(filename):
    allowed_files = {
        "resource-optimization-agent-1.0.1.exe",
        "resource-optimization-agent-1.0.1.deb",
    }
    if filename not in allowed_files:
        abort(404)
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

@app.route("/signup", methods=["POST"])
def signup():
    if users_collection is None and not connect_to_mongodb():
        return jsonify({"error": "Database not connected", "database_error": database_error}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 409
    hashed_password = generate_password_hash(password)
    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    })
    return jsonify({"status": "registered"})

@app.route("/login", methods=["POST"])
def login():
    if users_collection is None and not connect_to_mongodb():
        return jsonify({"error": "Database not connected", "database_error": database_error}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"message": "Missing email or password"}), 400
    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid login"}), 401
    token = jwt.encode({
        "user_id": str(user["_id"]),
        "email": user["email"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")
    return jsonify({
        "token": token,
        "user": {
            "name": user.get("name", "User"),
            "email": user.get("email")
        }
    })




@app.route("/analyze", methods=["POST"])

@token_required
def analyze():
    user_email = request.user["email"]
    print(f"DEBUG: Received /analyze request from {user_email}")
    session = get_user_session(user_email)

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    session["LAST_SYSTEM_DATA"] = data
    session["LAST_TELEMETRY_AT"] = datetime.datetime.now(datetime.timezone.utc)

    if not MODEL_LOADED or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500
    
    # Ensure ram_components exists for the dashboard
    if "ram_components" not in data:
        total = float(data.get("memory_total_gb", 1))
        used = float(data.get("memory_used_gb", 0))
        free = float(data.get("memory_free_gb", 0))
        data["ram_components"] = {
            "active": used,
            "inactive": 0,
            "cached": 0,
            "free": free,
            "total": total
        }

    session["SYSTEM_HISTORY"].append({
        "time": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
        "cpu": data.get("cpu_usage", 0),
        "ram": data.get("memory_usage", 0),
        "net_up": data.get("net_upload_mbps", 0),
        "net_down": data.get("net_download_mbps", 0)
    })

    if len(session["SYSTEM_HISTORY"]) > 30:
        session["SYSTEM_HISTORY"].pop(0)

    try:
        cpu_val = float(data.get("cpu_usage", 0))
        mem_val = float(data.get("memory_usage", 0))

        raw_features = [
            cpu_val,
            mem_val,
            float(data.get("net_upload_mbps", 0)),
            float(data.get("net_download_mbps", 0)),
            float(data.get("disk_read_mbps", 0)),
            float(data.get("disk_write_mbps", 0))
        ]
        session["FEATURE_BUFFER"].append(raw_features)

        if len(session["FEATURE_BUFFER"]) > SEQUENCE_LENGTH:
            session["FEATURE_BUFFER"].pop(0)

        if len(session["FEATURE_BUFFER"]) < SEQUENCE_LENGTH:
            padded = [session["FEATURE_BUFFER"][0]] * (SEQUENCE_LENGTH - len(session["FEATURE_BUFFER"])) + session["FEATURE_BUFFER"]
        else:
            padded = session["FEATURE_BUFFER"]

        future_cpu = cpu_val
        if scaler and model:
            scaled = scaler.transform(padded)
            sequence = np.array(scaled).reshape(1, SEQUENCE_LENGTH, len(raw_features))
            prediction = model.predict(sequence, verbose=0)
            future_cpu = float(prediction[0][0]) * 100
        
        top_proc_name = "N/A"
        top_proc_cpu = 0
        if psutil:
            try:
                processes = list(psutil.process_iter(['name', 'cpu_percent']))
                if processes:
                    top_p = max(processes, key=lambda x: x.info['cpu_percent'])
                    top_proc_name = top_p.info['name']
                    top_proc_cpu = top_p.info['cpu_percent']
            except: pass

        intelligence_report = []
        
        if future_cpu > 80:
            state = "Critical Saturation Imminent"
            summary = f"CRITICAL ALERT: Our neural engine predicts a severe resource spike. Total CPU utilization is projected to hit {future_cpu:.1f}% within the next observation window."
        elif future_cpu > 50:
            state = "High Load Warning"
            summary = f"SYSTEM WARNING: Sustained high-load patterns detected. Expect resource pressure to stabilize around {future_cpu:.1f}%."
        elif future_cpu > 20:
            state = "Active Optimization"
            summary = f"STABLE: The system is operating within balanced parameters. Projected overhead is nominal at {future_cpu:.1f}%."
        else:
            state = "Idle / Cooling"
            summary = f"OPTIMAL: System is currently in a low-power state. Neural forecast indicates minimal activity ({future_cpu:.1f}%)."

        intelligence_report.append(summary)

        diagnostic = f"DIAGNOSTIC: Current CPU is {cpu_val}% with {mem_val}% Memory saturation. "
        if top_proc_cpu > 10:
            diagnostic += f"The primary driver appears to be '{top_proc_name}' consuming {top_proc_cpu}% of available cycles. "
        
        if mem_val > 85:
            diagnostic += "CRITICAL: Memory pressure is extremely high, which may cause I/O wait times to climb regardless of CPU load."
        
        intelligence_report.append(diagnostic)

        if future_cpu > 75:
            advice = "ADVICE: Immediate intervention recommended. Please terminate high-impact background tasks. If this is a server, consider scaling or enabling aggressive throttling."
        elif mem_val > 80:
            advice = "ADVICE: Resource Leak Detected? Memory usage is disproportionate to CPU load. Consider restarting memory-intensive applications to reclaim heap space."
        else:
            advice = "ADVICE: Performance is healthy. This is an ideal window for running maintenance tasks or complex compilations."

        intelligence_report.append(advice)

        session["LAST_PREDICTION"] = {
            "time": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
            "future_cpu": round(future_cpu, 2),
            "state": state,
            "recommendations": intelligence_report
        }

        return jsonify({
            "user": request.user["email"],
            "future_cpu": round(future_cpu, 2),
            "state": state,
            "recommendations": intelligence_report
        })



    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route("/client-system", methods=["GET"])
@token_required
def client_system():
    user_email = request.user["email"]
    session = get_user_session(user_email)
    if session["LAST_SYSTEM_DATA"] is None or session["LAST_TELEMETRY_AT"] is None:
        return jsonify({"status": "warming_up", "telemetry_active": False})
    age_seconds = (
        datetime.datetime.now(datetime.timezone.utc) - session["LAST_TELEMETRY_AT"]
    ).total_seconds()
    telemetry_active = age_seconds <= 10
    return jsonify({
        "current": session["LAST_SYSTEM_DATA"],
        "history": session["SYSTEM_HISTORY"],
        "telemetry_active": telemetry_active,
        "last_received_at": session["LAST_TELEMETRY_AT"].isoformat(),
    })
@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "database_connected": users_collection is not None,
        "database_backend": database_backend,
        "database_error": database_error,
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH,
        "model_error": MODEL_LOAD_ERROR
    })
@app.route("/predicted", methods=["GET"])
@token_required
def predicted():
    user_email = request.user["email"]
    session = get_user_session(user_email)
    if session["LAST_PREDICTION"] is None:
        return jsonify({
            "status": "warming_up",
            "message": "Waiting for prediction"
        })
    return jsonify(session["LAST_PREDICTION"])
@app.route("/delete-account", methods=["DELETE"])
@token_required
def delete_account():
    if users_collection is None:
        return jsonify({"error": "Database not connected"}), 500
    user_email = request.user["email"]
    try:
        result = users_collection.delete_one({"email": user_email})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404
        if user_email in USER_SESSIONS:
            del USER_SESSIONS[user_email]
        return jsonify({"status": "deleted", "message": "Account successfully terminated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print("Starting AI Resource Optimization Backend...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_mode)

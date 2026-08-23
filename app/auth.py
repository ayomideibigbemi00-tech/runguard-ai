import os
import bcrypt
from fastapi import Request
from itsdangerous import URLSafeTimedSerializer, BadSignature
from app.db import get_db

# Read secret key from environment variable (set on Railway)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-key-12345")
serializer = URLSafeTimedSerializer(SECRET_KEY)

def hash_password(password: str) -> str:
    # bcrypt has a 72-byte limit, so we truncate safely
    return bcrypt.hashpw(password.encode('utf-8')[:72], bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8')[:72], hashed.encode('utf-8'))
    except ValueError:
        return False

def create_session_token(user_id: int) -> str:
    return serializer.dumps({"user_id": user_id})

def get_current_user(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        data = serializer.loads(token, max_age=86400)  # 1 day
        user_id = data.get("user_id")
    except BadSignature:
        return None
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None
from fastapi import Request, HTTPException, Depends
from itsdangerous import URLSafeTimedSerializer, BadSignature
from passlib.hash import bcrypt
from app.db import get_db

# Secret key for signing cookies (change this in production!)
SECRET_KEY = "change-this-to-a-random-secret-string"
serializer = URLSafeTimedSerializer(SECRET_KEY)

def hash_password(password: str) -> str:
    return bcrypt.hash(password)

def verify_password(password: str, hash: str) -> bool:
    return bcrypt.verify(password, hash)

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
# ─────────────────────────────────────────────────────────────────
#  auth.py  —  Traffic Controller authentication backend
#
#  Provides:
#    POST /api/auth/signup   — register a new controller account
#    POST /api/auth/login    — log in and create a session
#    POST /api/auth/logout   — clear the session
#    GET  /api/auth/me       — return current logged-in user info
#
#  Storage: SQLite database (same DB_PATH as traffic data)
#  Passwords: hashed with werkzeug (pbkdf2:sha256)
#  Sessions: Flask server-side sessions
# ─────────────────────────────────────────────────────────────────

import sqlite3
import os
from datetime import datetime
from functools import wraps

from flask import (Blueprint, request, jsonify,
                   session, redirect, url_for)
from werkzeug.security import generate_password_hash, check_password_hash

from config import DB_PATH

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

# ── DB helpers ────────────────────────────────────────────

def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_table():
    """Create the controllers table if it doesn't exist yet."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS controllers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname    TEXT    NOT NULL,
            lastname     TEXT    NOT NULL,
            badge_id     TEXT    NOT NULL UNIQUE,
            email        TEXT    NOT NULL UNIQUE,
            username     TEXT    NOT NULL UNIQUE,
            password_hash TEXT   NOT NULL,
            role         TEXT    NOT NULL DEFAULT 'controller',
            approved     INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT    NOT NULL
        )
    """)
    
    # Ensure default admin exists
    admin = conn.execute("SELECT id FROM controllers WHERE username = 'ram'").fetchone()
    if not admin:
        pw_hash = generate_password_hash("hazarika1?")
        conn.execute("""
            INSERT INTO controllers (firstname, lastname, badge_id, email, username, password_hash, role, approved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("Ram", "Admin", "ADMIN-001", "ram@admin.com", "ram", pw_hash, "admin", 1, datetime.utcnow().isoformat()))
        
    conn.commit()
    conn.close()
    print("[Auth] controllers table ready.")


# ── Login-required decorator ──────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


# ── Admin-required decorator ────────────────────────────

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return jsonify({"error": "Admin privileges required."}), 403
        return f(*args, **kwargs)
    return decorated


# ── Routes ────────────────────────────────────────────────

@auth_bp.route("/signup", methods=["POST"])
def signup():
    """Register a new traffic controller or admin account."""
    data = request.get_json(silent=True) or {}

    firstname = (data.get("firstname") or "").strip()
    lastname  = (data.get("lastname")  or "").strip()
    badge_id  = (data.get("badge_id")  or "").strip()
    email     = (data.get("email")     or "").strip().lower()
    username  = (data.get("username")  or "").strip().lower()
    password  = (data.get("password")  or "")

    # ── Server-side validation ────────────────────────────
    if not all([firstname, lastname, badge_id, email, username, password]):
        return jsonify({"success": False,
                        "error": "All fields are required."}), 400

    import re
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({"success": False,
                        "error": "Invalid email address."}), 400

    if len(username) < 3:
        return jsonify({"success": False,
                        "error": "Username must be at least 3 characters."}), 400

    pw_hash = generate_password_hash(password)

    role = "controller"
    approved = 0

    try:
        conn = _get_conn()

        conn.execute("""
            INSERT INTO controllers
              (firstname, lastname, badge_id, email, username, password_hash, role, approved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (firstname, lastname, badge_id, email, username,
              pw_hash, role, approved, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError as exc:
        msg = str(exc)
        if "username" in msg:
            field = "Username"
        elif "email" in msg:
            field = "Email address"
        elif "badge_id" in msg:
            field = "Badge ID"
        else:
            field = "A field"
        return jsonify({"success": False,
                        "error": f"{field} is already registered."}), 409

    return jsonify({
        "success": True,
        "message": "Account created. Awaiting administrator approval."
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """Log in with username (or email) + password."""
    data = request.get_json(silent=True) or {}

    identifier = (data.get("username") or "").strip().lower()
    password   = (data.get("password") or "")
    remember   = bool(data.get("remember", False))
    is_admin   = bool(data.get("is_admin", False))

    if not identifier or not password:
        return jsonify({"success": False,
                        "error": "Username and password are required."}), 400

    conn = _get_conn()
    user = conn.execute("""
        SELECT * FROM controllers
        WHERE username = ? OR email = ?
    """, (identifier, identifier)).fetchone()
    conn.close()

    if not user:
        return jsonify({"success": False,
                        "error": "Invalid credentials."}), 401

    if not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False,
                        "error": "Invalid credentials."}), 401

    # Role validation
    if is_admin and user["role"] != "admin":
        return jsonify({"success": False, "error": "This account is not an admin."}), 403
    if not is_admin and user["role"] == "admin":
        return jsonify({"success": False, "error": "Please check 'Login as Admin' to log in to your admin account."}), 403

    if not user["approved"]:
        return jsonify({"success": False,
                        "error": "Your account is pending administrator approval."}), 403

    # Create session
    session.permanent = remember
    session["user_id"]   = user["id"]
    session["username"]  = user["username"]
    session["role"]      = user["role"]
    session["firstname"] = user["firstname"]

    return jsonify({
        "success":  True,
        "redirect": "/admin" if user["role"] == "admin" else "/",
        "user": {
            "id":        user["id"],
            "username":  user["username"],
            "firstname": user["firstname"],
            "lastname":  user["lastname"],
            "role":      user["role"],
        }
    })

# ── User Management Endpoints (Admin only) ────────────────

@auth_bp.route("/users", methods=["GET"])
@login_required
@admin_required
def get_users():
    """Fetch all users."""
    conn = _get_conn()
    users = conn.execute("SELECT id, firstname, lastname, badge_id, email, username, role, approved, created_at FROM controllers ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])


@auth_bp.route("/users/<int:user_id>/approve", methods=["POST"])
@login_required
@admin_required
def approve_user(user_id):
    """Approve a user account."""
    conn = _get_conn()
    conn.execute("UPDATE controllers SET approved = 1 WHERE id = ? AND role != 'admin'", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@auth_bp.route("/users/<int:user_id>/revoke", methods=["POST"])
@login_required
@admin_required
def revoke_user(user_id):
    """Revoke a user account."""
    conn = _get_conn()
    conn.execute("UPDATE controllers SET approved = 0 WHERE id = ? AND role != 'admin'", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@auth_bp.route("/users/<int:user_id>", methods=["DELETE"])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user account."""
    conn = _get_conn()
    conn.execute("DELETE FROM controllers WHERE id = ? AND role != 'admin'", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """Clear the current session."""
    session.clear()
    return jsonify({"success": True, "redirect": "/login"})


@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    """Return current logged-in user info."""
    conn = _get_conn()
    user = conn.execute(
        "SELECT id, firstname, lastname, badge_id, email, username, role, created_at "
        "FROM controllers WHERE id = ?",
        (session["user_id"],)
    ).fetchone()
    conn.close()

    if not user:
        session.clear()
        return jsonify({"error": "User not found."}), 404

    return jsonify(dict(user))

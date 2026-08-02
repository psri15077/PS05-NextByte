"""
AI-Based Missing Person Identification and Reporting System — prototype backend.

Run with:  python3 app.py
Serves the frontend (static/) and a JSON API on http://localhost:5000

NOTE ON SCOPE: this is a self-contained prototype using SQLite + local file
storage so it runs anywhere with no cloud credentials. Swap the storage
layer for Firebase/Cloud Storage and the matcher for a trained embedding
model to move toward production, per the suggested tech stack.
"""

import os
import sqlite3
import uuid
import functools
from datetime import datetime

from flask import Flask, request, jsonify, session, send_from_directory, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import matching

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "instance", "mpis.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.secret_key = os.environ.get("MPIS_SECRET_KEY", "dev-secret-change-me")


# ---------------------------------------------------------------- DB setup

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('reporter','investigator','admin')),
            full_name TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number TEXT UNIQUE NOT NULL,
            case_type TEXT NOT NULL CHECK(case_type IN ('missing','found')),
            person_name TEXT,
            age INTEGER,
            gender TEXT,
            last_seen_location TEXT,
            last_seen_date TEXT,
            description TEXT,
            distinguishing_marks TEXT,
            photo_path TEXT,
            status TEXT NOT NULL DEFAULT 'Open',
            reporter_name TEXT,
            reporter_contact TEXT,
            created_by INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            candidate_case_id INTEGER NOT NULL,
            visual_score REAL,
            attribute_score REAL,
            composite_score REAL,
            confidence_note TEXT,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','confirmed','rejected')),
            reviewed_by INTEGER,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(id),
            FOREIGN KEY (candidate_case_id) REFERENCES cases(id),
            FOREIGN KEY (reviewed_by) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT NOT NULL,
            target TEXT,
            timestamp TEXT NOT NULL
        );
        """
    )
    db.commit()
    db.close()


def log_action(action, target=""):
    db = get_db()
    user = session.get("user")
    db.execute(
        "INSERT INTO audit_log (user_id, username, action, target, timestamp) VALUES (?,?,?,?,?)",
        (
            user["id"] if user else None,
            user["username"] if user else "anonymous",
            action,
            target,
            datetime.utcnow().isoformat(),
        ),
    )
    db.commit()


# ---------------------------------------------------------------- auth helpers

def login_required(roles=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            user = session.get("user")
            if not user:
                return jsonify({"error": "Authentication required"}), 401
            if roles and user["role"] not in roles:
                return jsonify({"error": "Insufficient permissions for this action"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def current_user():
    return session.get("user")


# ---------------------------------------------------------------- auth routes

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    role = data.get("role", "reporter")

    if role not in ("reporter",):
        # Investigator/admin accounts are provisioned by an admin, not self-service.
        return jsonify({"error": "Only reporter accounts can self-register"}), 403
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?,?,?,?,?)",
            (username, generate_password_hash(password), role, full_name, datetime.utcnow().isoformat()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already taken"}), 409

    return jsonify({"message": "Account created. Please log in."}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    user = {"id": row["id"], "username": row["username"], "role": row["role"], "full_name": row["full_name"]}
    session["user"] = user
    log_action("login")
    return jsonify({"user": user})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    log_action("logout")
    session.pop("user", None)
    return jsonify({"message": "Logged out"})


@app.route("/api/auth/me", methods=["GET"])
def me():
    return jsonify({"user": current_user()})


# ---------------------------------------------------------------- case routes

def row_to_case(row, include_contact=False):
    d = dict(row)
    if not include_contact:
        d.pop("reporter_contact", None)
    return d


@app.route("/api/cases", methods=["POST"])
def create_case():
    """
    Public case registration (the 'Case Registration Portal'). Anyone can
    file a report; if logged in as a reporter, the case is linked to their
    account so they can track status. Photo upload is optional but required
    for AI matching to run on this case.
    """
    form = request.form
    case_type = form.get("case_type", "missing")
    if case_type not in ("missing", "found"):
        return jsonify({"error": "case_type must be 'missing' or 'found'"}), 400

    required = ["person_name", "age", "gender", "last_seen_location", "reporter_name", "reporter_contact"]
    missing_fields = [f for f in required if not form.get(f)]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    photo_path = None
    file = request.files.get("photo")
    if file and file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({"error": "Photo must be a PNG or JPG image"}), 400
        filename = f"{uuid.uuid4().hex}.{ext}"
        full_path = os.path.join(UPLOAD_DIR, filename)
        file.save(full_path)
        photo_path = filename

    case_number = f"MPIS-{datetime.utcnow().strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}"
    user = current_user()

    db = get_db()
    cur = db.execute(
        """INSERT INTO cases
           (case_number, case_type, person_name, age, gender, last_seen_location,
            last_seen_date, description, distinguishing_marks, photo_path, status,
            reporter_name, reporter_contact, created_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            case_number, case_type, form.get("person_name"), form.get("age"), form.get("gender"),
            form.get("last_seen_location"), form.get("last_seen_date"), form.get("description"),
            form.get("distinguishing_marks"), photo_path, "Open",
            form.get("reporter_name"), form.get("reporter_contact"),
            user["id"] if user else None, datetime.utcnow().isoformat(),
        ),
    )
    db.commit()
    log_action("case_created", case_number)

    return jsonify({"message": "Case registered", "case_number": case_number, "id": cur.lastrowid}), 201


@app.route("/api/cases", methods=["GET"])
@login_required(roles=("investigator", "admin"))
def list_cases():
    """Dashboard listing with filters — investigator/admin only (protects reporter PII)."""
    db = get_db()
    query = "SELECT id, case_number, case_type, person_name, age, gender, last_seen_location, last_seen_date, status, photo_path, created_at FROM cases WHERE 1=1"
    params = []

    if status := request.args.get("status"):
        query += " AND status = ?"
        params.append(status)
    if case_type := request.args.get("case_type"):
        query += " AND case_type = ?"
        params.append(case_type)
    if location := request.args.get("location"):
        query += " AND last_seen_location LIKE ?"
        params.append(f"%{location}%")
    if age_min := request.args.get("age_min"):
        query += " AND age >= ?"
        params.append(age_min)
    if age_max := request.args.get("age_max"):
        query += " AND age <= ?"
        params.append(age_max)
    if q := request.args.get("q"):
        query += " AND (person_name LIKE ? OR case_number LIKE ?)"
        params.extend([f"%{q}%", f"%{q}%"])

    query += " ORDER BY created_at DESC"
    rows = db.execute(query, params).fetchall()
    log_action("case_list_viewed")
    return jsonify({"cases": [dict(r) for r in rows]})


@app.route("/api/cases/<int:case_id>", methods=["GET"])
def get_case(case_id):
    db = get_db()
    row = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not row:
        return jsonify({"error": "Case not found"}), 404

    user = current_user()
    is_staff = user and user["role"] in ("investigator", "admin")
    is_owner = user and row["created_by"] == user["id"]

    if not (is_staff or is_owner):
        # Public / reporter view: status tracking only, no PII about the case contact.
        return jsonify({
            "case": {
                "case_number": row["case_number"],
                "person_name": row["person_name"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
        })

    log_action("case_viewed", row["case_number"])
    return jsonify({"case": row_to_case(row, include_contact=True)})


@app.route("/api/cases/lookup/<case_number>", methods=["GET"])
def lookup_case_by_number(case_number):
    """Public status tracking by case number — no PII exposed, no login required."""
    db = get_db()
    row = db.execute("SELECT case_number, person_name, status, created_at FROM cases WHERE case_number = ?", (case_number,)).fetchone()
    if not row:
        return jsonify({"error": "No case found with that case number"}), 404
    log_action("public_status_check", case_number)
    return jsonify({"case": dict(row)})


@app.route("/api/cases/<int:case_id>/status", methods=["PATCH"])

@login_required(roles=("investigator", "admin"))
def update_case_status(case_id):
    data = request.get_json(force=True)
    new_status = data.get("status")
    valid = {"Open", "Under Investigation", "Potential Match Found", "Verified", "Closed", "Reunited"}
    if new_status not in valid:
        return jsonify({"error": f"status must be one of {sorted(valid)}"}), 400

    db = get_db()
    row = db.execute("SELECT case_number FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not row:
        return jsonify({"error": "Case not found"}), 404

    db.execute("UPDATE cases SET status = ? WHERE id = ?", (new_status, case_id))
    db.commit()
    log_action(f"status_changed:{new_status}", row["case_number"])
    return jsonify({"message": "Status updated"})


@app.route("/uploads/<path:filename>")
@login_required(roles=("investigator", "admin"))
def serve_upload(filename):
    # Photos are only ever served to authenticated staff, never publicly.
    return send_from_directory(UPLOAD_DIR, filename)


# ---------------------------------------------------------------- matching routes

@app.route("/api/cases/<int:case_id>/match", methods=["POST"])
@login_required(roles=("investigator", "admin"))
def run_match(case_id):
    """
    AI-assisted comparison: compares this case's photo against every case
    of the OPPOSITE type (missing <-> found) and stores candidate matches
    as 'pending' — never auto-confirmed. Investigator reviews each one.
    """
    db = get_db()
    case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not case:
        return jsonify({"error": "Case not found"}), 404
    if not case["photo_path"]:
        return jsonify({"error": "This case has no photo on file — cannot run visual matching"}), 400

    opposite_type = "found" if case["case_type"] == "missing" else "missing"
    candidates = db.execute(
        "SELECT * FROM cases WHERE case_type = ? AND id != ? AND photo_path IS NOT NULL",
        (opposite_type, case_id),
    ).fetchall()

    case_photo = os.path.join(UPLOAD_DIR, case["photo_path"])
    results = []

    for cand in candidates:
        cand_photo = os.path.join(UPLOAD_DIR, cand["photo_path"])
        try:
            visual = matching.image_visual_similarity(case_photo, cand_photo)
        except Exception as e:
            continue
        attr_score = matching.attribute_similarity(dict(case), dict(cand))
        composite = matching.composite_match_score(visual, attr_score)

        # Only surface plausible candidates to avoid flooding investigators with noise
        if composite < 15:
            continue

        db.execute(
            """INSERT INTO matches
               (case_id, candidate_case_id, visual_score, attribute_score, composite_score,
                confidence_note, status, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                case_id, cand["id"], visual["visual_score"], attr_score, composite,
                visual["confidence_note"], "pending", datetime.utcnow().isoformat(),
            ),
        )
        results.append({
            "candidate_case_id": cand["id"],
            "candidate_case_number": cand["case_number"],
            "candidate_person_name": cand["person_name"],
            "visual_score": visual["visual_score"],
            "attribute_score": attr_score,
            "composite_score": composite,
            "confidence_note": visual["confidence_note"],
        })

    db.commit()
    if case["status"] == "Open" and results:
        db.execute("UPDATE cases SET status = 'Potential Match Found' WHERE id = ?", (case_id,))
        db.commit()

    log_action("match_run", case["case_number"])
    results.sort(key=lambda r: r["composite_score"], reverse=True)
    return jsonify({"matches_found": len(results), "results": results})


@app.route("/api/matches", methods=["GET"])
@login_required(roles=("investigator", "admin"))
def list_matches():
    db = get_db()
    status = request.args.get("status", "pending")
    rows = db.execute(
        """SELECT m.*, 
                  c1.case_number AS case_number, c1.person_name AS case_person,
                  c2.case_number AS candidate_case_number, c2.person_name AS candidate_person
           FROM matches m
           JOIN cases c1 ON m.case_id = c1.id
           JOIN cases c2 ON m.candidate_case_id = c2.id
           WHERE m.status = ?
           ORDER BY m.composite_score DESC""",
        (status,),
    ).fetchall()
    return jsonify({"matches": [dict(r) for r in rows]})


@app.route("/api/matches/<int:match_id>/review", methods=["POST"])
@login_required(roles=("investigator", "admin"))
def review_match(match_id):
    """
    Human-in-the-loop enforcement point: an authorized investigator must
    explicitly confirm or reject every AI-suggested match. This is the
    only path by which a match can change a case's real-world status.
    """
    data = request.get_json(force=True)
    action = data.get("action")
    if action not in ("confirm", "reject"):
        return jsonify({"error": "action must be 'confirm' or 'reject'"}), 400

    db = get_db()
    match = db.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
    if not match:
        return jsonify({"error": "Match not found"}), 404

    new_status = "confirmed" if action == "confirm" else "rejected"
    user = current_user()
    db.execute(
        "UPDATE matches SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
        (new_status, user["id"], datetime.utcnow().isoformat(), match_id),
    )

    if action == "confirm":
        db.execute("UPDATE cases SET status = 'Verified' WHERE id IN (?, ?)",
                   (match["case_id"], match["candidate_case_id"]))

    db.commit()
    log_action(f"match_{new_status}", f"match#{match_id}")
    return jsonify({"message": f"Match {new_status}"})


# ---------------------------------------------------------------- admin routes

@app.route("/api/admin/users", methods=["GET"])
@login_required(roles=("admin",))
def list_users():
    db = get_db()
    rows = db.execute("SELECT id, username, role, full_name, created_at FROM users ORDER BY created_at DESC").fetchall()
    return jsonify({"users": [dict(r) for r in rows]})


@app.route("/api/admin/users", methods=["POST"])
@login_required(roles=("admin",))
def create_staff_user():
    """Admins provision investigator/admin accounts — no self-service for these roles."""
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    role = data.get("role")
    full_name = (data.get("full_name") or "").strip()

    if role not in ("investigator", "admin"):
        return jsonify({"error": "role must be 'investigator' or 'admin'"}), 400
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?,?,?,?,?)",
            (username, generate_password_hash(password), role, full_name, datetime.utcnow().isoformat()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already taken"}), 409

    log_action("staff_user_created", username)
    return jsonify({"message": "Staff account created"}), 201


@app.route("/api/admin/audit-log", methods=["GET"])
@login_required(roles=("admin",))
def audit_log():
    db = get_db()
    rows = db.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 200").fetchall()
    return jsonify({"log": [dict(r) for r in rows]})


# ---------------------------------------------------------------- static frontend

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


if __name__ == "__main__":
    init_db()
    print(f"DB ready at {DB_PATH}")
    debug_mode = os.environ.get("MPIS_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False)

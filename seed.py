"""
seed.py — creates demo accounts and a few sample cases so judges/reviewers
can explore the system immediately without registering anything by hand.

Run with: python3 seed.py   (after app.py has been run once, or it will
init the DB itself)
"""

import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
import cv2
from werkzeug.security import generate_password_hash

import app as app_module  # reuses init_db(), paths

DB_PATH = app_module.DB_PATH
UPLOAD_DIR = app_module.UPLOAD_DIR


def make_placeholder_face(seed_val, filename):
    """
    Generates a simple synthetic 'face-like' grayscale image (oval + eyes +
    mouth) so the matching pipeline has something real to open, detect on,
    and compare. Two images built from the same seed are near-duplicates
    (simulating the same person photographed twice); different seeds are
    visually distinct. This is placeholder demo data only — replace with
    real (consented) case photos in actual use.
    """
    rng = np.random.RandomState(seed_val)
    img = np.full((300, 300, 3), 235, dtype=np.uint8)
    cx, cy = 150, 150
    skin = (int(180 + rng.randint(-10, 10)), int(200 + rng.randint(-10, 10)), int(220 + rng.randint(-10, 10)))
    cv2.ellipse(img, (cx, cy), (80, 100), 0, 0, 360, skin, -1)
    eye_off = 30 + rng.randint(-3, 3)
    cv2.circle(img, (cx - eye_off, cy - 20), 10, (60, 60, 60), -1)
    cv2.circle(img, (cx + eye_off, cy - 20), 10, (60, 60, 60), -1)
    cv2.ellipse(img, (cx, cy + 40), (25, 10), 0, 0, 180, (90, 60, 90), 3)
    noise = rng.randint(0, 8, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, img)
    return filename


def seed():
    app_module.init_db()
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    users = [
        ("admin", "admin123", "admin", "System Administrator"),
        ("investigator1", "invest123", "investigator", "Insp. R. Sharma"),
        ("reporter1", "report123", "reporter", "Anita Verma"),
    ]
    for username, password, role, full_name in users:
        existing = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if not existing:
            db.execute(
                "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?,?,?,?,?)",
                (username, generate_password_hash(password), role, full_name, datetime.utcnow().isoformat()),
            )
    db.commit()
    reporter_id = db.execute("SELECT id FROM users WHERE username='reporter1'").fetchone()["id"]

    existing_cases = db.execute("SELECT COUNT(*) c FROM cases").fetchone()["c"]
    if existing_cases > 0:
        print("Cases already seeded, skipping.")
        db.close()
        return

    # Seed value 42 used twice -> simulates the SAME person appearing in
    # both a 'missing' report and a 'found' report (a true match to review).
    missing_photo = make_placeholder_face(42, f"{uuid.uuid4().hex}.jpg")
    found_photo = make_placeholder_face(42, f"{uuid.uuid4().hex}.jpg")
    decoy_photo_1 = make_placeholder_face(7, f"{uuid.uuid4().hex}.jpg")
    decoy_photo_2 = make_placeholder_face(99, f"{uuid.uuid4().hex}.jpg")

    now = datetime.utcnow().isoformat()
    cases = [
        ("MPIS-202601-A1B2C3", "missing", "Ravi Kumar", 34, "Male", "Anna Nagar Chennai",
         "2026-07-15", "Last seen wearing a blue shirt near the bus stand.", "Scar on left eyebrow",
         missing_photo, "Open", "Anita Verma", "9876500001", reporter_id, now),
        ("MPIS-202601-D4E5F6", "found", "Unidentified Male", 35, "Male", "Anna Nagar West Chennai",
         "2026-07-20", "Found disoriented, unable to state name.", "Scar on left eyebrow",
         found_photo, "Open", "Govt. Shelter Home", "9876500002", None, now),
        ("MPIS-202601-G7H8I9", "missing", "Priya Singh", 22, "Female", "Sector 21 Noida",
         "2026-06-01", "Left home after an argument, has not returned.", "Mole on right cheek",
         decoy_photo_1, "Open", "Sunil Singh", "9876500003", None, now),
        ("MPIS-202601-J1K2L3", "found", "Unidentified Female", 40, "Female", "Karol Bagh Delhi",
         "2026-07-25", "Found near railway station.", "None noted",
         decoy_photo_2, "Open", "Railway Police", "9876500004", None, now),
    ]
    db.executemany(
        """INSERT INTO cases
           (case_number, case_type, person_name, age, gender, last_seen_location,
            last_seen_date, description, distinguishing_marks, photo_path, status,
            reporter_name, reporter_contact, created_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        cases,
    )
    db.commit()
    db.close()
    print("Seed complete.")
    print("Demo logins:")
    print("  admin / admin123        (role: admin)")
    print("  investigator1 / invest123  (role: investigator)")
    print("  reporter1 / report123    (role: reporter)")
    print("Try: log in as investigator1 -> open case MPIS-202601-A1B2C3 -> Run AI Match")
    print("     (it should surface MPIS-202601-D4E5F6 as a strong candidate)")


if __name__ == "__main__":
    seed()

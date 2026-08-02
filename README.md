# MPIS — AI-Based Missing Person Identification & Reporting System (Prototype)

A working prototype for the NCRB problem statement: case registration, an
investigator dashboard, AI-assisted image comparison, and case status
tracking — with role-based access control and mandatory human review before
any match is treated as confirmed.

This is a **self-contained local prototype**: Flask + SQLite + local file
storage, so it runs anywhere with just Python 3 and no cloud credentials or
internet access. The suggested tech stack (Firebase/Cloud Storage,
TensorFlow) is a straightforward upgrade path — see "Moving to production"
below.

## Quick start

```bash
cd backend
pip install flask opencv-python-headless numpy pillow werkzeug  # if not already installed
python3 seed.py     # creates demo accounts + 4 sample cases (one true match pair)
python3 app.py       # starts the server on http://localhost:5000
```

Open `http://localhost:5000` in a browser.

### Demo accounts (created by `seed.py`)

| Username        | Password    | Role         |
|------------------|-------------|--------------|
| admin            | admin123    | admin        |
| investigator1    | invest123   | investigator |
| reporter1        | report123   | reporter     |

### Suggested demo flow
1. Sign in as `investigator1`.
2. Open the **Case dashboard**, click case `MPIS-202601-A1B2C3` (Ravi Kumar, missing).
3. Click **Run AI match against opposite-type cases**.
4. It should surface `MPIS-202601-D4E5F6` (an unidentified found person) as a
   ~90%+ candidate — the two seeded cases are built from the same synthetic
   photo to simulate a true match, while the other two seeded cases are
   visually distinct decoys.
5. Go to **Match review queue** and **Confirm** or **Reject** the candidate.
   Notice a match can *only* change case status once a human reviewer acts —
   the AI step alone never closes or verifies a case.
6. Sign in as `admin` to see the **audit log** (every login, case view, and
   match decision is logged) and to provision new investigator accounts.
7. From the auth screen's **Track a case** tab, paste any case number to see
   the no-login public status check (deliberately exposes status only, no
   PII).

## What's implemented against the brief

- **Case registration portal** — public form, no login required, optional
  photo upload, auto-generated case number, status trackable by that number.
- **Investigator dashboard** — searchable/filterable case list (status,
  type, location, free text), case detail view, status lifecycle (Open →
  Under Investigation → Potential Match Found → Verified → Closed/Reunited).
- **AI-assisted image comparison** — see "How matching works" below.
  Produces a transparent, explainable score, never an automatic
  identification.
- **Case status tracking** — status changes are investigator/admin-only and
  logged; reporters and the public get a read-only status view.
- **Role-based access control** — `reporter` (file + track only),
  `investigator` (dashboard, matching, review), `admin` (+ user
  provisioning, audit log). Reporter contact details and case photos are
  never exposed to unauthenticated requests.
- **Human-in-the-loop enforcement** — `POST /api/cases/<id>/match` only ever
  creates `pending` match records. The *only* endpoint that can mark a
  match `confirmed` is `POST /api/matches/<id>/review`, which requires an
  authenticated investigator/admin session. This is the constraint the brief
  explicitly calls out, so it's enforced at the API layer, not just the UI.
- **Audit log** — every login, case view, status change, match run, and
  match decision is recorded with who/when for accountability.

## How matching works (and its honest limits)

`backend/matching.py` implements a classic computer-vision pipeline that
runs entirely offline:

1. **Face detection** — OpenCV Haar cascade locates and crops the face in
   each photo (falls back to whole-image comparison if no face is found,
   and flags this clearly in the result).
2. **Feature extraction** — ORB keypoint descriptors on the cropped face.
3. **Visual similarity** — ratio-test-filtered keypoint matching between two
   photos, giving a 0–100 score.
4. **Attribute similarity** — age proximity, gender match, and last-seen
   location overlap, also 0–100.
5. **Composite score** — a weighted blend (70% visual / 30% attribute) shown
   to the investigator, along with *both* sub-scores and a confidence note,
   so nothing is a black box.

This is a legitimate working baseline, not a trained deep-learning face
embedding model (no FaceNet/ArcFace/dlib — those weren't available without
internet access in this environment). It correctly distinguishes a true
match from decoys in the seed data. For production-grade accuracy, swap
`extract_descriptors()` for a pretrained face-embedding model (FaceNet,
ArcFace, or a fine-tuned model via TensorFlow, as the brief suggests) and
compare embeddings with cosine similarity — the rest of the pipeline
(thresholding, review queue, human-in-the-loop enforcement) doesn't need to
change.

**Matching only ever produces suggestions.** Every candidate lands in a
`pending` queue; only an authenticated investigator's explicit
confirm/reject action changes anything, per the brief's constraint.

## Data protection choices

- Passwords hashed with Werkzeug's `generate_password_hash` (PBKDF2).
- Investigator/admin accounts can't self-register — only an admin can
  provision them (`/api/admin/users`), preventing privilege escalation via
  the public form.
- Case photos are served only via an authenticated route
  (`/uploads/<file>`, investigator/admin only) — never linked publicly.
- Reporter contact info is stripped from every response except to
  investigators/admins and the case's own creator.
- The public "track a case" endpoint returns status only — name and status,
  nothing else.
- All access to case data and all match decisions are written to
  `audit_log` with user, action, target, and timestamp.

## Moving to production

- Swap SQLite for a managed DB (Firestore, as suggested) and local
  `uploads/` for Cloud Storage — the storage calls are isolated in a handful
  of functions in `app.py`, so this is a contained change.
- Replace ORB/Haar with a trained face-embedding model (TensorFlow), and
  store embeddings instead of recomputing them per match run for scale.
- Add HTTPS, real session/JWT expiry, rate limiting on `/api/cases` and
  `/api/*/match` (to prevent bulk-query misuse), and 2FA for investigator/
  admin accounts.
- Add consent/legal-basis capture on the registration form and a data
  retention/deletion job for closed cases, per data-protection requirements
  for this category of data.

## Project structure

```
mpis/
├── backend/
│   ├── app.py         # Flask app: auth, cases, matching, admin, audit routes
│   ├── matching.py     # OpenCV face detection + ORB comparison engine
│   └── seed.py         # Demo accounts + sample cases (run once)
├── static/
│   ├── index.html      # SPA shell + view templates
│   ├── styles.css       # Design system
│   └── app.js           # Frontend logic (auth, routing, API calls)
├── uploads/             # Case photos (created at runtime)
└── instance/mpis.db     # SQLite database (created at runtime)
```

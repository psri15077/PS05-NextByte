"""
matching.py — Lightweight, dependency-free image comparison engine.

IMPORTANT (read this before treating scores as "AI facial recognition"):
This uses classic computer-vision techniques available offline (Haar cascade
face detection + ORB keypoint descriptors), NOT a trained deep face-embedding
model (e.g. FaceNet/ArcFace). It is a working, honest baseline that plugs
into the same pipeline a production embedding model would use — swap
`extract_descriptors()` for a real embedding model later without touching
the rest of the app. Every match is a SUGGESTION for human review, never
an automatic identification.
"""

import cv2
import numpy as np
import os

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)

_orb = cv2.ORB_create(nfeatures=500)
_bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def detect_and_crop_face(image_path, pad_ratio=0.25):
    """
    Returns (face_gray_image, face_found: bool).
    If no face is detected, returns the whole image (grayscale) so the
    pipeline still degrades gracefully, flagged via face_found=False.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return cv2.resize(gray, (256, 256)), False

    # Use the largest detected face (most likely the subject)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)

    face = gray[y0:y1, x0:x1]
    face = cv2.resize(face, (256, 256))
    return face, True


def extract_descriptors(face_gray):
    """ORB keypoint descriptors for a face/image region."""
    keypoints, descriptors = _orb.detectAndCompute(face_gray, None)
    return descriptors


def compare_descriptors(desc1, desc2):
    """
    Returns a 0-100 visual similarity score based on ratio-test-filtered
    ORB keypoint matches between two descriptor sets.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return 0.0

    matches = _bf_matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    denom = min(len(desc1), len(desc2))
    if denom == 0:
        return 0.0
    raw_score = len(good) / denom
    return round(min(raw_score, 1.0) * 100, 2)


def image_visual_similarity(image_path_a, image_path_b):
    """
    Full pipeline: detect faces in both images, extract descriptors,
    compare. Returns dict with score and metadata for transparency
    in the investigator UI (never hide how a score was produced).
    """
    face_a, found_a = detect_and_crop_face(image_path_a)
    face_b, found_b = detect_and_crop_face(image_path_b)

    desc_a = extract_descriptors(face_a)
    desc_b = extract_descriptors(face_b)

    score = compare_descriptors(desc_a, desc_b)

    return {
        "visual_score": score,
        "face_detected_a": found_a,
        "face_detected_b": found_b,
        "confidence_note": (
            "Face detected in both images."
            if found_a and found_b
            else "Face not clearly detected in one or both images — "
                 "score is less reliable, manual review strongly advised."
        ),
    }


def attribute_similarity(case_a, case_b):
    """
    Compares non-visual case attributes to support/contextualize the
    visual score. Returns 0-100.
    """
    score = 0.0
    weight_total = 0.0

    # Age proximity (within 3 years = full credit, decaying to 0 by 10 years)
    if case_a.get("age") is not None and case_b.get("age") is not None:
        diff = abs(int(case_a["age"]) - int(case_b["age"]))
        age_score = max(0.0, 1 - diff / 10)
        score += age_score * 30
        weight_total += 30

    # Gender match
    if case_a.get("gender") and case_b.get("gender"):
        score += (30 if case_a["gender"] == case_b["gender"] else 0)
        weight_total += 30

    # Location overlap (simple substring/word overlap on last-seen location)
    loc_a = (case_a.get("last_seen_location") or "").lower()
    loc_b = (case_b.get("last_seen_location") or "").lower()
    if loc_a and loc_b:
        words_a, words_b = set(loc_a.split()), set(loc_b.split())
        overlap = len(words_a & words_b) / max(1, len(words_a | words_b))
        score += overlap * 40
        weight_total += 40

    if weight_total == 0:
        return 0.0
    return round((score / weight_total) * 100, 2)


def composite_match_score(visual_result, attr_score, visual_weight=0.7):
    """
    Blends visual + attribute similarity into one headline number shown
    to investigators. Weighting favors visual evidence but attributes
    help when photo quality is poor.
    """
    v = visual_result["visual_score"]
    a = attr_score
    composite = visual_weight * v + (1 - visual_weight) * a
    return round(composite, 2)

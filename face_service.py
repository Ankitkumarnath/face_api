import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from settings import (
    DATA_DIR, PEOPLE_DIR, INDEX_PATH, IDS_PATH,
    COSINE_SIM_THRESHOLD, INSIGHTFACE_PROVIDER, INSIGHTFACE_DET_SIZE,
    MAX_FACES_PER_IMAGE
)

# Lazy import to speed cold start
_insight_app = None

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)

def load_insightface():
    global _insight_app
    if _insight_app is not None:
        return _insight_app
    from insightface.app import FaceAnalysis
    # Pass providers to the constructor, not to .prepare()
    app = FaceAnalysis(name="buffalo_l", providers=INSIGHTFACE_PROVIDER)
    app.prepare(ctx_id=0, det_size=INSIGHTFACE_DET_SIZE)
    _insight_app = app
    return _insight_app


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt(np.maximum(np.sum(x * x, axis=axis, keepdims=True), eps))
    return x / norm

class FaceIndex:
    """
    Holds embeddings and IDs; uses sklearn NearestNeighbors with cosine metric.
    """
    def __init__(self):
        self.embeddings = None  # (N, 512)
        self.ids: List[str] = []
        self.nn: Optional[NearestNeighbors] = None
        self.load()

    def load(self):
        _ensure_dirs()
        if INDEX_PATH.exists():
            self.embeddings = np.load(INDEX_PATH)
        else:
            self.embeddings = np.empty((0, 512), dtype=np.float32)

        if IDS_PATH.exists():
            self.ids = json.loads(Path(IDS_PATH).read_text())
        else:
            self.ids = []

        self._rebuild()

    def _rebuild(self):
        if len(self.ids) == 0:
            self.nn = None
        else:
            self.nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            self.nn.fit(self.embeddings)

    def save(self):
        np.save(INDEX_PATH, self.embeddings)
        Path(IDS_PATH).write_text(json.dumps(self.ids, ensure_ascii=False, indent=2))

    def add(self, embedding: np.ndarray, person_id: str):
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        self.embeddings = np.vstack([self.embeddings, embedding.astype(np.float32)])
        self.ids.append(person_id)
        self._rebuild()
        self.save()

    def search(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Returns (match_id, cosine_similarity). If no index, returns (None, 0).
        """
        if self.nn is None or len(self.ids) == 0:
            return None, 0.0

        if embedding.ndim == 1:
            embedding = embedding[None, :]
        # sklearn cosine metric returns cosine distance = 1 - cosine_similarity
        dist, idx = self.nn.kneighbors(embedding, n_neighbors=1, return_distance=True)
        dist = float(dist[0][0])
        nn_index = int(idx[0][0])
        cosine_sim = 1.0 - dist
        match_id = self.ids[nn_index] if cosine_sim >= COSINE_SIM_THRESHOLD else None
        return match_id, cosine_sim

face_index = FaceIndex()

def detect_and_embed(pil_image: Image.Image):
    """
    Returns list of dicts: {bbox, kps, embedding (512,), det_score}
    """
    app = load_insightface()
    bgr = pil_to_bgr(pil_image)
    faces = app.get(bgr)[:MAX_FACES_PER_IMAGE]
    results = []
    for f in faces:
        emb = f.normed_embedding  # already L2-normalized (float32, 512)
        if emb is None or emb.shape[0] != 512:
            # Some versions require manual get embedding. Fallback:
            emb = app.models['recognition'].get(bgr, [f])[0]
            emb = emb.astype(np.float32)
            emb = emb / max(np.linalg.norm(emb), 1e-12)
        results.append({
            "bbox": [int(x) for x in f.bbox.astype(int)],
            "kps": f.kps.tolist(),
            "embedding": emb.astype(np.float32),
            "det_score": float(getattr(f, "det_score", 0.0)),
        })
    return results

def crop_face(bgr: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return bgr[y1:y2, x1:x2]

def register_face(pil_image: Image.Image, note: Optional[str] = None) -> dict:
    app = load_insightface()
    detections = detect_and_embed(pil_image)
    if not detections:
        return {"status": "no_face", "message": "No face detected."}

    # pick the most confident detection
    det = max(detections, key=lambda d: d["det_score"])
    emb = det["embedding"]
    # Create person id
    person_id = f"p_{int(time.time())}"
    person_dir = PEOPLE_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    # Save crop + meta
    bgr = pil_to_bgr(pil_image)
    crop = crop_face(bgr, det["bbox"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = person_dir / f"face_{ts}.jpg"
    cv2.imwrite(str(img_path), crop)

    meta = {
        "person_id": person_id,
        "registered_at": ts,
        "note": note or "",
        "bbox": det["bbox"],
        "det_score": det["det_score"],
    }
    (person_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Add to index
    face_index.add(emb, person_id)

    return {"status": "registered", "person_id": person_id, "meta": meta}

def recognize_faces(pil_image: Image.Image):
    detections = detect_and_embed(pil_image)
    results = []
    for det in detections:
        emb = det["embedding"]
        match_id, sim = face_index.search(emb)
        status = "known" if match_id else "new"
        results.append({
            "status": status,
            "match_id": match_id,
            "similarity": sim,
            "bbox": det["bbox"],
            "det_score": det["det_score"],
        })
    return results

def auto_register_if_new(pil_image: Image.Image, auto_register: bool = True):
    results = recognize_faces(pil_image)
    # If any NEW face and auto_register enabled, register the best one
    new_faces = [r for r in results if r["status"] == "new"]
    registered = None
    if auto_register and new_faces:
        # Choose the highest det_score among new faces
        best_new = max(new_faces, key=lambda r: r["det_score"])
        registered = register_face(pil_image)
    return {"matches": results, "registered": registered}

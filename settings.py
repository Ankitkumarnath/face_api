from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
PEOPLE_DIR = DATA_DIR / "people"
INDEX_PATH = DATA_DIR / "index.npy"
IDS_PATH = DATA_DIR / "ids.json"

# Matching
# We use cosine similarity on L2-normalized ArcFace embeddings.
# Typical good range: 0.35–0.45 (higher = stricter). Start moderate:
COSINE_SIM_THRESHOLD = 0.40

# InsightFace options
INSIGHTFACE_PROVIDER = ["CPUExecutionProvider"]  # M1 CPU works great
INSIGHTFACE_DET_SIZE = (640, 640)               # detector input size
MAX_FACES_PER_IMAGE = 5                          # safety cap

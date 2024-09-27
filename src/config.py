import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN", "df9039b5ed73edb8f4fb0c23a7c1e2d2")
MODEL = "GDino1_5_Pro"
DETECTION_TARGETS = ["Mask", "BBox"]
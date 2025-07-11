import os
from dotenv import load_dotenv

load_dotenv()

# Must be set in the environment
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_HOME = os.environ.get("HF_HOME", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# Database configuration, defaults to SQLite in memory
DB_DRIVER = os.environ.get("DB_DRIVER", "sqlite")
DB_NAME = os.environ.get("DB_NAME", "/:memory:")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "admin")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

# Safe to leave the default values
DOCUMENTS_DIR =  os.environ.get("DOCUMENTS_DIR", "_documents")
STORAGE = os.environ.get("STORAGE", "_storage")
CACHE_NAME = os.environ.get("CACHE_NAME", "cag.cache")
EMBED_MODEL_DIR = os.environ.get("EMBED_MODEL_DIR", "embed-models")

# Obtained from the previous variables
CACHE_PATH = os.path.join(STORAGE, CACHE_NAME)
VECTOR_STORE_DIR = os.path.join(STORAGE, "vector_store")
DB_URL = f"{DB_DRIVER}://{DB_NAME}" if DB_DRIVER == "sqlite" else f"{DB_DRIVER}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
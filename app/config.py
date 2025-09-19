import os
import pathlib
from dotenv import load_dotenv

# Find the project's root directory
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(BASE_DIR / '.env')

class Config:
    """Contains configuration settings for the application."""
    # API Keys
    OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

    # Model Settings
    LLM_CHAT_MODEL = "openai/gpt-oss-20b:free"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # Database Settings
    CHROMA_DB_PATH = BASE_DIR / "chroma_db"
    ONET_COLLECTION_NAME = "onet_data"

    # Data File Paths
    DATA_DIR = BASE_DIR / "data"
    ABILITIES_FILE_PATH = DATA_DIR / "abilities.xlsx"
    INTERESTS_FILE_PATH = DATA_DIR / "interests.xlsx"
    KNOWLEDGE_FILE_PATH = DATA_DIR / "knowledge.xlsx"
    OCCUPATIONS_FILE_PATH = DATA_DIR / "occupations.xlsx"
    SKILLS_FILE_PATH = DATA_DIR / "skills.xlsx"

    
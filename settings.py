import sys
import os
from pathlib import Path

OPENAI_API_KEY = "xxxxxx"
KEY_OWNER = "Adam Sun"
DEBUG = True
MAX_CHUNK_SIZE = 4
LLM_VERS = "gpt-4o-mini"
BASE_DIR = f"{Path(__file__).resolve().parent.parent}"
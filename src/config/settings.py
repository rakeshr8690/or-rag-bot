"""
Configuration settings for the OR RAG Bot.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", 32768))

VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb")
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", BASE_DIR / "vector_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "or_problems")

DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", BASE_DIR / "data_raw"))
DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", BASE_DIR / "data_processed"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

PROBLEM_TEMPLATE = """
===========================================
PROBLEM ID: {problem_id}
PROBLEM TYPE: {problem_type}
INDUSTRY: {industry}
DIFFICULTY: {difficulty}
===========================================

PROBLEM TITLE:
{title}

BUSINESS CONTEXT:
{context}

DECISION VARIABLES:
{variables}

OBJECTIVE:
{objective}

CONSTRAINTS:
{constraints}

NUMERICAL DATA:
{numerical_data}

MATHEMATICAL FORMULATION:
{formulation}

SOLUTION APPROACH:
{solution_approach}

OPTIMAL SOLUTION:
{optimal_solution}

SENSITIVITY ANALYSIS NOTES:
{sensitivity_notes}

RELATED PROBLEMS:
{related_problems}

KEYWORDS:
{keywords}
===========================================
"""

DATASET_URLS = {
    "nl4opt": "CardinalOperations/NL4OPT",
    "industry_or": "CardinalOperations/IndustryOR",
}

DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
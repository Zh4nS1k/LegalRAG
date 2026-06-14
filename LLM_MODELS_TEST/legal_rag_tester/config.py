"""Configuration settings for Legal RAG Tester."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # === Pinecone ===
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "")
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "")
        self.pinecone_top_k = int(os.getenv("PINECONE_TOP_K", "10"))
        self.pinecone_final_k = int(os.getenv("PINECONE_FINAL_K", "5"))
        self.pinecone_score_threshold = float(os.getenv("PINECONE_SCORE_THRESHOLD", "0.75"))

        # === Embedding model ===
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.embedding_prefix = os.getenv("EMBEDDING_PREFIX", "query:")

        # === LLM models to test ===
        llm_models_str = os.getenv("LLM_MODELS", "")
        self.llm_models = [m.strip() for m in llm_models_str.split(",")] if llm_models_str else []
        self.hf_token = os.getenv("HF_TOKEN", "")

        # === Groq API ===
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.groq_rpm_delay = float(os.getenv("GROQ_RPM_DELAY", "2.1"))

        # === OpenRouter API ===
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_site_url = os.getenv("OPENROUTER_SITE_URL", "https://legalrag.kz")
        self.openrouter_app_name = os.getenv("OPENROUTER_APP_NAME", "LegalRAG-Tester")
        self.scorer_model = os.getenv("SCORER_MODEL", "meta-llama/llama-3.3-70b-instruct")

        # === Google API ===
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")

        # === OpenAI API ===
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # === IO ===
        self.input_excel = os.getenv("INPUT_EXCEL", "questions.xlsx")
        self.input_sheet = os.getenv("INPUT_SHEET", "Sheet1")
        self.question_column = os.getenv("QUESTION_COLUMN", "question")
        self.id_column = os.getenv("ID_COLUMN", "id")
        self.output_dir = os.getenv("OUTPUT_DIR", "results")

        # === Pipeline behavior ===
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "2.0"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.chunk_text_field = os.getenv("CHUNK_TEXT_FIELD", "text")
        self.verbose = os.getenv("VERBOSE", "false").lower() == "true"
        self.query_rewriter_model = os.getenv("QUERY_REWRITER_MODEL", "llama-3.1-8b-instant")
        self.enable_query_rewriting = os.getenv("ENABLE_QUERY_REWRITING", "true").lower() == "true"

# Create global settings instance
settings = Settings()
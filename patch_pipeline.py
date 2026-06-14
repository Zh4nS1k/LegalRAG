import sys
import re

with open("LLM_MODELS_TEST/legal_rag_tester/pipeline.py", "r") as f:
    content = f.read()

# 1. Imports
content = re.sub(r'from embedder import Embedder\n', '', content)
content = re.sub(r'from retriever import PineconeRetriever\n', '', content)
content = re.sub(r'from prompt_builder import PromptBuilder\n', '', content)
content = re.sub(r'from query_rewriter import QueryRewriter\n', '', content)

# 2. __init__
content = re.sub(r'        self\.embedder = Embedder\(\)\n', '', content)
content = re.sub(r'        self\.retriever = PineconeRetriever\(settings\)\n', '', content)
content = re.sub(r'        self\.prompt_builder = PromptBuilder\(\)\n', '', content)
content = re.sub(r'        self\.query_rewriter = QueryRewriter\(\)\n', '', content)

# 3. Add health check
health_check = """
    def _check_ai_service(self) -> bool:
        import requests
        import os
        try:
            r = requests.get(
                f"{os.getenv('AI_SERVICE_URL', 'http://localhost:8000')}/health",
                timeout=5
            )
            return r.status_code < 500
        except Exception:
            return False

    def run(self,"""
content = content.replace('    def run(self,', health_check)

# Add health check call in run
health_check_call = """        # ── Graceful signal handlers ────────────────────────────────────
        def _signal_handler(sig, frame):
            pipeline_logger.log_warning(
                f"⌨️  Interrupted! Saving {len(checkpoint.rows)} rows..."
            )
            checkpoint.finalize()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        if not self._check_ai_service():
            pipeline_logger.log_error("pipeline", "health", "❌ ai_service не запущен. Запусти: uvicorn ai_service.api.api:app --port 8000")
            sys.exit(1)"""
content = content.replace('        # ── Graceful signal handlers ────────────────────────────────────\n        def _signal_handler(sig, frame):\n            pipeline_logger.log_warning(\n                f"⌨️  Interrupted! Saving {len(checkpoint.rows)} rows..."\n            )\n            checkpoint.finalize()\n            sys.exit(0)\n\n        signal.signal(signal.SIGINT, _signal_handler)\n        signal.signal(signal.SIGTERM, _signal_handler)', health_check_call)

# 4. Remove chunk logic
content = re.sub(r'                rewritten_query = self\.query_rewriter\.rewrite\(question\.text\)\n', '', content)
content = re.sub(r'                pipeline_logger\.log_simple_info\(f"🔍 Original: \{question\.text\[:100\]\}\.\.\."\)\n', '', content)
content = re.sub(r'                pipeline_logger\.log_simple_info\(f"🔍 Rewritten: \{rewritten_query\}"\)\n', '', content)
content = re.sub(r'                with StepTimer\("embed"\) as t_embed:\n                    embedding = self\.embedder\.embed\(rewritten_query\)\n                pipeline_logger\.log_timing\("embed", t_embed\.elapsed_ms\)\n', '', content)
content = re.sub(r'                with StepTimer\("retrieve"\) as t_retrieve:\n                    chunks = self\.retriever\.query\(embedding, query_text=question\.text\)\n                pipeline_logger\.log_retrieval\(question\.id, chunks, t_retrieve\.elapsed_ms\)\n', '', content)
content = re.sub(r'                if len\(chunks\) < 2:.*?                    \)\n', '', content, flags=re.DOTALL)
content = re.sub(r'                prompt = self\.prompt_builder\.build\(question\.text, chunks\)\n', '', content)
content = re.sub(r'                prompt_token_count = self\.token_counter\.count_prompt\(prompt\["system"\], prompt\["user"\]\)\n', '                prompt_token_count = 0\n', content)
content = re.sub(r'                pipeline_logger\.log_simple_info\(.*?                \)\n', '', content, flags=re.DOTALL)

# 5. Modify call
content = content.replace(
    'result = self.llm_client.call(\n                                model_name, prompt["system"], prompt["user"], q_id=question.id\n                            )',
    'result = self.llm_client.call(\n                                model_name, question.text, q_id=question.id\n                            )'
)

# 6. Adjust metrics assignments
metrics_rm = """                    result.chunks_used = len(chunks)
                    result.retrieved_scores = [c.score for c in chunks]
                    result.avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
                    result.embed_ms = t_embed.elapsed_ms
                    result.retrieve_ms = t_retrieve.elapsed_ms
                    result.llm_ms = t_llm.elapsed_ms
                    result.total_ms = t_embed.elapsed_ms + t_retrieve.elapsed_ms + t_llm.elapsed_ms"""
content = content.replace(metrics_rm, "                    result.llm_ms = t_llm.elapsed_ms\n                    result.total_ms = result.latency_ms")

# Adjust answer_scorer params
content = content.replace(
    'question.text, prompt["user"], result.answer, question.id',
    'question.text, question.text, result.answer, question.id'
)

# Adjust logger params
content = content.replace('prompt["system"], prompt["user"]', '"" , question.text')

content = content.replace('t_embed.elapsed_ms + t_retrieve.elapsed_ms + total_llm_ms', 'total_llm_ms')

with open("LLM_MODELS_TEST/legal_rag_tester/pipeline.py", "w") as f:
    f.write(content)
